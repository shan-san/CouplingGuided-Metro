"""Source-only retrieval and ordinary supervised guidance for native commands.

No vehicle, node or solver-variable ID is transferred between topologies.
All suggested assignments still require the shared trajectory oracle.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from delay_events import EVENT_TARGETS
from dispatch_plan import PARAMETERS
from dispatch_candidates import applicable_options

FAMILIES = tuple(PARAMETERS)
EVENTS = tuple(sorted(EVENT_TARGETS))
FEATURE_VERSION = "metro_baseline_semantic_v2_seconds"
FEATURE_DIM = len(FAMILIES)+len(EVENTS)+14


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_features(state):
    queues = list(state.get("waiting_passengers", {}).values())
    vehicles = list(state.get("dispatch_state", {}).get("vehicles", {}).values())
    events = [e for e in state.get("disruptions", []) if e.get("active", True)]
    return [float(any(e["type"] == kind for e in events)) for kind in EVENTS] + [
        min(10., sum(queues)/max(1, len(queues))/1000.), min(10., max(queues, default=0.)/1000.),
        sum(bool(v.get("failed")) for v in vehicles)/max(1, len(vehicles)),
        sum(v.get("depot") is not None for v in vehicles)/max(1, len(vehicles)),
        sum(float(v.get("passenger_count", 0.)) for v in vehicles)/max(1, len(vehicles))/2520.,
        len(state.get("dispatch_state", {}).get("operations", {}))/max(1, len(vehicles))]


def option_features(state, entry, horizon):
    commands = entry.get("commands", [])
    families = {c["action"] for c in commands}
    events = [e for e in state.get("disruptions", []) if e.get("active", True)]
    group = entry.get("group", "")
    vehicles = state.get("dispatch_state", {}).get("vehicles", {})
    ids = {c.get("vehicle_id") for c in commands} | {group}
    ids |= {c.get("parameters", {}).get("failed") for c in commands}
    records = [v for key, v in vehicles.items() if key in ids]
    params = [c.get("parameters", {}) for c in commands]
    platforms = {p["platform_id"] for p in params if "platform_id" in p}
    platforms |= {v.get("position") for v in records}
    waiting = state.get("waiting_passengers", {})
    queue = sum(float(waiting.get(str(p), waiting.get(p, 0.))) for p in platforms)
    outside = sum(sum(float(v) for v in queues.get("outside", {}).values())
                  for queues in state.get("queues", {}).values())
    operations = state.get("dispatch_state", {}).get("operations", {}).values()
    busy = any(o.get("vehicle_id") in ids or o.get("partner") in ids for o in operations)
    def maximum(key):
        return max((float(p.get(key, 0.)) for p in params), default=0.)
    noop = not commands and not entry.get("actions") and not entry.get("pauses") and entry.get("option") not in ("wait_window", "defer_movement")
    extra = [float(group.startswith("platform_")), float(noop), min(1., len(commands)/3.),
        min(10., sum(float(v.get("passenger_count", 0.)) for v in records)/max(1, len(records))/2520.),
        float(any(v.get("failed") for v in records)), float(any(v.get("depot") is not None for v in records)),
        float(busy), min(10., queue/1000.), min(10., outside/max(1, len(waiting))/1000.),
        min(10., maximum("seconds")/300.), min(10., maximum("headway_seconds")/300.),
        min(10., maximum("rate")/10.),
        min(1., max((len(p.get("path_node_ids", [])) for p in params), default=0)/max(1, state.get("n_nodes", 1))),
        min(10., horizon/12.)]
    result = ([float(kind in families) for kind in FAMILIES]
              + [float(any(e["type"] == kind for e in events)) for kind in EVENTS] + extra)
    if len(result) != FEATURE_DIM or not all(math.isfinite(v) for v in result):
        raise ValueError("Nonfinite or incompatible baseline features")
    return result


def model_schema(problem):
    return {f"command_{gi}_{oi}": dict(group=g, option=o.id, commands=o.commands,
                                      actions=o.actions, pauses=o.pauses)
            for gi, (g, options) in enumerate(problem.groups.items()) for oi, o in enumerate(options)}


def control_signature(entry):
    """Keep continuation, primitive waiting and explicit measures distinct."""
    if entry.get("commands"):
        return tuple(c["action"] for c in entry["commands"])
    if entry.get("pauses") or entry.get("option") == "defer_movement":
        return ("defer_movement",)
    actions = entry.get("actions", {})
    if entry.get("option") == "wait_window" or actions and all(not a for seq in actions.values() for a in seq):
        return ("wait_window",)
    return ("primitive_control",) if actions else ("continuation",)


def load_sft_cases(path, source_topology, split="train"):
    cases, seen = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row.get("metadata", {})
            if meta.get("split") != split or meta.get("source_topology") != source_topology:
                raise ValueError("Guidance data must match the declared source topology and split")
            uid = meta.get("case_id")
            episode = meta.get("episode_id") or uid
            if not uid or uid in seen or any(f"combination_{key}" in uid or f"combination_{key}" in episode
                                            for key in ("B", "D", "F")):
                raise ValueError("Duplicate, unidentified or held-out test combination in preparation data")
            seen.add(uid)
            messages = row["messages"]
            payload = json.loads(next(m["content"] for m in messages if m["role"] == "user"))
            label = json.loads(next(m["content"] for m in messages if m["role"] == "assistant"))
            state, schema = payload["state"], payload["solver_variables"]
            if state.get("timing", {}).get("schema") != "native_seconds_v1":
                raise ValueError("Guidance labels must be regenerated for native_seconds_v1; old 30-second labels are incompatible")
            horizon = int(payload["specification"]["planning_horizon"])
            groups = {}
            for name, entry in schema.items():
                groups.setdefault(entry["group"], []).append((name, option_features(state, entry, horizon)))
            samples, selected, selected_records = [], [], []
            warm = label.get("warm_start", {})
            for group, values in groups.items():
                choices = [i for i, (name, _) in enumerate(values) if warm.get(name) == 1]
                if len(choices) > 1:
                    raise ValueError("Teacher contains multiple positive assignments for one group")
                # Partial SFT labels do not make every omitted group negative.
                if choices:
                    samples.append(dict(features=[v for _, v in values], target=choices[0]))
                    selected.append(values[choices[0]][1])
                    name = values[choices[0]][0]
                    selected_records.append(dict(features=values[choices[0]][1],
                        signature=control_signature(schema[name])))
            if samples:
                cases.append(dict(case_id=uid, episode_id=episode, source_topology=source_topology,
                                  embedding=state_features(state), selected=selected, samples=samples,
                                  selected_records=selected_records))
    if not cases:
        raise ValueError("No qualified labelled decision groups in source data")
    return cases


class RetrievalGuidance:
    def __init__(self, path, source_topology):
        self.cases = load_sft_cases(path, source_topology)
        self.provenance = dict(kind="source_sft_retrieval", source_topology=source_topology,
                               artifact_sha256=file_hash(path), cases=len(self.cases), trained=False,
                               mapping_protocol="one_to_one_semantic_groups_v2")

    def propose(self, problem, state):
        vector = state_features(state)
        distance = lambda other: sum((a-b)**2 for a, b in zip(vector, other))
        case = min(self.cases, key=lambda c: (distance(c["embedding"]), c["case_id"]))
        masks, _ = applicable_options(problem)
        selection, scores, pairs = [], [], []
        records = case["selected_records"]
        for gi, (g, options) in enumerate(problem.groups.items()):
            candidates = [option_features(state, dict(group=g, option=o.id, commands=o.commands,
                          actions=o.actions, pauses=o.pauses), problem.horizon) for o in options]
            selection.append(next((oi for oi, o in enumerate(options)
                if not o.commands and not o.actions and not o.pauses), 0))
            scores.append(0.)
            for si, record in enumerate(records):
                if record["features"][-14] != float(g.startswith("platform_")):
                    continue
                for oi, (op, features) in enumerate(zip(options, candidates)):
                    signature = control_signature(dict(option=op.id, commands=op.commands,
                                                       actions=op.actions, pauses=op.pauses))
                    if masks[g][oi] and signature == record["signature"]:
                        pairs.append((sum((a-b)**2 for a, b in zip(features, record["features"])), gi, si, oi))
        # A retrieved teacher plan supplies decisions, not an unlimited supply
        # of prototypes. One held train must not become a certain hold for an
        # entire fleet whose features happen to be equal.
        used_groups, used_source, ambiguous = set(), set(), 0
        for d, gi, si, oi in sorted(pairs):
            if gi in used_groups or si in used_source:
                continue
            alternative = min((v for v, gj, sj, oj in pairs
                               if gj == gi and oj != oi or sj == si and gj != gi), default=math.inf)
            margin = 1. if not math.isfinite(alternative) else max(0., alternative-d)/(1.+alternative)
            selection[gi], scores[gi] = oi, margin/(1.+d)
            ambiguous += int(margin <= 1e-12)
            used_groups.add(gi)
            used_source.add(si)
        return tuple(selection), scores, dict(case_id=case["case_id"], distance=distance(case["embedding"]),
            mapping="one-to-one semantic groups; no source IDs copied; applicability masks; ambiguity margin",
            mapped_groups=len(used_groups), unmatched_groups=len(selection)-len(used_groups),
            ambiguous_groups=ambiguous, source_group_reuse=False,
            confidence_basis="heuristic similarity and assignment margin, not calibrated probabilities")


class GuidanceMLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, value):
        return self.network(value).squeeze(-1)


class LearnedGuidance:
    def __init__(self, path, source_topology):
        data = torch.load(path, map_location="cpu", weights_only=True)
        meta = data.get("metadata", {})
        if (data.get("kind") != "metro_guidance_mlp_v1" or data.get("feature_version") != FEATURE_VERSION
                or meta.get("source_topology") != source_topology or not meta.get("training_complete")):
            raise ValueError("Missing, untrained or incompatible source-only guidance checkpoint")
        self.model = GuidanceMLP(data["hidden"])
        self.model.load_state_dict(data["weights"])
        self.model.eval()
        self.provenance = dict(meta, artifact_sha256=file_hash(path), trained=True)

    def propose(self, problem, state):
        selection, confidences = [], []
        masks, _ = applicable_options(problem)
        with torch.no_grad():
            for g, options in problem.groups.items():
                features = [option_features(state, dict(group=g, option=o.id, commands=o.commands,
                            actions=o.actions, pauses=o.pauses), problem.horizon) for o in options]
                logits = self.model(torch.tensor(features, dtype=torch.float32))
                probabilities = logits.masked_fill(~torch.tensor(masks[g], dtype=torch.bool), -math.inf).softmax(0)
                index = int(probabilities.argmax())
                selection.append(index); confidences.append(float(probabilities[index]))
        return tuple(selection), confidences, dict(feature_version=FEATURE_VERSION)
