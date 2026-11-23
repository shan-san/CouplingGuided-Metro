"""Bounded, label-blind guidance queries shared by SFT and local inference.

Each query contains one complete decision group's candidates. Other trains and
queues are contextual features, not a replacement for the exact physical model.
The solver retains the entire catalog and checks joint feasibility independently.
"""
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
import argparse
import hashlib
import json
import sys

from LLM.finetune.prepare_sft import dumps, runs
from LLM.finetune.chat_format import render_chat

SCHEMA = "metro_guidance_group_v2"
RELATION_INSTRUCTION = (
    "coupling.relationships may be {\"sets\":[{\"relations\":[...],\"groups\":[...]}]}; "
    "each listed group has exactly the listed relation types. This factors repeated lists; no peer is omitted. "
)
SYSTEM = (
    "Return JSON with warm_start, variable_fixing and branching_priority, keyed only by "
    "the supplied solver_variables. Starts/fixes are binary; priorities are nonnegative integers. "
    "This is metro_guidance_group_v2: one decision group with its complete candidates and "
    "a summary of public operating context. Select at most one positive start per group. "
    "Do not output dispatch actions. Missing variables are unspecified, not zero. "
    "path_runs are lossless [start,step,count] integer-node runs. Times and durations are seconds; "
    "dt_s is relative to time_s. This compatible group_v2 extension permits each vehicle's "
    "actions/pauses value to be a list or {\"runs\":[[start,step,count],...]}; concatenate "
    "start+step*i for i=0..count-1 per run. Decoded actions[k] is the action at zero-based "
    "stage k; decoded pauses lists paused stage indexes, not durations. "
    "events is a list or {\"shared\":{...},\"rows\":[{...},...]}; merge shared into each row "
    "to recover every event in order, including duplicates and all fields. "
    "Coupling summaries and peer features are not full capacity constraints; joint feasibility "
    "and any guidance restrictions are checked by the unchanged optimization/execution model. "
    "Only applicability_zero=true variables may receive zero-fixing suggestions."
)
LEGACY_SYSTEM = SYSTEM
SYSTEM = SYSTEM + " " + RELATION_INSTRUCTION + (
    "solver_variable_defaults supplies omitted fields of every solver_variables entry; "
    "solver_command_defaults supplies omitted fields of each entry's commands. "
    "Explicit fields override defaults; all candidates and parameter values are retained."
)
FIELDS = ("warm_start", "variable_fixing", "branching_priority")
PATH_FIELDS = {"path", "path_node_ids", "approach_node_ids", "tow_node_ids"}


def packed(value, field=None):
    if isinstance(value, dict):
        return {k: packed(v, k) for k, v in value.items() if k not in ("command_id", "version")}
    if isinstance(value, (list, tuple)):
        if field in PATH_FIELDS and all(type(n) is int for n in value):
            return {"path_runs": runs(value)}
        return [packed(v) for v in value]
    return value


def stage_controls(controls):
    """Keep every control value; compact integer sequences only when smaller."""
    result = {}
    for vehicle, sequence in controls.items():
        plain = packed(sequence)
        if isinstance(sequence, (list, tuple)) and all(type(v) is int for v in sequence):
            compact = {"runs": runs(sequence)}
            if len(dumps(compact)) < len(dumps(plain)):
                plain = compact
        result[vehicle] = plain
    return result


def public_events(events):
    """Factor exact common fields without selecting or merging event records."""
    plain = deepcopy(events)
    if not isinstance(plain, list) or len(plain) < 2 or not all(isinstance(e, dict) for e in plain):
        return plain
    # JSON equality distinguishes bool/int/float and missing keys from null.
    shared = {key: value for key, value in plain[0].items()
              if all(key in event and dumps(event[key]) == dumps(value) for event in plain[1:])}
    if not shared:
        return plain
    compact = dict(shared=shared, rows=[{k: v for k, v in event.items() if k not in shared} for event in plain])
    return compact if len(dumps(compact)) < len(dumps(plain)) else plain


def relationship_map(value):
    """Decode the optional exact relation-list factoring, including legacy maps."""
    if not (isinstance(value, dict) and set(value) == {"sets"}):
        return value
    result = {}
    for row in value["sets"]:
        for group in row["groups"]:
            if group in result:
                raise ValueError("Repeated coupling peer")
            result[group] = list(row["relations"])
    return result


def compact_relationships(value):
    buckets = defaultdict(list)
    for group, relations in value.items():
        buckets[tuple(relations)].append(group)
    compact = dict(sets=[dict(relations=list(relations), groups=groups)
                        for relations, groups in buckets.items()])
    return compact if len(dumps(compact)) < len(dumps(value)) else value


def expand_query(query):
    """Canonical complete query for checking old and compact wire encodings."""
    result = dict(query)
    defaults = result.pop("solver_variable_defaults", {})
    command_defaults = result.pop("solver_command_defaults", {})
    if defaults or command_defaults:
        result["solver_variables"] = {name: dict(defaults, **value) for name, value in query["solver_variables"].items()}
        for value in result["solver_variables"].values():
            value["commands"] = [dict(command_defaults, **c) for c in value["commands"]]
    if "relationships" in query.get("coupling", {}):
        result["coupling"] = dict(query["coupling"], relationships=relationship_map(query["coupling"]["relationships"]))
    return result


def compact_candidates(query):
    """Factor repeated fields, never prune options or round their parameters."""
    variables = query["solver_variables"]
    defaults = dict(actions={}, pauses={}, applicability_zero=False)
    commands = [c for value in variables.values() for c in value["commands"]]
    common = {key: commands[0][key] for key in ("vehicle_id", "expected_service_id", "dt_s")
              if commands and key in commands[0] and all(
                  key in c and dumps(c[key]) == dumps(commands[0][key]) for c in commands)}
    compact = dict(query, solver_variable_defaults=defaults, solver_command_defaults=common,
        solver_variables={name: {k: ([{f: x for f, x in c.items() if f not in common} for c in v]
            if k == "commands" else v) for k, v in value.items()
            if k not in defaults or dumps(v) != dumps(defaults[k])} for name, value in variables.items()})
    return compact if len(dumps(compact)) < len(dumps(query)) else query


def group_queries(payload, *, context_peers=4):
    """No labels are consulted when choosing groups, candidates or context."""
    state = payload["state"]
    snapshot = state["dispatch_state"]
    vehicles = snapshot["vehicles"]
    now = snapshot["time"]
    groups = defaultdict(dict)
    for name, choice in payload["solver_variables"].items():
        groups[choice["group"]][name] = choice
    queues = state.get("queues", {})
    totals = {side: sum(float(v) for q in queues.values() for v in q[side].values())
              for side in ("inside", "outside")}
    totals.update(onboard=sum(v["passenger_count"] for v in vehicles.values()),
                  vehicle_status_counts=dict(Counter(v["status"] for v in vehicles.values())))
    pressure = sorted(queues, key=lambda p: -sum(float(v) for side in queues[p].values() for v in side))[:4]
    events = public_events(state.get("disruptions", []))
    evidence = defaultdict(list)
    for row in state.get("coupling", {}).get("evidence", []):
        for group in row.get("groups", []):
            evidence[group].append(row)
    for group, catalog in groups.items():
        related = set()
        relationships = defaultdict(set)
        relation_counts, event_ids, measure_counts = Counter(), set(), Counter()
        for row in evidence[group]:
            related.update(row.get("groups", []))
            for peer in row.get("groups", []):
                if peer != group:
                    relationships[peer].update(row.get("relations", []))
            relation_counts.update(row.get("relations", []))
            event_ids.update(row.get("affected_event_ids", []))
            measure_counts.update(row.get("action_families", []))
        related.discard(group)
        peers = sorted(related)
        local_platforms = set()
        if group.startswith("platform_"):
            local_platforms.add(group.removeprefix("platform_"))
        if group in vehicles:
            local_platforms.add(str(vehicles[group]["position"]))
        candidates, participants, service_ids = {}, set(), set()
        for name, choice in catalog.items():
            commands = []
            for raw in choice["commands"]:
                c = packed(raw)
                c["dt_s"] = c.pop("effective_at") - now
                commands.append(c)
                params = raw.get("parameters", {})
                participants.update(v for v in (raw.get("vehicle_id"), params.get("failed")) if v in vehicles and v != group)
                if params.get("service_id") is not None:
                    service_ids.add(params["service_id"])
                p = params.get("platform_id")
                if p is not None:
                    local_platforms.add(str(p))
            candidates[name] = dict(option=choice["option"], commands=commands,
                actions=stage_controls(choice.get("actions", {})), pauses=stage_controls(choice.get("pauses", {})),
                applicability_zero=bool(choice.get("applicability_zero", False)))
            if choice.get("control_measures"):
                candidates[name].update(control_measures=choice["control_measures"], control_basis=choice["control_basis"])
        operations = [packed(op) for op in snapshot.get("operations", {}).values()
                      if group in (op.get("vehicle_id"), op.get("partner"))]
        query = dict(encoding=SCHEMA, time_s=now, timing=state.get("timing", {}), group=group,
            specification=payload["specification"], vehicle=packed(vehicles.get(group)),
            agent_clock=next((packed(a) for v, a in zip(vehicles, state.get("agent_positions", [])) if v == group), None),
            operations=operations,
            participants={v: packed(vehicles[v]) for v in sorted(participants)},
            service=packed(snapshot.get("services", {}).get(vehicles.get(group, {}).get("service_id"))),
            candidate_services={s: packed(snapshot.get("services", {}).get(s)) for s in sorted(service_ids)},
            depot_inventory={d: len(v) for d, v in snapshot.get("inventory", {}).items()},
            queues={p: packed(queues[p]) for p in sorted(local_platforms) if p in queues},
            queue_pressure={p: {side: sum(float(v) for v in q.values()) for side, q in queues[p].items()} for p in pressure},
            events=events, global_summary=totals,
            coupling=dict(relations=dict(relation_counts), event_ids=sorted(event_ids),
                          relationships=compact_relationships({p: sorted(v) for p, v in sorted(relationships.items())}),
                          measure_counts=dict(measure_counts), related_groups=len(peers),
                          peers=[dict(group=p, vehicle=packed({k: v for k, v in vehicles.get(p, {}).items()
                                if k in ("position", "status", "depot", "service_id", "failed", "passenger_count")}))
                                 for p in peers[:context_peers]],
                          omitted_peer_details=max(0, len(peers)-context_peers)),
            solver_variables=candidates)
        yield group, compact_candidates(query)


def inference_payload(payload):
    """Drop transport-only fields without changing any group_queries output.

    In particular, repeated physical node lists and RAG source snapshots are not
    read by group_queries. Keep the complete candidates and every relation row.
    """
    state = payload["state"]
    snapshot = state["dispatch_state"]
    compact = {k: state[k] for k in ("agent_positions", "queues", "timing", "disruptions") if k in state}
    compact["dispatch_state"] = {k: snapshot[k] for k in (
        "vehicles", "time", "operations", "services", "inventory") if k in snapshot}
    compact["coupling"] = dict(evidence=[{k: row[k] for k in (
        "groups", "relations", "affected_event_ids", "action_families") if k in row}
        for row in state.get("coupling", {}).get("evidence", [])])
    return dict(state=compact, specification=payload["specification"], solver_variables=payload["solver_variables"])


def chat_tokens(tokenizer, messages, *, generation=False):
    text = render_chat(tokenizer, messages, generation=generation)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def group_messages(query, labels=None):
    messages = [dict(role="system", content=SYSTEM), dict(role="user", content=dumps(query))]
    if labels is not None:
        messages.append(dict(role="assistant", content=dumps(labels)))
    return messages


def validate_guidance(query, guidance):
    query = expand_query(query)
    if set(guidance) != set(FIELDS):
        raise ValueError("Wrong guidance fields")
    variables = query["solver_variables"]
    for field, values in guidance.items():
        if not isinstance(values, dict) or not set(values) <= set(variables):
            raise ValueError("Guidance references variables outside this query")
        if any(type(v) is not int or v < 0 or (field != "branching_priority" and v not in (0, 1))
               for v in values.values()):
            raise ValueError("Invalid guidance value")
    if sum(guidance["warm_start"].values()) > 1:
        raise ValueError("Conflicting one-hot starts")
    if any(v != 0 or not variables[n]["applicability_zero"] for n, v in guidance["variable_fixing"].items()):
        raise ValueError("Fixing lacks a supplied immediate-applicability exclusion")


def merge_guidance(queries, predictions):
    """Only merge validated, nonoverlapping group predictions."""
    merged = {f: {} for f in FIELDS}
    seen = set()
    if len(queries) != len(predictions):
        raise ValueError("Missing group prediction")
    for query, prediction in zip(queries, predictions):
        validate_guidance(query, prediction)
        names = set(query["solver_variables"])
        if names & seen:
            raise ValueError("Repeated variable group")
        seen |= names
        for field in FIELDS:
            merged[field].update(prediction[field])
    return merged


def teacher_group_labels(query, guidance, selected):
    """Recover one complete teacher label, including legacy safe exclusions."""
    query = expand_query(query)
    group, choices = query["group"], query["solver_variables"]
    chosen = [n for n, choice in choices.items() if choice["option"] == selected.get(group)]
    if len(chosen) != 1:
        raise ValueError(f"Missing/ambiguous complete teacher selection for {group}")
    name = chosen[0]
    fixes = {n: 0 for n, v in choices.items() if v["applicability_zero"] and n != name}
    # Older records only persisted a subset of safe zero exclusions.
    for n, v in guidance.get("variable_fixing", {}).items():
        if n in choices and v == 0 and n != name:
            choices[n]["applicability_zero"] = True
            fixes[n] = 0
    labels = dict(warm_start={name: 1}, variable_fixing=dict(list(fixes.items())[:8]),
                  branching_priority={name: 10})
    validate_guidance(query, labels)
    return labels, name, choices[name]["option"] not in ("continue", "unchanged")


def training_views(payload, guidance, selected, *, tokenizer, max_tokens=8192, continuation_groups=2):
    """Reconstruct full selected starts, bound length, and limit tied continuations.

    Output omission is recorded, never hidden by tokenizer truncation. All changed
    decisions survive unless the complete group itself exceeds the token budget.
    """
    if tokenizer is None or max_tokens < 1 or continuation_groups < 0:
        raise ValueError("A tokenizer and valid length/continuation limits are required")
    kept, rejected, continuation_count = [], [], 0
    queries = list(group_queries(payload))
    # Rotate unchanged examples without relying on labels or group position.
    queries.sort(key=lambda pair: hashlib.sha256(
        f"{payload['state']['dispatch_state']['time']}:{pair[0]}".encode()).hexdigest())
    for group, query in queries:
        labels, name, changed = teacher_group_labels(query, guidance, selected)
        unchanged = not changed
        if unchanged and continuation_count >= continuation_groups:
            continue
        if unchanged:
            continuation_count += 1
        messages = group_messages(query, labels)
        length = chat_tokens(tokenizer, messages)
        reserved = chat_tokens(tokenizer, messages[:2], generation=True) + 512
        if max(length, reserved) > max_tokens:
            rejected.append(dict(group=group, tokens=length, prompt_plus_generation=reserved,
                                 reason="complete_group_overlength"))
        else:
            kept.append(dict(group=group, messages=messages, tokens=length,
                             changed=not unchanged, selected_variable=name))
    return kept, rejected


def serve_jsonl(tokenizer, *, max_tokens=4096, continuation_groups=2,
                input_stream=None, output_stream=None):
    """Keep one local tokenizer alive; stdout is exclusively the JSONL protocol."""
    if max_tokens < 1 or continuation_groups < 0:
        raise ValueError("Invalid length/continuation limit")
    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stdout if output_stream is None else output_stream
    identity = str(tokenizer.name_or_path)

    def emit(value):
        print(dumps(value), file=output_stream, flush=True)

    emit(dict(event="tokenizer_ready", tokenizer=identity))
    for line in input_stream:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            with redirect_stdout(sys.stderr):
                views, rejected = training_views(request["payload"], request["guidance"], request["selected"],
                    tokenizer=tokenizer, max_tokens=max_tokens, continuation_groups=continuation_groups)
            emit(dict(views=views, rejected=rejected, tokenizer=identity))
        except Exception as exc:
            emit(dict(error=str(exc) or type(exc).__name__, type=type(exc).__name__))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Persistent local-tokenizer JSONL guidance worker")
    parser.add_argument("--tokenizer", required=True, help="Local tokenizer directory; no model weights loaded")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--continuation-groups", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        if args.max_tokens < 1 or args.continuation_groups < 0:
            raise ValueError("Invalid length/continuation limit")
        with redirect_stdout(sys.stderr):
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, fix_mistral_regex=True)
        serve_jsonl(tokenizer, max_tokens=args.max_tokens, continuation_groups=args.continuation_groups)
    except Exception as exc:
        print(dumps(dict(error=str(exc) or type(exc).__name__, type=type(exc).__name__)), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
