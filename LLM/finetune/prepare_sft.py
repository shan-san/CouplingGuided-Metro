"""Prepare bounded native-guidance SFT data and recover full teacher selections.

This is an offline data view, not a change to simulation or the MILP domain.
Keep the original solver records for replay. Group-v2 deployment uses the same
group_queries encoding; --view lossless retains the historical compact-v1 view.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topology.metro_lines import LAYOUT_REVISION


SCHEMA = "metro_guidance_compact_v1"
LINE_IDS = {"sz20": 101, "sz1": 102, "sz10": 103, "sz6b": 104, "sh16": 105}
GUIDANCE_FIELDS = {"warm_start", "variable_fixing", "branching_priority"}
SYSTEM = (
    "Return JSON with warm_start, variable_fixing and branching_priority. "
    "Use only the original variable names in solver_variables; starts/fixes are binary, "
    "priorities are nonnegative integers. Do not generate dispatch actions. "
    "The input uses metro_guidance_compact_v1. solver_variables maps each binary variable "
    "to a choice_catalog index; the g in command_g_i is its candidate_groups index. "
    "In a choice command, $group, $position and $service refer to that group's current "
    "vehicle identity, position and service. dt_s is relative to state.time_s. "
    "path_ref indexes path_catalog: each [start,step,count] run expands to integer nodes. "
    "Tables have named columns, optional constants and row keys. They contain rows or "
    "column_data: each column has values or runs of [value,repetition_count]. "
    "In coupling.evidence, group_indexes indexes candidate_groups, resource_set indexes "
    "resource_sets ([first,last] inclusive node ranges), and set_columns index value_sets. "
    "Conflict second_runs preserve exact seconds within each reference stage. "
    "OD tables omit zero entries; absent values are zero. Historical completed command "
    "details are summarized, while current operations and unfinished commands are retained. "
    "Passenger counts are input features rounded to 0.1 persons and passenger waiting mass "
    "to whole person-seconds; exact source states, command parameters and labels are unchanged."
)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def passenger_feature(value, digits=1):
    value = round(value, digits)
    return int(value) if value == int(value) else value


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read_records(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def runs(values):
    """Lossless arithmetic runs; the integer-second node list is not coarsened."""
    values = list(values)
    if any(type(v) is not int for v in values):
        raise ValueError("Node/second runs require integers")
    result, i = [], 0
    while i < len(values):
        if i + 1 == len(values):
            result.append([values[i], 0, 1])
            break
        step = values[i + 1] - values[i]
        j = i + 2
        while j < len(values) and values[j] - values[j - 1] == step:
            j += 1
        result.append([values[i], step, j - i])
        i = j
    return result


def expand_runs(encoded):
    return [start + step * i for start, step, count in encoded for i in range(count)]


def ranges(values):
    values = list(values)
    if values != sorted(set(values)) or any(type(v) is not int for v in values):
        raise ValueError("Resource sets must be sorted unique integer nodes")
    result = []
    for value in values:
        if result and value == result[-1][1] + 1:
            result[-1][1] = value
        else:
            result.append([value, value])
    return result


def table(records, keys=None):
    if not records:
        return {"columns": [], "rows": []}
    columns = list(dict.fromkeys(key for row in records for key in row))
    constants = {key: records[0][key] for key in columns if key in records[0]
                 and all(key in row and row[key] == records[0][key] for row in records)}
    columns = [key for key in columns if key not in constants]
    result = {"columns": columns, "rows": [[row.get(key) for key in columns] for row in records]}
    if constants:
        result["constants"] = constants
    if keys is not None:
        result["keys"] = list(keys)
    if len(records) > 4:
        data = []
        for column in columns:
            values = [row.get(column) for row in records]
            repeated = []
            for value in values:
                if repeated and repeated[-1][0] == value:
                    repeated[-1][1] += 1
                else:
                    repeated.append([value, 1])
            plain, compressed = {"values": values}, {"runs": repeated}
            data.append(compressed if len(dumps(compressed)) < len(dumps(plain)) else plain)
        columnar = {k: v for k, v in result.items() if k != "rows"}
        columnar.update(row_count=len(records), column_data=data)
        if len(dumps(columnar)) < len(dumps(result)):
            return columnar
    return result


class Encoder:
    def __init__(self, payload):
        self.payload = payload
        self.state = payload["state"]
        self.snapshot = self.state["dispatch_state"]
        self.vehicles = self.snapshot["vehicles"]
        self.now = self.snapshot["time"]
        self.groups = list(dict.fromkeys(v["group"] for v in payload["solver_variables"].values()))
        self.group_index = {group: i for i, group in enumerate(self.groups)}
        self.path_catalog, self.path_index = [], {}
        self.resource_sets, self.resource_index = [], {}
        self.value_sets, self.value_index = [], {}

    def path(self, nodes):
        key = tuple(nodes)
        if key not in self.path_index:
            encoded = runs(nodes)
            if expand_runs(encoded) != list(nodes):
                raise AssertionError("Path encoding changed physical nodes")
            self.path_index[key] = len(self.path_catalog)
            self.path_catalog.append(encoded)
        return {"path_ref": self.path_index[key]}

    def packed(self, value, field=None):
        if isinstance(value, dict):
            return {key: self.packed(v, key) for key, v in value.items()}
        if isinstance(value, list):
            if field in ("path", "path_node_ids", "approach_node_ids", "tow_node_ids"):
                return self.path(value)
            if field in ("passengers_by_destination", "passenger_waiting_mass"):
                raise ValueError("Expected an OD mapping")
            return [self.packed(v) for v in value]
        return value

    def resource_set(self, nodes):
        key = tuple(nodes)
        if key not in self.resource_index:
            encoded = ranges(nodes)
            if [n for start, end in encoded for n in range(start, end + 1)] != list(nodes):
                raise AssertionError("Resource encoding changed capacity nodes")
            self.resource_index[key] = len(self.resource_sets)
            self.resource_sets.append(encoded)
        return self.resource_index[key]

    def value_set(self, value):
        key = dumps(value)
        if key not in self.value_index:
            self.value_index[key] = len(self.value_sets)
            self.value_sets.append(copy.deepcopy(value))
        return self.value_index[key]

    def commands(self, commands, group):
        result = []
        vehicle = self.vehicles.get(group, {})
        for command in commands:
            c = self.packed(command)
            c.pop("command_id", None)
            if c.get("vehicle_id") == group:
                c["vehicle_id"] = "$group"
            if c.get("expected_node") is not None and c["expected_node"] == vehicle.get("position"):
                c["expected_node"] = "$position"
            if c.get("expected_service_id") is not None and c["expected_service_id"] == vehicle.get("service_id"):
                c["expected_service_id"] = "$service"
            c["dt_s"] = c.pop("effective_at") - self.now
            c = {k: v for k, v in c.items() if v is not None and not (k == "locked" and v is False)}
            result.append(c)
        return result

    def candidates(self):
        choices, indexes, variables = [], {}, {}
        for name, variable in self.payload["solver_variables"].items():
            if variable.get("binary") is not True:
                raise ValueError("Only binary native-command variables are supported")
            group = variable["group"]
            choice = {"option": variable["option"], "commands": self.commands(variable["commands"], group)}
            key = dumps(choice)
            if key not in indexes:
                indexes[key] = len(choices)
                choices.append(choice)
            match = re.fullmatch(r"command_(\d+)_(\d+)", name)
            if not match or int(match[1]) != self.group_index[group]:
                raise ValueError("Variable group index differs from its declared name")
            variables[name] = indexes[key]
        return variables, choices

    @staticmethod
    def od_rows(mapping, digits=1):
        return [[destination, passenger_feature(count, digits)] for destination, count in mapping.items()
                if passenger_feature(count, digits) != 0]

    def current_state(self):
        vehicles = []
        for name, original in self.vehicles.items():
            row = self.packed(original)
            for key in ("passengers_by_destination", "passenger_waiting_mass"):
                if key in original:
                    row[key] = self.od_rows(original[key], 0 if key == "passenger_waiting_mass" else 1)
            if "passenger_count" in row:
                row["passenger_count"] = passenger_feature(row["passenger_count"])
            vehicles.append(row)
        operations = self.snapshot.get("operations", {})
        statuses = self.snapshot.get("command_results", {})
        unfinished = {key: value for key, value in statuses.items()
                      if value.get("status") in ("pending", "running")}
        history = {}
        for status in statuses.values():
            if status.get("status") not in ("pending", "running"):
                key = status.get("action", "unknown") + ":" + status.get("status", "unknown")
                history[key] = history.get(key, 0) + 1
        queue_rows = []
        for platform, queue in self.state.get("queues", {}).items():
            inside, outside = queue["inside"], queue["outside"]
            for destination in dict.fromkeys([*inside, *outside]):
                a = passenger_feature(inside.get(destination, 0))
                b = passenger_feature(outside.get(destination, 0))
                if a != 0 or b != 0:
                    queue_rows.append([platform, destination, a, b])
        result = {"time_s": self.now, "timing": self.state["timing"],
                  "n_agents": self.state["n_agents"], "n_nodes": self.state["n_nodes"],
                  "n_platforms": self.state["n_platforms"],
                  "agent_positions": [dict(v, **({"passengers": passenger_feature(v["passengers"])}
                                              if "passengers" in v else {}))
                                      for v in self.state["agent_positions"]],
                  "vehicles": table(vehicles, self.vehicles),
                  "services": self.packed(self.snapshot.get("services", {})),
                  "operations": table([self.packed(v) for v in operations.values()], operations),
                  "unfinished_commands": table(list(unfinished.values()), unfinished),
                  "history_summary": history,
                  "queues": {"columns": ["platform", "destination", "inside", "outside"], "rows": queue_rows},
                  "waiting_passengers": {p: passenger_feature(v) for p, v in self.state["waiting_passengers"].items()},
                  "events": self.state["disruptions"]}
        # Keep other execution commitments, but not the opaque snapshot hash.
        for key, value in self.snapshot.items():
            if key not in ("time", "vehicles", "services", "operations", "command_results", "events", "version"):
                result[key] = self.packed(value)
        if self.snapshot.get("events", []) != result["events"]:
            result["snapshot_events"] = self.snapshot["events"]
        return result

    def coupling(self):
        original = self.state.get("coupling", {})
        result = {k: self.packed(v) for k, v in original.items()
                  if k not in ("evidence", "reference_conflicts", "events", "current_od_gap")}
        set_fields = ("relations", "vehicles", "shared_vehicles", "depots", "tasks",
                      "affected_platforms", "action_families", "affected_event_ids")
        evidence = []
        for entry in original.get("evidence", []):
            row = copy.deepcopy(entry)
            row["group_indexes"] = [self.group_index[group] for group in row.pop("groups")]
            row["resource_set"] = self.resource_set(row.pop("nodes"))
            if "current_passenger_gap" in row:
                row["current_passenger_gap"] = passenger_feature(row["current_passenger_gap"])
            for field in set_fields:
                if field in row:
                    row[field] = self.value_set(row[field])
            evidence.append(row)
        result["evidence"] = table(evidence)
        result["evidence"]["set_columns"] = list(set_fields)
        grouped = {}
        for conflict in original.get("reference_conflicts", []):
            base = {k: v for k, v in conflict.items() if k != "second"}
            key = dumps(base)
            if key not in grouped:
                grouped[key] = dict(base=base, seconds=[])
            if "second" in conflict:
                grouped[key]["seconds"].append(conflict["second"])
        result["reference_conflicts"] = table([
            dict(item["base"], second_runs=runs(item["seconds"])) for item in grouped.values()])
        result["current_od_gap"] = [[platform, destination, passenger_feature(value)]
            for platform, destinations in original.get("current_od_gap", {}).items()
            for destination, value in destinations.items() if passenger_feature(value) != 0]
        if "stranded_onboard" in result:
            result["stranded_onboard"] = {v: passenger_feature(n) for v, n in result["stranded_onboard"].items()}
        if original.get("events", []) != self.state["disruptions"]:
            result["events"] = original.get("events", [])
        return result

    def encode(self):
        variables, choices = self.candidates()
        result = {"encoding": SCHEMA, "state": self.current_state(),
                  "specification": copy.deepcopy(self.payload["specification"]),
                  "candidate_groups": self.groups, "choice_catalog": choices,
                  "solver_variables": variables, "coupling": self.coupling(),
                  "path_catalog": self.path_catalog, "resource_sets": self.resource_sets,
                  "value_sets": self.value_sets}
        verify_catalog(self.payload, result)
        return result


def compact_payload(payload):
    """Deterministic, label-blind encoding shared with future inference callers."""
    return Encoder(payload).encode()


def restore_paths(value, catalog):
    if isinstance(value, dict):
        if set(value) == {"path_ref"}:
            return expand_runs(catalog[value["path_ref"]])
        return {k: restore_paths(v, catalog) for k, v in value.items()}
    if isinstance(value, list):
        return [restore_paths(v, catalog) for v in value]
    return value


def semantic_command(command):
    return {k: v for k, v in command.items() if k != "command_id" and v is not None
            and not (k == "locked" and v is False)}


def verify_catalog(original, compact):
    if set(original["solver_variables"]) != set(compact["solver_variables"]):
        raise ValueError("Candidate variables were removed or introduced")
    vehicles = original["state"]["dispatch_state"]["vehicles"]
    for name, source in original["solver_variables"].items():
        gi = int(re.fullmatch(r"command_(\d+)_(\d+)", name)[1])
        oi = compact["solver_variables"][name]
        group = compact["candidate_groups"][gi]
        choice = compact["choice_catalog"][oi]
        restored = []
        for command in choice["commands"]:
            c = restore_paths(command, compact["path_catalog"])
            if c.get("vehicle_id") == "$group":
                c["vehicle_id"] = group
            if c.get("expected_node") == "$position":
                c["expected_node"] = vehicles[group]["position"]
            if c.get("expected_service_id") == "$service":
                c["expected_service_id"] = vehicles[group]["service_id"]
            c["effective_at"] = compact["state"]["time_s"] + c.pop("dt_s")
            restored.append(semantic_command(c))
        if (group != source["group"] or choice["option"] != source["option"]
                or restored != [semantic_command(c) for c in source["commands"]]):
            raise ValueError(f"Changed command semantics for {name}")


def validate_record(record, split, topology):
    if [m["role"] for m in record["messages"]] != ["system", "user", "assistant"]:
        raise ValueError("Expected system/user/assistant messages")
    meta = record["metadata"]
    if meta.get("split") != split or meta.get("source_topology") != topology:
        raise ValueError("Wrong split or source topology")
    if any(re.search(r"combination_[BDF](?:_|$)", meta.get(key, ""))
           for key in ("episode_id", "case_id", "source_case_id")):
        raise ValueError("Held-out B/D/F cases cannot enter SFT")
    if meta.get("layout_revision", LAYOUT_REVISION) != LAYOUT_REVISION:
        raise ValueError("Wrong layout revision")
    payload = json.loads(record["messages"][1]["content"])
    labels = json.loads(record["messages"][2]["content"])
    if set(labels) != GUIDANCE_FIELDS:
        raise ValueError("Wrong solver-guidance label fields")
    for field, values in labels.items():
        if not isinstance(values, dict) or not set(values) <= set(payload["solver_variables"]):
            raise ValueError("Labels reference unknown variables")
        if any(type(v) is not int or v < 0 or (field != "branching_priority" and v not in (0, 1))
               for v in values.values()):
            raise ValueError("Invalid guidance values")
    return payload, labels


def stats(values):
    return dict(min=min(values), median=statistics.median(values), max=max(values)) if values else {}


def frozen_prefix(path):
    """Hash a fixed complete-line prefix while an append-only collector runs."""
    size = Path(path).stat().st_size
    digest, boundary = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        while stream.tell() < size:
            raw = stream.readline(size - stream.tell())
            if not raw.endswith(b"\n"):
                break
            digest.update(raw)
            boundary += len(raw)
    return dict(path=str(Path(path).resolve()), bytes=boundary, sha256=digest.hexdigest())


def prefix_records(info):
    if "parts" in info:
        for part in info["parts"]:
            for row in prefix_records(part):
                episode = row.get("metadata", {}).get("episode_id") or row.get("case_id")
                if episode not in part["episode_ids"]:
                    raise ValueError("Record belongs to a different collection shard")
                yield row
        return
    digest = hashlib.sha256()
    with Path(info["path"]).open("rb") as stream:
        while stream.tell() < info["bytes"]:
            raw = stream.readline(info["bytes"] - stream.tell())
            if not raw.endswith(b"\n"):
                raise ValueError("Frozen source prefix changed")
            digest.update(raw)
            if raw.strip():
                yield json.loads(raw)
    if digest.hexdigest() != info["sha256"]:
        raise ValueError("Frozen source prefix changed during preparation")


def collection_sources(folder, *, allow_partial=False):
    folder = Path(folder)
    if any(folder.glob("shard_*")):
        from MILP.collection_shards import read_shards
        folders, identity, summary = read_shards(folder, allow_partial=allow_partial)
        manifests = identity["shard_manifests"]
        def source(name, required=False):
            parts = []
            for path, manifest in zip(folders, manifests):
                file = path/name
                if not file.exists():
                    if required or not allow_partial:
                        raise ValueError(f"{file}: missing shard stream")
                    continue
                parts.append(dict(frozen_prefix(file), episode_ids=manifest["sharding"]["episode_ids"]))
            if not parts:
                return None
            return dict(parts=parts, bytes=sum(p["bytes"] for p in parts),
                        sha256=hashlib.sha256(dumps(parts).encode()).hexdigest())
        return identity, summary, source("sft.jsonl", True), source("solver_records.jsonl"), source("teacher_states.jsonl")
    summary_path, run_path = folder/"summary.json", folder/"run_manifest.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    identity = summary or (json.loads(run_path.read_text()) if run_path.exists() else None)
    if summary is None and not allow_partial:
        raise ValueError(f"{folder}: collection incomplete; use --allow-partial to freeze complete rows")
    if (summary and summary.get("sft_export_version") in ("coupled_teacher_v6", "coupled_teacher_v7", "coupled_teacher_v8", "coupled_teacher_v9")
            and not summary.get("ready_for_sft") and not allow_partial):
        raise ValueError(f"{folder}: coupled-data readiness failed: {summary.get('readiness_checks')}")
    return (identity, summary, frozen_prefix(folder/"sft.jsonl"),
            frozen_prefix(folder/"solver_records.jsonl") if (folder/"solver_records.jsonl").exists() else None,
            frozen_prefix(folder/"teacher_states.jsonl") if (folder/"teacher_states.jsonl").exists() else None)


def teacher_states(info, split, topology):
    """Index v5 raw archives, accepting payload envelopes or legacy chat rows."""
    states = {}
    for row in prefix_records(info):
        meta = row.get("metadata", {})
        if "messages" in row:
            payload, guidance = validate_record(row, split, topology)
        else:
            payload, guidance = row["payload"], row.get("guidance", {})
        uid = meta.get("source_case_id") or meta.get("case_id") or row.get("case_id") or row.get("state_uid")
        if not uid or "::" in uid or uid in states:
            raise ValueError("Missing or repeated teacher archive state identity")
        for key, expected in (("split", split), ("source_topology", topology), ("layout_revision", LAYOUT_REVISION)):
            if meta.get(key, row.get(key, expected)) != expected:
                raise ValueError("Wrong teacher archive split, topology or layout revision")
        selected = (meta.get("teacher_selection") or row.get("selected")
                    or row.get("selected_options") or row.get("teacher_selection"))
        if not selected:
            raise ValueError(f"{uid}: complete teacher selection unavailable in archive")
        states[uid] = dict(payload=payload, guidance=guidance, selected=selected, metadata=meta)
    return states


def validate_group_sample(row, query, labels, archive, *, guidance_quality="useful"):
    """Check an existing short sample against its full, immutable teacher state."""
    from LLM.finetune.guidance_view import (SCHEMA as group_schema, SYSTEM as group_system,
        LEGACY_SYSTEM, expand_query, group_queries, teacher_group_labels, validate_guidance)
    meta = row["metadata"]
    uid, group = meta.get("source_case_id"), query.get("group")
    if (query.get("encoding") != group_schema or meta.get("input_encoding") != group_schema
            or not uid or not group or meta["case_id"] != uid+"::"+group
            or not meta.get("episode_id") or meta.get("layout_revision") != LAYOUT_REVISION):
        raise ValueError("Invalid grouped sample identity, encoding or layout revision")
    if row["messages"][0]["content"] not in (group_system, LEGACY_SYSTEM):
        raise ValueError("Grouped sample uses a different system instruction")
    if archive is None:
        raise ValueError(f"{uid}: teacher_states.jsonl archive required to verify complete candidates")
    source_meta = archive["metadata"]
    check = source_meta.get("guidance_validation")
    if (source_meta.get("accepted_for_sft") is not True or not isinstance(check, dict)
            or check.get("noninferior") is not True
            or (guidance_quality == "useful" and check.get("useful") is not True)):
        raise ValueError("v5 teacher archive lacks accepted paired guidance evidence")
    tested = check.get("tested_guidance")
    if (not isinstance(tested, dict) or set(tested) != GUIDANCE_FIELDS
            or any(not isinstance(values, dict) or any(type(v) is not int for v in values.values())
                   for values in tested.values())):
        raise ValueError("v5 teacher archive lacks complete tested_guidance")
    short_check = {k: v for k, v in check.items() if k not in ("arms", "tested_guidance")}
    if (meta.get("accepted_for_sft") is not True
            or json.dumps(meta.get("guidance_validation"), sort_keys=True, allow_nan=False)
            != json.dumps(short_check, sort_keys=True, allow_nan=False)):
        raise ValueError("Short guidance evidence differs from teacher archive")
    for key, value in source_meta.items():
        if key not in ("case_id", "teacher_selection", "guidance_validation") and (key not in meta or meta[key] != value):
            raise ValueError(f"Short metadata {key} differs from teacher archive")
    if "teacher_selection" in meta and meta["teacher_selection"] != archive["selected"]:
        raise ValueError("Short teacher selection differs from teacher archive")
    if source_meta.get("episode_id", meta["episode_id"]) != meta["episode_id"]:
        raise ValueError("Grouped sample episode differs from teacher archive")
    expected = next((q for g, q in group_queries(archive["payload"]) if g == group), None)
    if expected is None:
        raise ValueError("Grouped sample references an unknown teacher group")
    expected_labels, name, changed = teacher_group_labels(expected, archive["guidance"], archive["selected"])
    if expand_query(query) != expand_query(expected):
        raise ValueError("Grouped sample changed complete candidates or public context")
    validate_guidance(query, labels)
    if labels != expected_labels:
        raise ValueError("Grouped sample labels differ from complete teacher selection")
    if "changed_decision" in meta and meta["changed_decision"] != changed:
        raise ValueError("Grouped sample changed-decision flag differs from teacher selection")
    return name, changed


def prepare_group_data(source_root, output, lines, *, tokenizer, max_tokens=8192,
                       allow_partial=False, continuation_groups=2, improved_only=False):
    from collections import Counter
    from MILP.guidance_labels import families
    from LLM.finetune.guidance_view import SCHEMA as group_schema, chat_tokens, merge_guidance, training_views
    if tokenizer is None:
        raise ValueError("The target model tokenizer is required; character counts are not token bounds")
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(source_root) or source_root.is_relative_to(output):
        raise ValueError("Use a fresh output directory outside the raw source tree")
    if max_tokens < 1 or continuation_groups < 0:
        raise ValueError("Invalid length/continuation limit")
    output.mkdir(parents=True)
    (output/"combined").mkdir()
    manifest = dict(schema=group_schema, status="preparing", source_lines=list(lines), max_tokens=max_tokens,
        tokenizer=str(getattr(tokenizer, "name_or_path", "")), token_checked=True,
        labels="Full selected options restored; one complete candidate group per example; safe zero exclusions only.",
        interpretation="All source lines enter training; they are not unseen-line test data.",
        inference="Use group_queries/group_messages and merge_guidance; never send raw snapshots to this adapter.",
        continuation_groups_per_state=continuation_groups, improved_only=improved_only, records=[], combined={})
    manifest_path = output/"manifest.json"
    manifest_path.write_text(dumps(manifest)+"\n")
    episodes = {s: set() for s in ("train", "validation")}
    seen, seen_samples, seen_inputs = {}, set(), {}
    with (output/"combined/train.jsonl").open("w") as train, (output/"combined/validation.jsonl").open("w") as validation:
        writers = dict(train=train, validation=validation)
        for line in lines:
            (output/line).mkdir()
            for split in writers:
                folder = source_root/line/split
                identity, summary, source, solver_source, archive_source = collection_sources(folder, allow_partial=allow_partial)
                if summary is None and not allow_partial:
                    raise ValueError(f"{folder}: collection incomplete; use --allow-partial to freeze complete rows")
                if (summary and summary.get("sft_export_version") in ("coupled_teacher_v6", "coupled_teacher_v7", "coupled_teacher_v8", "coupled_teacher_v9")
                        and not summary.get("ready_for_sft") and not allow_partial):
                    raise ValueError(f"{folder}: coupled-data readiness failed: {summary.get('readiness_checks')}; "
                                     "inspect coverage before training, or use --allow-partial for a pilot dataset")
                if identity and (identity.get("source_topology") != LINE_IDS[line] or identity.get("source_split") != split
                        or identity.get("scenario_recipe", {}).get("layout_revision") != LAYOUT_REVISION):
                    raise ValueError("Wrong source split, topology or layout revision")
                guidance_quality = (identity or {}).get("collection_config", {}).get("guidance_quality", "useful")
                if guidance_quality not in ("useful", "noninferior"):
                    raise ValueError("Unknown collection guidance_quality")
                archives = teacher_states(archive_source, split, LINE_IDS[line]) if archive_source else {}
                teachers, by_window = {}, {}
                for record in prefix_records(solver_source) if solver_source else ():
                    if record.get("qualified") and record.get("selected_options"):
                        key = (record["case_id"], record["stage"], record["horizon"])
                        teachers[(*key, record.get("budget_s"))] = record
                        by_window.setdefault(key, []).append(record)
                counts, teacher_families, exported_families = Counter(), Counter(), Counter()
                lengths, source_lengths, source_states = [], [], set()
                grouped_states = {}
                with (output/line/(split+".jsonl")).open("w") as destination, \
                     (output/line/(split+"_rejected.jsonl")).open("w") as rejected_file:

                    def export_view(meta, payload, uid, episode, view, *, paired):
                        input_key = hashlib.sha256(json.dumps(view["messages"][:2], sort_keys=True,
                            separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
                        if input_key in seen_inputs:
                            rejected_file.write(dumps(dict(case_id=uid, group=view["group"],
                                reason="duplicate_input", previous_split=seen_inputs[input_key]))+"\n")
                            counts["duplicate_inputs"] += 1
                            return
                        seen_inputs[input_key] = split
                        sample = dict(messages=view["messages"], metadata=dict(meta,
                            case_id=uid+"::"+view["group"], source_case_id=uid, episode_id=episode,
                            input_encoding=group_schema, token_count=view["tokens"],
                            tokenizer=manifest["tokenizer"], max_tokens=max_tokens,
                            source_sft_sha256=source["sha256"], teacher_labels_reconstructed=not paired,
                            changed_decision=view["changed"], source_line=line,
                            paired_guidance_validated=paired,
                            guidance_validation_status=meta.get("sft_export_version", "paired_teacher") if paired else "legacy_unpaired"))
                        validate_record(sample, split, LINE_IDS[line])
                        encoded = dumps(sample)+"\n"
                        destination.write(encoded)
                        writers[split].write(encoded)
                        lengths.append(view["tokens"])
                        exported_families.update(families(payload["solver_variables"][view["selected_variable"]]))
                        counts["samples"] += 1
                        counts["changed_decisions"] += int(view["changed"])

                    for row in prefix_records(source):
                        payload, guidance = validate_record(row, split, LINE_IDS[line])
                        meta = row["metadata"]
                        grouped = meta.get("input_encoding") == group_schema or payload.get("encoding") == group_schema
                        if identity and identity.get("sft_export_version") == "validated_teacher_v5" and not grouped:
                            raise ValueError("v5 sft.jsonl must contain already grouped samples")
                        uid = meta.get("source_case_id") if grouped else meta["case_id"]
                        if not uid:
                            raise ValueError("Missing source state identity")
                        episode = meta.get("episode_id", uid.rsplit("_stage", 1)[0])
                        state_identity = (split, line, episode, grouped)
                        if uid in seen and (not grouped or seen[uid] != state_identity):
                            raise ValueError("Repeated source state or train/validation overlap")
                        if meta["case_id"] in seen_samples:
                            raise ValueError("Repeated source sample or train/validation overlap")
                        seen[uid] = state_identity
                        seen_samples.add(meta["case_id"])
                        if episode in episodes["validation" if split == "train" else "train"]:
                            raise ValueError("Training and validation episodes overlap")
                        episodes[split].add(episode)
                        first_state = uid not in source_states
                        source_states.add(uid)
                        counts["source_states"] += int(first_state)
                        counts["source_samples"] += 1
                        source_lengths.append(sum(len(m["content"]) for m in row["messages"]))
                        teacher = None
                        if grouped:
                            name, changed = validate_group_sample(row, payload, guidance, archives.get(uid),
                                                                  guidance_quality=guidance_quality)
                            grouped_states.setdefault(uid, []).append(dict(row=row, query=payload,
                                labels=guidance, selected_variable=name, changed=changed))
                            counts["validated_source_states"] += int(first_state)
                        else:
                            counts["legacy_source_states"] += 1
                            stage = int(uid.rsplit("_stage", 1)[1])
                            spec = payload["specification"]
                            key = (episode, stage, spec["planning_horizon"])
                            teacher = teachers.get((*key, spec.get("computational_budget")))
                            if teacher is None and len(by_window.get(key, [])) == 1:
                                teacher = by_window[key][0]
                            selected = meta.get("teacher_selection") or (teacher or {}).get("selected_options")
                            if not selected:
                                raise ValueError(f"{uid}: complete teacher selection unavailable; do not invent missing starts")
                        common_cost = meta.get("teacher_common_cost", (teacher or {}).get("common_evaluation", {}).get("cost"))
                        reference_cost = meta.get("reference_common_cost", (teacher or {}).get("reference_evaluation", {}).get("cost"))
                        improved = common_cost is not None and reference_cost is not None and common_cost < reference_cost-1e-6
                        counts["improved_states" if improved else "tied_or_unreported_states"] += int(first_state)
                        if improved_only and not improved:
                            counts["quality_filtered_states"] += int(first_state)
                        if grouped:
                            continue
                        if improved_only and not improved:
                            continue
                        for choice in payload["solver_variables"].values():
                            if selected.get(choice["group"]) == choice["option"]:
                                teacher_families.update(families(choice))
                        views, rejected = training_views(payload, guidance, selected, tokenizer=tokenizer,
                            max_tokens=max_tokens, continuation_groups=continuation_groups)
                        for item in rejected:
                            rejected_file.write(dumps(dict(case_id=uid, **item))+"\n")
                            counts["overlength_groups"] += 1
                        for view in views:
                            export_view(meta, payload, uid, episode, view, paired=False)

                    # Paired evidence belongs to the entire original group bundle.
                    # Buffer states so no row escapes before all groups are checked.
                    for uid, items in grouped_states.items():
                        archive = archives[uid]
                        merged = merge_guidance([item["query"] for item in items], [item["labels"] for item in items])
                        if merged != archive["metadata"]["guidance_validation"]["tested_guidance"]:
                            if allow_partial and summary is None:
                                rejected_file.write(dumps(dict(case_id=uid, reason="incomplete_guidance_bundle",
                                    requires_revalidation=True, quarantined_samples=len(items)))+"\n")
                                counts["quarantined_states"] += 1
                                counts["quarantined_samples"] += len(items)
                                continue
                            raise ValueError(f"{uid}: source group labels differ from archive.tested_guidance")
                        meta = archive["metadata"]
                        common_cost, reference_cost = meta.get("teacher_common_cost"), meta.get("reference_common_cost")
                        if improved_only and not (common_cost is not None and reference_cost is not None
                                                  and common_cost < reference_cost-1e-6):
                            continue
                        views, rejected = [], []
                        for item in items:
                            messages, query = item["row"]["messages"], item["query"]
                            teacher_families.update(families(query["solver_variables"][item["selected_variable"]]))
                            length = chat_tokens(tokenizer, messages)
                            reserved = chat_tokens(tokenizer, messages[:2], generation=True)+512
                            if max(length, reserved) > max_tokens:
                                rejected.append(dict(group=query["group"], tokens=length,
                                    prompt_plus_generation=reserved, reason="complete_group_overlength"))
                            views.append(dict(group=query["group"], messages=messages, tokens=length,
                                selected_variable=item["selected_variable"], changed=item["changed"]))
                        if rejected:
                            rejected_file.write(dumps(dict(case_id=uid, reason="guidance_group_set_changed",
                                requires_revalidation=True, groups=[view["group"] for view in views],
                                rejected_groups=rejected, quarantined_samples=len(items)))+"\n")
                            counts["overlength_groups"] += len(rejected)
                            counts["quarantined_states"] += 1
                            counts["quarantined_samples"] += len(items)
                            continue
                        keys = [hashlib.sha256(json.dumps(v["messages"][:2], sort_keys=True,
                            separators=(",", ":"), ensure_ascii=False).encode()).hexdigest() for v in views]
                        if len(set(keys)) != len(keys) or any(k in seen_inputs for k in keys):
                            rejected_file.write(dumps(dict(case_id=uid, reason="duplicate_input_bundle",
                                quarantined_samples=len(items),
                                cross_split=any(seen_inputs.get(k, split) != split for k in keys)))+"\n")
                            counts["duplicate_inputs"] += sum(k in seen_inputs for k in keys)
                            counts["quarantined_states"] += 1
                            counts["quarantined_samples"] += len(items)
                            continue
                        for item, view in zip(items, views):
                            row_meta = item["row"]["metadata"]
                            export_view(row_meta, item["query"], uid, row_meta["episode_id"], view, paired=True)
                        active = meta.get("sampling", {}).get("active_event_types", [])
                        shape = meta.get("joint_label_coverage", {})
                        counts["exported_active_event_states"] += bool(active)
                        counts["exported_coupled_joint_states"] += (len(active) > 1 and shape.get("multi_measure", False)
                                                                   and shape.get("coupled_adjustment", False))
                if summary and counts["source_states"] != summary["qualified"]:
                    raise ValueError("Completed source count differs from its summary")
                if summary and summary.get("sft_export_version") == "validated_teacher_v5" and "sft_samples" not in summary:
                    raise ValueError("Completed v5 summary must declare sft_samples")
                if summary and "sft_samples" in summary and counts["source_samples"] != summary["sft_samples"]:
                    raise ValueError("Completed source sample count differs from summary.sft_samples")
                if summary and summary.get("sft_export_version") in ("coupled_teacher_v6", "coupled_teacher_v7", "coupled_teacher_v8", "coupled_teacher_v9") and not allow_partial:
                    from MILP.collection_quality import dataset_readiness
                    ready = dataset_readiness(dict(sft_samples=counts["samples"],
                        active_event_states=counts["exported_active_event_states"],
                        coupled_joint_states=counts["exported_coupled_joint_states"]), split)
                    if not ready["ready_for_sft"]:
                        raise ValueError(f"{folder}: readiness failed after cross-shard deduplication: {ready['readiness_checks']}")
                report = dict(line=line, split=split, counts=dict(counts), tokens=stats(lengths),
                    original_chars=stats(source_lengths), teacher_families=dict(teacher_families),
                    exported_families=dict(exported_families), source=source, solver_source=solver_source,
                    teacher_source=archive_source,
                    source_export_version=(identity or {}).get("sft_export_version", "legacy_raw"),
                    paired_guidance_validated=bool(grouped_states) and not counts["legacy_source_states"],
                    collection_complete=summary is not None, layout_verified=identity is not None)
                manifest["records"].append(report)
                manifest_path.write_text(dumps(manifest)+"\n")
                print(dumps(dict(event="prepared_groups", **report)), flush=True)
    if episodes["train"] & episodes["validation"]:
        raise ValueError("Training and validation episodes overlap")
    for split in writers:
        count = sum(r["counts"].get("samples", 0) for r in manifest["records"] if r["split"] == split)
        if count == 0:
            raise ValueError(f"No bounded {split} samples; inspect the rejection records")
        path = output/"combined"/(split+".jsonl")
        manifest["combined"][split] = dict(path=str(path), samples=count, sha256=file_hash(path), episodes=len(episodes[split]))
    manifest.update(status="complete", train_validation_overlap=False,
        overlength_groups=sum(r["counts"].get("overlength_groups", 0) for r in manifest["records"]))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    return manifest


def prepare(source_root, output, lines, *, tokenizer=None, max_tokens=32768):
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh SFT output directory")
    if output.is_relative_to(source_root) or source_root.is_relative_to(output):
        raise ValueError("Prepared data must not replace or contain the raw source directory")
    source_info = {}
    for line in lines:
        for split in ("train", "validation"):
            folder = source_root/line/split
            summary = json.loads((folder/"summary.json").read_text())
            if (summary.get("source_topology") != LINE_IDS[line]
                    or summary.get("source_split") != split
                    or summary.get("scenario_recipe", {}).get("layout_revision") != LAYOUT_REVISION):
                raise ValueError("Source collection is incomplete or has the wrong identity")
            source = folder/"sft.jsonl"
            source_info[(line, split)] = dict(path=str(source), sha256=file_hash(source),
                                             expected_samples=summary["qualified"])
    output.mkdir(parents=True)
    manifest = dict(schema=SCHEMA, status="preparing", source_lines=list(lines),
        layout_revision=LAYOUT_REVISION,
        interpretation="All listed lines are training sources; this is not zero-shot evaluation on those lines.",
        labels="Unchanged; original variable names, values, actions, participants, parameters and paths checked.",
        view_changes=["Summarize finished command history; retain unfinished commands and current operations.",
                      "Lossless path runs, shared resource sets, columnar coupling including inactive evidence.",
                      "Aggregate identical per-second conflicts with lossless second runs; sparse zero-default OD.",
                      "Intern choice templates; retain every variable; remove opaque proposed-command identifiers.",
                      "Round passenger-count features to 0.1 persons and waiting-mass features to whole person-seconds; exact raw states unchanged."],
        inference="Use compact_payload on the original guidance payload; do not feed raw snapshots to this model.",
        tokenizer=str(getattr(tokenizer, "name_or_path", "")) if tokenizer else None,
        tokenizer_purpose="Length audit only; no model weights loaded or selected as a fine-tuning base.",
        max_tokens=max_tokens, records=[], combined={})
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    merged = {split: [] for split in ("train", "validation")}
    seen = {split: set() for split in merged}
    episode_ids = {split: set() for split in merged}
    for line in lines:
        (output/line).mkdir()
        for split in ("train", "validation"):
            info = source_info[(line, split)]
            raw_chars, compact_chars, lengths, records = [], [], [], []
            for row in read_records(info["path"]):
                payload, labels = validate_record(row, split, LINE_IDS[line])
                uid = row["metadata"]["case_id"]
                if uid in seen[split]:
                    raise ValueError(f"Duplicate state label: {uid}")
                seen[split].add(uid)
                episode_ids[split].add(row["metadata"].get("episode_id", uid))
                compact = compact_payload(payload)
                result = dict(messages=[dict(role="system", content=SYSTEM),
                                        dict(role="user", content=dumps(compact)),
                                        copy.deepcopy(row["messages"][2])],
                              metadata=dict(row["metadata"], input_encoding=SCHEMA,
                                            layout_revision=LAYOUT_REVISION,
                                            source_sft_sha256=info["sha256"], source_line=line))
                if json.loads(result["messages"][2]["content"]) != labels:
                    raise AssertionError("Guidance labels changed")
                validate_record(result, split, LINE_IDS[line])
                raw_chars.append(sum(len(m["content"]) for m in row["messages"]))
                compact_chars.append(sum(len(m["content"]) for m in result["messages"]))
                if tokenizer is not None:
                    rendered = tokenizer.apply_chat_template(result["messages"], tokenize=False,
                                                              add_generation_prompt=False)
                    length = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
                    lengths.append(length)
                    if length > max_tokens:
                        raise ValueError(f"{uid}: {length} tokens exceeds {max_tokens}; no truncation or label removal allowed")
                records.append(result)
            if len(records) != info["expected_samples"]:
                raise ValueError("Sample count differs from completed collection summary")
            if file_hash(info["path"]) != info["sha256"]:
                raise ValueError("Raw input changed during preparation")
            destination = output/line/(split+".jsonl")
            destination.write_text("".join(dumps(r)+"\n" for r in records))
            report = dict(line=line, split=split, samples=len(records), source=info,
                          path=str(destination), sha256=file_hash(destination),
                          original_chars=stats(raw_chars), compact_chars=stats(compact_chars),
                          tokens=stats(lengths), token_checked=bool(tokenizer),
                          character_reduction_pct=100*(1-sum(compact_chars)/sum(raw_chars)))
            manifest["records"].append(report)
            merged[split].extend(records)
            print(dumps(dict(event="prepared", **report)), flush=True)
    if seen["train"] & seen["validation"] or episode_ids["train"] & episode_ids["validation"]:
        raise ValueError("Training and validation episodes overlap")
    (output/"combined").mkdir()
    for split, records in merged.items():
        path = output/"combined"/(split+".jsonl")
        path.write_text("".join(dumps(r)+"\n" for r in records))
        manifest["combined"][split] = dict(path=str(path), samples=len(records), sha256=file_hash(path),
                                           episodes=len(episode_ids[split]))
    manifest.update(status="complete", train_validation_overlap=False,
                    original_sources_unchanged=all(file_hash(v["path"]) == v["sha256"] for v in source_info.values()))
    if not manifest["original_sources_unchanged"]:
        raise ValueError("Raw source changed before final validation")
    (output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    print(dumps(dict(event="complete", output=str(output), combined=manifest["combined"])), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lines", nargs="+", choices=list(LINE_IDS), default=list(LINE_IDS))
    parser.add_argument("--tokenizer", help="Local tokenizer directory used only for length validation")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--view", choices=["group", "lossless"], default="group")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--continuation-groups", type=int, default=2)
    parser.add_argument("--improved-only", action="store_true")
    args = parser.parse_args()
    if len(set(args.lines)) != len(args.lines) or args.max_tokens < 1:
        parser.error("Use distinct lines and a positive max-tokens")
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, fix_mistral_regex=True)
    if args.view == "group":
        prepare_group_data(args.source_root, args.output, args.lines, tokenizer=tokenizer,
            max_tokens=args.max_tokens, allow_partial=args.allow_partial, continuation_groups=args.continuation_groups,
            improved_only=args.improved_only)
    else:
        prepare(args.source_root, args.output, args.lines, tokenizer=tokenizer, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
