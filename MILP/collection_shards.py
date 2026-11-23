"""Deterministic whole-episode partitions and strict shard metadata checks."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

SCHEMA = "metro_collection_episode_shards_v1"


def partition_episodes(entries, count=1):
    if type(count) is not int or count < 1 or count > len(entries):
        raise ValueError("Shard count must be between one and the selected episode count")
    ids = [e["case_id"] for e in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated episode identity")
    if count == 1:
        return [list(entries)]
    def category(entry):
        name = entry["scenario"]
        return (0 if name.startswith("combination_") else
                1 if "breakdown" in name or name == "operation_depot_recovery" else 2)
    groups = [[] for _ in range(count)]
    for index, entry in sorted(enumerate(entries), key=lambda pair: (category(pair[1]), pair[0])):
        target = min(range(count), key=lambda i: (len(groups[i]), i))
        groups[target].append((index, entry))
    return [[entry for _, entry in sorted(group)] for group in groups]


def shard_metadata(entries, count=1, index=0):
    groups = partition_episodes(entries, count)
    if type(index) is not int or not 0 <= index < count:
        raise ValueError("Shard index must be in [0, num_shards)")
    assignments = [[e["case_id"] for e in group] for group in groups]
    digest = hashlib.sha256(json.dumps(assignments, separators=(",", ":")).encode()).hexdigest()
    return dict(schema=SCHEMA, num_shards=count, shard_index=index,
                plan_sha256=digest, all_episode_ids=[e["case_id"] for e in entries],
                episode_ids=assignments[index])


def normalized_identity(value):
    result = copy.deepcopy(value)
    result.pop("sharding", None)
    result.get("configuration", {}).pop("shard_index", None)
    return result


def source_declaration(value):
    result = copy.deepcopy(value)
    for key in ("resume", "overwrite", "preflight_only", "stop_after_states"):
        result.get("collection_config", {}).pop(key, None)
    return result


def read_shards(folder, *, allow_partial=False):
    """Validate coverage of the declared episode partition, not per-shard labels."""
    folder = Path(folder)
    paths = sorted(folder.glob("shard_*"))
    if any(not p.is_dir() or p.is_symlink() for p in paths):
        raise ValueError("Invalid shard directory")
    if not paths:
        raise ValueError(f"{folder}: no shard directories")
    if (folder/"sft.jsonl").exists():
        raise ValueError("Mixed flat and sharded collection; use separate output roots")
    manifests, summaries = [], []
    expected, seen_indices, seen_episodes = None, set(), set()
    for path in paths:
        if not (path/"run_manifest.json").exists():
            if allow_partial:
                continue
            raise ValueError(f"{path}: shard has not started")
        manifest = json.loads((path/"run_manifest.json").read_text())
        if not isinstance(manifest.get("identity"), dict) or not manifest["identity"]:
            raise ValueError("Missing shard source identity")
        part = manifest.get("sharding", {})
        if (part.get("schema") != SCHEMA or type(part.get("num_shards")) is not int
                or part["num_shards"] < 1 or type(part.get("shard_index")) is not int
                or not 0 <= part["shard_index"] < part["num_shards"]):
            raise ValueError("Invalid shard metadata")
        if (not isinstance(part.get("plan_sha256"), str) or len(part["plan_sha256"]) != 64
                or any(not isinstance(part.get(key), list) or not part[key]
                       or any(not isinstance(e, str) or not e for e in part[key])
                       for key in ("all_episode_ids", "episode_ids"))):
            raise ValueError("Invalid shard episode plan")
        if path.name != f"shard_{part['shard_index']:02d}":
            raise ValueError("Shard directory and index disagree")
        if manifest.get("identity", {}).get("sharding") != part:
            raise ValueError("Shard identity and episode declaration disagree")
        configuration = manifest["identity"].get("configuration", {})
        if any(key in configuration and configuration[key] != part[key]
               for key in ("num_shards", "shard_index")):
            raise ValueError("Shard configuration and episode declaration disagree")
        common = {k: part[k] for k in ("schema", "num_shards", "plan_sha256", "all_episode_ids")}
        if expected is None:
            expected = common
        if common != expected or part["shard_index"] in seen_indices:
            raise ValueError("Mixed shard plans or repeated shard index")
        if manifests and any(manifest.get(k) != manifests[0].get(k) for k in
                ("source_topology", "source_split", "sft_export_version", "scenario_recipe")):
            raise ValueError("Shard source split, topology, protocol or scenario recipe differs")
        if normalized_identity(manifest["identity"]) != normalized_identity((manifests or [manifest])[0]["identity"]):
            raise ValueError("Shard code, tokenizer, weights, runtime or collection settings differ")
        assigned = part["episode_ids"]
        if (len(set(assigned)) != len(assigned) or seen_episodes.intersection(assigned)
                or not set(assigned) <= set(part["all_episode_ids"])):
            raise ValueError("Overlapping or foreign shard episodes")
        seen_indices.add(part["shard_index"])
        seen_episodes.update(assigned)
        summary = json.loads((path/"summary.json").read_text()) if (path/"summary.json").exists() else None
        declaration = source_declaration(manifest)
        certified = source_declaration(summary) if summary is not None else {}
        if summary is not None and (summary.get("completed") is not True
                or summary.get("identity") != manifest["identity"] or summary.get("sharding") != part
                or type(summary.get("completed_episodes")) is not int
                or summary["completed_episodes"] != len(assigned)
                or any(certified.get(key) != value for key, value in declaration.items())):
            raise ValueError("Shard summary does not certify its declared episodes")
        if summary is not None and (not isinstance(summary.get("counters"), dict)
                or any(key not in summary["counters"] for key in ("completed_episodes", "qualified", "sft_samples"))
                or any(type(value) is not int or value < 0 or summary.get(key) != value
                       for key, value in summary["counters"].items())):
            raise ValueError("Invalid shard summary counters")
        if summary is None and not allow_partial:
            raise ValueError(f"{path}: shard incomplete")
        manifests.append(manifest)
        summaries.append(summary)
    if expected is None:
        raise ValueError("No started collection shards")
    complete = (len(manifests) == expected["num_shards"] and all(s is not None for s in summaries))
    if not allow_partial and not complete:
        raise ValueError("Missing collection shards")
    if len(set(expected["all_episode_ids"])) != len(expected["all_episode_ids"]):
        raise ValueError("Repeated episodes in the parent plan")
    if len(manifests) == expected["num_shards"] and seen_episodes != set(expected["all_episode_ids"]):
        raise ValueError("Shard union omits episodes")
    if len(manifests) == expected["num_shards"]:
        assignments = [m["sharding"]["episode_ids"] for m in
                       sorted(manifests, key=lambda m: m["sharding"]["shard_index"])]
        digest = hashlib.sha256(json.dumps(assignments, separators=(",", ":")).encode()).hexdigest()
        if digest != expected["plan_sha256"]:
            raise ValueError("Shard assignment fingerprint differs from the declared plan")
    source = copy.deepcopy(manifests[0])
    source["sharding"] = dict(expected, merged=True, shard_indices=sorted(seen_indices))
    source["shard_manifests"] = manifests
    if not complete:
        return paths_for(manifests, paths), source, None
    counters, timing, coverage = Counter(), Counter(), {}
    for summary in summaries:
        counters.update(summary["counters"])
        timing.update(summary.get("timing_s", {}))
        for key, values in summary.get("coverage", {}).items():
            coverage.setdefault(key, Counter()).update(values)
    from MILP.collection_quality import dataset_readiness
    aggregate = dict(source, **dict(counters), counters=dict(counters), completed=True,
        coverage={key: dict(values) for key, values in coverage.items()}, timing_s=dict(timing),
        worker_elapsed_s_sum=sum(s["elapsed_s"] for s in summaries),
        worker_elapsed_s_max=max(s["elapsed_s"] for s in summaries),
        **dataset_readiness(counters, source["source_split"]))
    return paths_for(manifests, paths), source, aggregate


def paths_for(manifests, paths):
    by_index = {int(p.name.removeprefix("shard_")): p for p in paths}
    return [by_index[m["sharding"]["shard_index"]] for m in manifests]
