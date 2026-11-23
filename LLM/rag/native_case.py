"""Small, identifier-independent descriptions of published dispatch experience."""
from collections import Counter
import json


MAX_STATE_CHARS = 2_000_000
RELATIONS = {"C1", "C2", "C3", "C4"}


def case_state(case):
    state = case.get("metadata", {}).get("state_dict")
    if isinstance(state, dict) and state:
        return state, False
    description = case.get("state_description", "")
    if not isinstance(description, str) or len(description) > MAX_STATE_CHARS:
        return {}, bool(description)
    try:
        value = json.loads(description)
        return (value if isinstance(value, dict) else {}), False
    except ValueError:
        return {}, False


def state_profile(state):
    dispatch = state.get("dispatch_state", {})
    events = dispatch.get("events", state.get("disruptions", [])) or []
    active = Counter(e["type"] for e in events if e.get("active", True) and "type" in e)
    released = Counter(e["type"] for e in events if not e.get("active", True) and "type" in e)
    evidence = state.get("coupling", {}).get("evidence", [])
    relations, active_relations, patterns = Counter(), Counter(), Counter()
    for row in evidence:
        codes = sorted(set(row.get("relations", [])) & RELATIONS)
        relations.update(codes)
        if codes:
            patterns.update(["+".join(codes)])
        if row.get("active", False):
            active_relations.update(codes)
    counts = {k: state[k] for k in ("n_agents", "n_nodes", "n_platforms") if k in state}
    operations = Counter(op["action"] for op in dispatch.get("operations", {}).values() if "action" in op)
    return dict(state_available=bool(dispatch), time_s=dispatch.get("time"), source_counts=counts,
        event_phase=("active_restrictions" if active else "after_reported_restrictions" if released
                     else "no_reported_events" if dispatch else "unknown"),
        active_event_types=dict(active), released_event_types=dict(released),
        relation_counts=dict(relations), active_relation_counts=dict(active_relations),
        relation_patterns=dict(patterns), ongoing_measure_counts=dict(operations),
        available_measure_families=sorted(set(state.get("coupling", {}).get("covered_action_families", []))))


def case_profile(case):
    state, skipped = case_state(case)
    return dict(state_profile(state), source_snapshot_parse_skipped=skipped)


def transfer_match(current, historic, similarity):
    """Soft retrieval evidence, not a benefit or feasibility certificate.

    No source/current train, node or topology ID is compared. Budget and horizon
    are adaptations made later, not eligibility conditions for experience.
    """
    def jaccard(left, right):
        union = set(left) | set(right)
        return len(set(left) & set(right))/len(union) if union else 0.
    common_relations = sorted(set(current["relation_counts"]) & set(historic["relation_counts"]))
    common_patterns = sorted(set(current["relation_patterns"]) & set(historic["relation_patterns"]))
    common_measures = sorted(set(current["available_measure_families"]) & set(historic["available_measure_families"]))
    common_events = sorted(set(current["active_event_types"]) & set(historic["active_event_types"]))
    relations = .5*jaccard(current["relation_counts"], historic["relation_counts"]) + \
                .5*jaccard(current["relation_patterns"], historic["relation_patterns"])
    event_context = (jaccard(current["active_event_types"], historic["active_event_types"])
                     if current["active_event_types"] else float(not historic["active_event_types"]))
    # Initial untrained weights; source-line validation must assess their quality.
    score = .6*similarity + .25*relations + .1*jaccard(
        current["available_measure_families"], historic["available_measure_families"]) + .05*event_context
    return dict(score=round(float(score), 6), shared_relations=common_relations,
        shared_relation_patterns=common_patterns, shared_available_measures=common_measures,
        shared_active_event_types=common_events, historical_phase=historic["event_phase"],
        basis="soft structural analogy; not exact scenario identity or a benefit guarantee")
