"""Public, bounded context for choosing a joint optimization scope.

This view is not the MILP input. Exact candidates, OD flows, physical paths and
execution checks remain in the solver. Coupling adjacency is retained exactly;
only selected links carry detailed explanations in the language-model view.
"""
from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import json
import math

from LLM.rag.native_case import case_profile


SCHEMA = "metro_strategy_v1"
DEFAULT_INPUT_TOKENS = 32768
DEFAULT_OUTPUT_TOKENS = 1024
CHAT_RESERVE = 1024
INSTRUCTIONS = (
    "metro_strategy_v1 is a scope-selection view, not a feasibility model. "
    "Table rows follow their columns. Integer ranges [lo,hi] are inclusive; "
    "path_runs [start,step,count] preserve path order. In coupling graphs, "
    "[source_index,target_ranges] denotes undirected links using the groups list. "
    "graph_aliases maps a relation to another relation with identical adjacency. "
    "event_groups gives group-index ranges from links annotated with each public event. "
    "C1=shared infrastructure; C2=vehicle/depot/service competition; "
    "C3=passenger-service interaction; C4=route/time interaction. "
    "Links describe potential candidate coupling, not certified simultaneous conflicts or gains. "
    "All relation links, active links and group IDs are retained; detailed_links "
    "are examples, not the whole graph. Passenger top_od lists are summaries. "
    "A bound service with default stops follows the vehicle's current route; "
    "its binding is already present in vehicles, so its redundant task row may be omitted. "
    "Historic identifiers are not current decision variables. Choose current "
    "agent IDs (agent_N may be returned as integer N) and facility IDs only. "
    "The controller closes the scope over shared resources and ongoing work; "
    "the MILP retains all constraints and candidates regardless of this summary. "
    "Topology codes are categorical IDs, never node counts. Historic active_event_types "
    "and released_event_types describe different states; after_reported_restrictions "
    "does not assert complete recovery. Available measures are candidates, not proven "
    "teacher selections. Transfer matches are retrieval evidence, not certified gains."
)


class LLMInputTooLong(ValueError):
    def __init__(self, audit):
        self.audit = dict(audit, status="blocked_before_api")
        super().__init__("LLM input exceeds local limit: " + json.dumps(self.audit, sort_keys=True))


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def check_messages(messages, max_input_tokens=DEFAULT_INPUT_TOKENS):
    """Dependency-free conservative byte budget, NOT a provider tokenizer count.

    Byte-level tokenizers merge input bytes. Budgeting one token per UTF-8 byte
    plus a separate chat-framing reserve avoids relying on a Qwen tokenizer to
    count a different provider's tokens. The much smaller local budget is
    independent of the provider's advertised context capacity.
    """
    if type(max_input_tokens) is not int or max_input_tokens <= CHAT_RESERVE:
        raise ValueError("LLM input limit must exceed the chat-framing reserve")
    content = [m.content if hasattr(m, "content") else m["content"] for m in messages]
    if not all(isinstance(v, str) for v in content):
        raise TypeError("Strategic LLM messages must contain plain text")
    audit = dict(input_utf8_bytes=sum(len(v.encode("utf-8")) for v in content),
                 count_method="utf8_bytes_conservative_not_provider_tokenizer",
                 framing_reserve=CHAT_RESERVE, max_input_tokens=max_input_tokens)
    audit["input_tokens_upper_bound"] = audit["input_utf8_bytes"] + CHAT_RESERVE
    if audit["input_tokens_upper_bound"] > max_input_tokens:
        raise LLMInputTooLong(audit)
    audit["messages_sha256"] = hashlib.sha256(dumps(content).encode("utf-8")).hexdigest()
    return audit


def ranges(values):
    result = []
    for n in sorted(set(int(v) for v in values)):
        if result and n == result[-1][1] + 1:
            result[-1][1] = n
        else:
            result.append([n, n])
    return result


def path_runs(values):
    result = []
    for value in values:
        n = int(value)
        if result and result[-1][2] == 1:
            result[-1][1], result[-1][2] = n - result[-1][0], 2
        elif result and n == result[-1][0] + result[-1][1] * result[-1][2]:
            result[-1][2] += 1
        else:
            result.append([n, 0, 1])
    return result


def scalar(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        rounded = round(value, 3)
        return int(rounded) if rounded.is_integer() else rounded
    raise TypeError("Non-scalar value in strategic context")


def fields(value, names):
    return {name: scalar(value[name]) for name in names if name in value}


def top_od(values, count):
    nonzero = [(str(d), float(q)) for d, q in values.items() if float(q) > 0]
    nonzero.sort(key=lambda row: (-row[1], row[0]))
    return dict(total=scalar(sum(q for _, q in nonzero)), destinations=len(nonzero),
                top_od=[[d, scalar(q)] for d, q in nonzero[:count]])


def coupling_context(value, detail_links=8):
    evidence = value.get("evidence", [])
    groups = sorted({g for row in evidence for g in row.get("groups", [])})
    indexes = {g: i for i, g in enumerate(groups)}
    graphs, active = defaultdict(lambda: defaultdict(set)), defaultdict(set)
    event_groups = defaultdict(set)
    parents = list(range(len(groups)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for row in evidence:
        for event in row.get("affected_event_ids", []):
            event_groups[event].update(indexes[g] for g in row.get("groups", []))
        for a, b in combinations(sorted({indexes[g] for g in row.get("groups", [])}), 2):
            parents[find(b)] = find(a)
            for relation in row.get("relations", []):
                graphs[relation][a].add(b)
            if row.get("active"):
                active[a].add(b)

    def graph_rows(adjacency):
        return [[a, ranges(bs)] for a, bs in sorted(adjacency.items())]

    components = defaultdict(list)
    for i in range(len(groups)):
        components[find(i)].append(i)
    details = []
    ordered = sorted(evidence, key=lambda r: (
        not r.get("active", False), not r.get("reference_conflict", False),
        not r.get("work_blocked", False), -r.get("current_passenger_gap", 0), tuple(r.get("groups", []))))
    for row in ordered[:detail_links]:
        item = {k: row[k] for k in ("groups", "relations", "shared_resources", "shared_vehicles", "depots",
            "tasks", "affected_platforms", "action_families", "affected_event_ids", "active",
            "reference_conflict", "work_blocked", "persistent_stages") if k in row}
        item["shared_node_ranges"] = ranges(row.get("nodes", []))
        item["current_passenger_gap"] = scalar(row.get("current_passenger_gap", 0))
        details.append(item)
    conflicts = Counter(r.get("vehicle") for r in value.get("reference_conflicts", []))
    relation_graphs, aliases, seen = {}, {}, {}
    for relation, adjacency in sorted(graphs.items()):
        rows = graph_rows(adjacency)
        fingerprint = dumps(rows)
        if fingerprint in seen:
            aliases[relation] = seen[fingerprint]
        else:
            relation_graphs[relation] = rows
            seen[fingerprint] = relation
    return dict(groups=groups, relation_graphs=relation_graphs, graph_aliases=aliases,
        active_graph=graph_rows(active), components=list(components.values()),
        signals=value.get("signals", {}), thresholds=value.get("thresholds", {}),
        focus_vehicles=value.get("focus_vehicles", []),
        focus_node_ranges=ranges(value.get("focus_nodes", [])),
        focus_platforms=value.get("focus_platforms", []),
        relation_counts={k: sum(map(len, v.values())) for k, v in sorted(graphs.items())},
        event_ids=sorted({e for r in evidence for e in r.get("affected_event_ids", [])}),
        event_groups={k: ranges(v) for k, v in sorted(event_groups.items())},
        available_measure_families=sorted(set(value.get("covered_action_families", [])) |
            {f for r in evidence for f in r.get("action_families", [])}),
        reference_conflicts_by_vehicle=dict(conflicts),
        stranded_onboard={k: scalar(v) for k, v in value.get("stranded_onboard", {}).items()},
        detailed_links=details, evidence_rows=len(evidence),
        detailed_links_omitted=max(0, len(evidence)-len(details)))


def facility_context(world):
    if world is None or getattr(world, "dispatch", None) is None:
        return {}
    view = world.event_view(published=True)
    resources = {}
    for kind in ("turnbacks", "shared_routes", "switches"):
        rows = {}
        for key, spec in world.event_resources.get(kind, {}).items():
            row = fields(spec, ("node_id", "capacity"))
            if kind == "switches":
                row["route_ids"] = list(spec.get("route_ids", []))
                row["route_entry_nodes"] = dict(spec.get("route_entry_nodes", {}))
            row["segments"] = spec.get("segments", [])
            if kind == "turnbacks":
                row["effective_capacity"] = view.turnback_capacity(key, spec.get("capacity", 1))
            rows[key] = row
        resources[kind] = rows
    resources["depots"] = {k: fields(v, ("node_id", "exit_node_id", "capacity", "inspection_bays"))
                            for k, v in world.dispatch.depots.items()}
    for key, spec in world.dispatch.depots.items():
        resources["depots"][key].update(fields(spec, (
            "preparation_seconds", "inspection_seconds", "inspection_capacity")))
    resources["platform_limits"] = [[p, scalar(view.queue_capacity(p, world.safe_platform_queue_capacity)),
        scalar(view.boarding_multiplier(p)), view.boarding_allowed(p)] for p in sorted(world.platform_nodes)]
    resources["reserve_activation"] = dict(policy=world.dispatch.config.get("reserve_policy", "unrestricted"),
                                            **world.dispatch.reserve_need())
    return resources


def case_context(case, *, compact=False):
    meta, plan = case.get("metadata", {}), case.get("action_plan", {})
    profile = case.get("transfer_profile") or case_profile(case)
    spec = plan.get("specification", {}).get("solver_config", plan.get("specification", {}))
    result = dict(case_id=case.get("case_id"), source_topology_code=meta.get("source_topology"),
        similarity=scalar(case.get("similarity")),
        source_counts=profile["source_counts"], event_phase=profile["event_phase"],
        source_scope_counts=dict(vehicles=len(spec["focus_agents"]) if isinstance(spec.get("focus_agents"), list) else None,
                                 facilities=len(spec["focus_nodes"]) if isinstance(spec.get("focus_nodes"), list) else None),
        event_types=profile["active_event_types"], active_event_types=profile["active_event_types"],
        released_event_types=profile["released_event_types"],
        ongoing_measure_counts=profile["ongoing_measure_counts"],
        available_measure_families=profile["available_measure_families"],
        relation_counts=profile["relation_counts"], relation_patterns=profile["relation_patterns"],
        active_relation_counts=profile["active_relation_counts"],
        transfer_match=case.get("transfer_match", {}),
        configuration=fields(spec, ("time_horizon", "time_limit", "planning_horizon", "computational_budget", "scope_rule")),
        outcome=fields(case.get("outcome", {}), ("reward", "reference_cost", "feasible")),
        teacher_status=meta.get("teacher_status"), teacher_gap=scalar(meta.get("teacher_gap")),
        source_snapshot_omitted=True, source_snapshot_parse_skipped=profile["source_snapshot_parse_skipped"])
    if compact:
        result = {k: result[k] for k in ("case_id", "event_phase", "active_event_types", "released_event_types",
            "ongoing_measure_counts", "configuration")}
        result["relation_types"] = sorted(profile["relation_counts"])
        result["relation_patterns"] = sorted(profile["relation_patterns"])
    return result


def strategy_context(state, retrieved=None, request=None, *, world=None, detail_level=0):
    """Project known fields only; never serialize raw snapshots or a solver catalog."""
    od_count, n_links, n_cases = ((3, 8, 5), (1, 4, 3), (0, 0, 3))[detail_level]
    snapshot, coupling = state.get("dispatch_state", {}), state.get("coupling", {})
    vehicles = snapshot.get("vehicles", {})
    positions = {str(i): v for i, v in enumerate(state.get("agent_positions", []))}
    columns = ["id", "position", "status", "service_id", "depot", "role", "failed", "needs_inspection",
               "has_served", "passenger_count", "capacity", "dwell_time", "expected_dwell", "travel_delay_steps"]
    rows = []
    onboard = {}
    for ident, vehicle in vehicles.items():
        pos = positions.get(ident.removeprefix("agent_"), {})
        row = {**pos, **vehicle, "id": ident}
        rows.append([scalar(row.get(k)) for k in columns])
        if od_count and vehicle.get("passengers_by_destination"):
            onboard[ident] = top_od(vehicle["passengers_by_destination"], od_count)
    operations = {}
    for ident, raw in snapshot.get("operations", {}).items():
        row = fields(raw, ("action", "vehicle_id", "partner", "phase", "started_at", "deadline",
                           "remaining_seconds", "target", "depot", "wait_reason", "service_id", "turnback_entry"))
        row.update(path_runs=path_runs(raw.get("path", [])), continuation_runs=path_runs(raw.get("continuation", [])))
        operations[ident] = row
    service_columns = ["id", "vehicle_id", "reserved_vehicle_id", "started_at", "purpose", "required", "stops"]
    active_services = {key: row for key, row in snapshot.get("services", {}).items() if row.get("ended_at") is None}
    services = dict(columns=service_columns, rows=[
        [key]+[row.get(k) for k in service_columns[1:]]
        for key, row in active_services.items()
        if not (row.get("vehicle_id") in vehicles and row.get("stops") is None
                and not row.get("reserved_vehicle_id")
                and vehicles[row["vehicle_id"]].get("service_id") == key)])
    services["bound_default_details_omitted"] = len(active_services)-len(services["rows"])
    movement = {key: {k: row[k] for k in ("pending_edge", "profile") if row.get(k) is not None}
                for key, row in vehicles.items() if row.get("pending_edge") is not None or row.get("profile") is not None}
    clock_columns = ["id", "arrival_time", "target_departure_time", "dwell_min_limit", "dwell_max_limit"]
    clocks = dict(columns=clock_columns, rows=[])
    if world is not None:
        for agent in world.agents:
            clocks["rows"].append([agent.name]+[
                scalar(float(getattr(agent.state, k)[0])) if hasattr(getattr(agent, "state", None), k) else None
                for k in clock_columns[1:]])
    queues = []
    queue_columns = ["platform", "inside", "outside", "reference_remaining", "top_od"]
    for platform, sides in state.get("queues", {}).items():
        combined = Counter(sides.get("inside", {})) + Counter(sides.get("outside", {}))
        gap = coupling.get("current_od_gap", {}).get(platform,
            coupling.get("current_od_gap", {}).get(int(platform), {}))
        queues.append([platform, scalar(sum(sides.get("inside", {}).values())),
            scalar(sum(sides.get("outside", {}).values())), scalar(sum(gap.values())),
            top_od(combined, od_count)["top_od"]])
    cases = (retrieved or {}).get("cases", [])
    proposal = state.get("native_marl_proposal", {})
    repair = proposal.get("repair", {})
    result = dict(schema=SCHEMA, request=request, time_s=snapshot.get("time"), timing=state.get("timing", {}),
        topology_code=getattr(world, "topology", state.get("topology")),
        counts=fields(state, ("n_agents", "n_nodes", "n_platforms")),
        # Only the published event list enters this view; never read the registry.
        published_events=state.get("disruptions", snapshot.get("events", [])),
        vehicles=dict(columns=columns, rows=rows), onboard_od=onboard, movement_commitments=movement,
        vehicle_clocks=clocks,
        ongoing_operations=operations, current_service_tasks=services,
        inventory=snapshot.get("inventory", {}), depot_bookings=snapshot.get("depot_bookings", {}),
        admission_rates=snapshot.get("admission_rates", {}),
        queues=dict(columns=queue_columns, rows=queues),
        facilities=facility_context(world), coupling=coupling_context(coupling, n_links),
        marl_proposal=dict(rejected=bool(proposal.get("rejected")),
            repaired_groups=repair.get("repaired_groups", []), repair_fraction=proposal.get("repair_fraction")),
        retrieved_cases=[case_context(c, compact=detail_level==2) for c in cases[:n_cases]],
        omitted=dict(raw_solver_catalog=True, completed_command_history=True, full_historic_snapshots=True,
                     od_tail_detail=True, retrieved_cases=max(0, len(cases)-n_cases)), detail_level=detail_level)
    return result


def context_audit(state, retrieved, context):
    evidence = state.get("coupling", {}).get("evidence", [])
    return dict(schema=SCHEMA, detail_level=context["detail_level"],
        coupling_evidence_rows=len(evidence),
        raw_shared_node_entries=sum(len(row.get("nodes", [])) for row in evidence),
        coupling_groups=len(context["coupling"]["groups"]),
        relation_edges=sum(context["coupling"]["relation_counts"].values()),
        included_cases=len(context["retrieved_cases"]),
        case_ids=[case.get("case_id") for case in context["retrieved_cases"]],
        raw_case_description_chars=sum(len(c.get("state_description", "")) for c in (retrieved or {}).get("cases", [])))
