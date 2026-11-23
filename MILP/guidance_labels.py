"""Sparse teacher guidance, independent of vehicle/platform insertion order."""
from collections import Counter, defaultdict, deque
import hashlib


EVENT_RESPONSES = {
    "dwell_extension": {"adjust_departure", "hold", "run_profile", "reserve_insert"},
    "running_time_variation": {"adjust_departure", "run_profile", "reroute"},
    "speed_restriction": {"adjust_departure", "run_profile", "short_turn", "reroute", "reassign_service"},
    "track_blockage": {"short_turn", "reroute", "hold", "reserve_insert", "reassign_service"},
    "platform_closure": {"skip_stop", "short_turn", "reroute", "admission_control", "reassign_service"},
    "station_capacity_reduction": {"admission_control", "adjust_departure", "reserve_insert"},
    "switch_failure": {"reroute", "short_turn", "hold", "reassign_service"},
    "turnback_capacity_loss": {"reroute", "adjust_departure", "hold", "reassign_service"},
    "signal_degradation": {"adjust_departure", "hold", "run_profile", "reassign_service"},
    "power_restriction": {"run_profile", "reroute", "short_turn", "hold", "reassign_service"},
    "vehicle_capacity_loss": {"reserve_insert", "reassign_service", "adjust_departure", "admission_control"},
    "vehicle_breakdown": {"rescue_tow", "withdraw", "reserve_insert", "reassign_service", "evacuate"},
    "depot_access_restriction": {"reserve_insert", "reenter", "withdraw", "reassign_service"},
    "demand_surge": {"reserve_insert", "reassign_service", "adjust_departure", "admission_control"},
    "od_shift": {"reassign_service", "adjust_departure", "reroute"},
    "boarding_rate_reduction": {"adjust_departure", "hold", "admission_control", "reserve_insert"},
}


def public_option_priority(world, option, footprint, events):
    """Order trials using reported constraints/queues, never teacher labels."""
    kinds = footprint["families"]
    affinity = sum(bool(kinds & EVENT_RESPONSES.get(e["type"], set())) for e in events)
    pressure = max((world.waiting_at(p) / max(1., world.safe_platform_queue_capacity)
                    for p in footprint["platforms"]), default=0.)
    # Destructive service changes are not generic first trials in normal traffic.
    destructive = bool(kinds & {"cancel_service", "evacuate", "withdraw"}) and not affinity
    parameter_priority = 0.
    for command in option.commands:
        params = command["parameters"]
        if command["action"] == "admission_control":
            platform = params["platform_id"]
            if world.dispatch.admission_rates.get(platform, float("inf")) <= 1.:
                parameter_priority += min(params["rate"], world.boarding_rate_persons_per_second)/max(
                    1., world.boarding_rate_persons_per_second)
        if command["action"] == "hold" and any(e["type"] in
                {"boarding_rate_reduction", "demand_surge", "dwell_extension"} for e in events):
            parameter_priority += min(params["seconds"], 120.)/120.
    return 4*affinity + min(len(events), 3) + min(pressure, 2.) - 4*destructive + parameter_priority


def variable_catalog(model):
    result = {f"command_{gi}_{oi}": dict(group=g, option=op.id, commands=op.commands,
            actions=op.actions, pauses=op.pauses, binary=True)
            for gi, (g, options) in enumerate(model.groups.items()) for oi, op in enumerate(options)}
    world = getattr(model, "base", None)
    if world is None:
        return result
    pending = {v for plan in world.dispatch.plans.values() for c in plan.commands
               if world.dispatch.command_results[c.command_id]["status"] == "pending"
               and model.now <= c.effective_at < model.now+model.horizon*model.tick for v in c.participants()}
    for variable in result.values():
        group, actions = variable["group"], variable["actions"]
        if (variable["commands"] or variable["pauses"] or set(actions) != {group}
                or group not in world.dispatch.records or group in pending or any(actions[group])):
            continue
        record = world.dispatch.records[group]
        position = world.dispatch.position(group)
        if (position in world.platform_nodes and not record.failed and world.dispatch.service_enabled(record.agent)
                and world.event_view().vehicle_operable(group) and not world.dispatch.operation_for(group)):
            variable.update(control_measures=["hold"], control_basis=dict(kind="platform_hold_control",
                vehicle_id=group, platform_id=position, seconds=model.horizon*model.tick))
    return result


def families(variable):
    return sorted({c["action"] for c in variable.get("commands", [])} | set(variable.get("control_measures", [])))


def round_robin(items, key):
    buckets = defaultdict(deque)
    for item in items:
        buckets[key(item)].append(item)
    while buckets:
        for name in list(buckets):
            yield buckets[name].popleft()
            if not buckets[name]:
                del buckets[name]


def teacher_guidance(catalog, selected, *, invalid=(), limit=64):
    """Retain explicit interventions first, rotating families and group kinds.

    Missing starts remain unspecified, never implicitly fixed to zero. Full
    selections are stored separately so a bounded training view can recover them.
    Zero fixes must come from immediate applicability checks, not the teacher optimum.
    """
    if limit < 1:
        raise ValueError("guidance-variable-limit must be positive")
    chosen = [name for name, v in catalog.items() if selected.get(v["group"]) == v["option"]]
    if len(chosen) != len(selected):
        raise ValueError("Teacher selection does not match the current variable catalog")
    explicit = [n for n in chosen if families(catalog[n])]
    control = [n for n in chosen if not families(catalog[n]) and
               (catalog[n].get("actions") or catalog[n].get("pauses") or
                catalog[n]["option"] not in ("continue", "unchanged"))]
    rest = [n for n in chosen if n not in set(explicit + control)]
    order = list(round_robin(explicit, lambda n: tuple(families(catalog[n]))))
    order += list(round_robin(control + rest, lambda n: catalog[n]["group"].startswith("platform_")))
    # Spend most of the output on useful starts, not duplicate priorities.
    warm = {n: 1 for n in order[:max(1, limit * 3 // 4)]}
    fixes = [n for n in invalid if n in catalog and n not in chosen]
    fixes = list(round_robin(fixes, lambda n: tuple(families(catalog[n]))))
    fixing = {n: 0 for n in fixes[:min(limit // 8, limit - len(warm))]}
    priority = {n: 10 for n in order[:limit - len(warm) - len(fixing)]}
    guidance = dict(warm_start=warm, variable_fixing=fixing, branching_priority=priority)
    coverage = dict(selected_groups=len(chosen), exported_start_groups=len(warm),
        selected_families=sorted({f for n in chosen for f in families(catalog[n])}),
        selected_control_families=sorted({f for n in chosen for f in catalog[n].get("control_measures", [])}),
        exported_families=sorted({f for n in warm for f in families(catalog[n])}),
        omitted_selected_variables=[n for n in chosen if n not in warm])
    return guidance, coverage


def neighborhood_seeds(model, masks, family_counts=None, limit=12, coupling=None,
                       anchor=None, joint_first=False, cover_families=True):
    """Rotate proposals using completed-trial family counts, never label quality.

    The collector supplies tested_seed_families, not scheduled/accepted counts.
    Seeds do not fix variables, certify feasibility or change the finite domain.
    """
    if limit <= 0:
        return []
    counts = family_counts or {}
    reference = tuple(next(i for i, op in enumerate(opts)
                           if not op.commands and not op.actions and not op.pauses)
                      for opts in model.groups.values())
    if anchor is not None:
        reference = tuple(anchor)
        if len(reference) != len(model.groups) or any(not 0 <= i < len(opts)
                for i, opts in zip(reference, model.groups.values())):
            raise ValueError("Neighborhood anchor must use this catalog")
    options = []
    for gi, (group, choices) in enumerate(model.groups.items()):
        for oi, option in enumerate(choices):
            if oi != reference[gi] and masks[group][oi]:
                kinds = tuple(sorted({c["action"] for c in option.commands})) or (option.id,)
                options.append((gi, oi, kinds))
    priorities = (coupling or {}).get("option_priority", {})
    groups = list(model.groups)
    def priority(item):
        gi, oi, _ = item
        return priorities.get(groups[gi], {}).get(model.groups[groups[gi]][oi].id, 0.)
    options.sort(key=lambda x: (-priority(x), min(counts.get(f, 0) for f in x[2]), x[0], x[1]))
    options = list(round_robin(options, lambda x: x[2]))
    if joint_first and coupling:
        return _joint_seeds(model, options, reference, counts, limit, coupling, cover_families=cover_families)
    result = []
    single_limit = limit if not coupling else max(1, (2*limit+2)//3)
    for gi, oi, _ in options:
        if len(result) >= single_limit:
            break
        trial = list(reference)
        trial[gi] = oi
        result.append(trial)
    if coupling:
        groups = list(model.groups)
        linked = {frozenset(row["groups"]) for row in coupling.get("evidence", []) if len(row["groups"]) == 2}
        for left_index, (gi, oi, _) in enumerate(options):
            if len(result) >= limit:
                break
            for gj, oj, _ in options[left_index+1:]:
                if gi == gj or frozenset((groups[gi], groups[gj])) not in linked:
                    continue
                first, second = model.groups[groups[gi]][oi], model.groups[groups[gj]][oj]
                # Do not seed obvious same-instant reuse of a physical vehicle.
                def claims(option):
                    return {(c["effective_at"], v) for c in option.commands
                            for v in (c.get("vehicle_id"), c.get("parameters", {}).get("failed")) if v is not None}
                if claims(first) & claims(second):
                    continue
                trial = list(reference)
                trial[gi], trial[gj] = oi, oj
                if trial not in result:
                    result.append(trial)
                if len(result) >= limit:
                    break
    return result


def _joint_seeds(model, options, reference, counts, limit, coupling, *, cover_families=True):
    """Cover joint-capable families before repeating groups/parameter variants.

    Every explicit family with a structurally compatible related-group proposal
    gets a joint slot when limit permits. Physical feasibility is still unknown.
    """
    groups = list(model.groups)
    indices = {group: index for index, group in enumerate(groups)}
    edges = {frozenset(r["groups"]): r for r in coupling.get("evidence", [])
             if len(set(r.get("groups", []))) == 2 and r.get("active", True)
             and set(r["groups"]) <= set(groups)}
    priorities = coupling.get("option_priority", {})
    explicit = {c["action"] for gi, oi, _ in options for c in model.groups[groups[gi]][oi].commands}
    option_priorities = {item[:2]: priorities.get(groups[item[0]], {}).get(
        model.groups[groups[item[0]]][item[1]].id, 0.) for item in options}
    tie_keys = {item[:2]: hashlib.sha256(
        f"{getattr(model, 'token', '')}:{groups[item[0]]}:{item[1]}".encode()).hexdigest() for item in options}
    def priority(item):
        return option_priorities[item[:2]]
    def tie_key(item):
        return tie_keys[item[:2]]

    footprints, event_links = {}, {}
    world = getattr(model, "base", None)
    if world is not None:
        from LLM.coupling_trigger import option_footprints, event_relevant
        footprints = option_footprints(model)
        events = [e for e in world.get_published_events() if e.get("active", True)]
        for item in options:
            gi, oi, kinds = item
            fp = footprints[groups[gi]][oi]
            event_links[item[:2]] = {e["event_id"] for e in events
                if set(kinds) & EVENT_RESPONSES.get(e["type"], set()) and event_relevant(e, fp, world)}

    def claims(gi, oi):
        result = []
        for c in model.groups[groups[gi]][oi].commands:
            when, params = c["effective_at"], c.get("parameters", {})
            vehicles = {c.get("vehicle_id")}
            if c["action"] == "rescue_tow":
                vehicles.add(params["failed"])
            result.extend((when, ("vehicle", v)) for v in vehicles if v is not None)
            if c["action"] == "admission_control" and "platform_id" in params:
                result.append((when, ("admission", params["platform_id"])))
            if c["action"] == "adjust_departure" and c.get("vehicle_id") is None and "platform_id" in params:
                result.append((when, ("headway", params["platform_id"])))
        return Counter(result)

    needed = {item[:2] for item in options} | set(enumerate(reference))
    occupied = {(gi, oi): claims(gi, oi) for gi, oi in needed}
    anchor_claims = Counter()
    anchor_owners = defaultdict(Counter)
    for gi, oi in enumerate(reference):
        anchor_claims.update(occupied[gi, oi])
        for key, count in occupied[gi, oi].items():
            anchor_owners[key][gi] += count
    anchor_conflicts = {key for key, value in anchor_claims.items() if value > 1}
    claim_sets = {key: frozenset(values) for key, values in occupied.items()}
    duplicate_claims = {key for key, values in occupied.items() if any(n > 1 for n in values.values())}
    # Anchor claims disappear only when their owning groups are replaced.
    blocked_by_anchor = {key: frozenset(owner for resource in resources
        for owner in anchor_owners.get(resource, ()) if owner != key[0]) for key, resources in claim_sets.items()}
    def compatible(items):
        changed, claimed = {item[0] for item in items}, set()
        for gi, oi, _ in items:
            key = (gi, oi)
            if (key in duplicate_claims or not blocked_by_anchor[key] <= changed
                    or not claimed.isdisjoint(claim_sets[key])):
                return False
            claimed.update(claim_sets[key])
        return all(sum(n for owner, n in anchor_owners[key].items() if owner not in changed) <= 1
                   for key in anchor_conflicts)

    queued_platforms = {p for p in world.platform_nodes if world.waiting_at(p) > 0} if world is not None else set()
    def related(first, second):
        if not footprints:
            return True
        left = footprints[groups[first[0]]][first[1]]
        right = footprints[groups[second[0]]][second[1]]
        return any(not left[key].isdisjoint(right[key])
                   for key in ("nodes", "resources", "vehicles", "depots", "tasks")) or any(
            p in right["platforms"] and p in queued_platforms for p in left["platforms"])

    def item_key(item):
        kinds = set(item[2]) & explicit
        return (not bool(kinds), min((counts.get(f, 0) for f in kinds), default=0),
                -len(event_links.get(item[:2], ())), -priority(item), tie_key(item))

    # Keep every group/family bucket. Parameter variants are fallbacks when a
    # representative conflicts with the anchor or its proposed partner.
    buckets = defaultdict(lambda: defaultdict(list))
    for item in sorted(options, key=item_key):
        buckets[item[0]][item[2]].append(item)
    representatives = []
    for by_family in buckets.values():
        for items in by_family.values():
            representative = next((item for item in items if compatible((item,))), None)
            if representative is not None:
                representatives.append(representative)
    pairs_by_family = defaultdict(list)
    for edge, evidence in edges.items():
        left, right = sorted(indices[g] for g in edge)
        for left_items in buckets[left].values():
            for right_items in buckets[right].values():
                pair = next(((a, b) for a in left_items for b in right_items
                             if compatible((a, b)) and related(a, b)), None)
                if pair is None:
                    continue
                first, second = pair
                hits_a, hits_b = (event_links.get(item[:2], set()) for item in pair)
                affected = set(evidence.get("affected_event_ids", ()))
                kinds = tuple(sorted(set(first[2] + second[2])))
                score = (-len(hits_a & hits_b & affected), -len((hits_a | hits_b) & affected),
                         set(first[2]) == set(second[2]), -(priority(first) + priority(second)),
                         -len(evidence.get("relations", ())), "".join(sorted((tie_key(first), tie_key(second)))))
                rows = pairs_by_family[kinds]
                rows.append((score, first, second))
                rows.sort(key=lambda row: row[0])
                del rows[max(2, min(limit, 4)):]
    pairs = sorted((row for rows in pairs_by_family.values() for row in rows), key=lambda row: row[0])
    result, emitted = [], set()
    def add(items):
        trial = list(reference)
        for gi, oi, _ in items:
            trial[gi] = oi
        trial = tuple(trial)
        if trial != reference and trial not in emitted and len(result) < limit and compatible(items):
            result.append(trial)
            emitted.add(trial)
            return True
        return False

    if not cover_families:
        # Online requests rank current operational relevance, not the data
        # collector's coverage quotas. Mix joint and single revisions so an
        # unhelpful partner does not hide a useful local adjustment.
        from dispatch_candidates import stock_start_options
        lookup = {item[:2]: item for item in options}
        if world is not None:
            masks = {g: [False]*len(opts) for g, opts in model.groups.items()}
            for gi, oi, _ in options:
                masks[groups[gi]][oi] = True
            stock = stock_start_options(model, masks)
            if stock and stock[0] in lookup:
                add((lookup[stock[0]],))
        singles = deque(sorted(representatives, key=lambda item: (-priority(item), tie_key(item))))
        joint = deque(pairs)
        while len(result) < limit and (joint or singles):
            if joint:
                _, first, second = joint.popleft()
                add((first, second))
            if singles:
                add((singles.popleft(),))
        return result

    singles = list(round_robin(sorted(representatives, key=item_key), lambda item: item[2]))
    single_queue = deque(singles)
    def add_single():
        while single_queue:
            if add((single_queue.popleft(),)):
                return True
        return False

    triple_count, triple_limit = 0, max(1, limit//6)
    third_options = sorted(options, key=item_key)
    def add_triple(first, second, uncovered):
        nonlocal triple_count
        if triple_count >= triple_limit or len(result) >= limit:
            return False
        base_families = set(first[2] + second[2]) & explicit
        if len(base_families) < 2:
            return False
        for third in third_options:
            if third[0] in (first[0], second[0]) or not (set(third[2]) & explicit) - base_families:
                continue
            if (uncovered and not uncovered & set(third[2])
                    and limit - len(result) <= len(uncovered)):
                continue
            if not any(frozenset((groups[peer[0]], groups[third[0]])) in edges and related(peer, third)
                       for peer in (first, second)):
                continue
            if add((first, second, third)):
                triple_count += 1
                uncovered.difference_update(first[2] + second[2] + third[2])
                return True
        return False

    uncovered = {f for _, a, b in pairs for f in a[2] + b[2]} & explicit
    while uncovered and len(result) < limit:
        def coverage_key(row):
            score, first, second = row
            new = uncovered & set(first[2] + second[2])
            return (min(counts.get(f, 0) for f in new), -len(new), score)
        row = min((r for r in pairs if uncovered & set(r[1][2] + r[2][2])), key=coverage_key)
        _, first, second = row
        add((first, second))
        uncovered.difference_update(first[2] + second[2])
        # Leave one slot per remaining family even if none can share a pair.
        if limit - len(result) > len(uncovered):
            add_single()
        # Keep one three-measure opportunity near the front, not behind every
        # pair. Its newly covered families also discharge the reserved quota.
        if triple_count == 0:
            add_triple(first, second, uncovered)

    pair_queue = deque(round_robin(pairs, lambda row: tuple(sorted(set(row[1][2] + row[2][2])))))
    while len(result) < limit and (pair_queue or single_queue):
        while pair_queue:
            _, first, second = pair_queue.popleft()
            if add((first, second)):
                add_triple(first, second, uncovered)
                break
        if len(result) < limit:
            add_single()
    for item in sorted(options, key=item_key):
        if len(result) >= limit:
            break
        add((item,))
    return result
