"""Named line snapshots using the original directed-platform/route construction.

Turnback locations are explicit line inputs, not station-index heuristics.
Engineering reports and operating records are distinguished from inferred
facilities. Station-level connections and unit conflict capacity are macroscopic
modeling assumptions, not surveyed interlocking routes. Depot access stations
follow the cited sources; lead durations, directional station access and fleet
sizes remain simulation assumptions.
Legacy topology IDs are intentionally unchanged.
"""
import copy
from itertools import combinations

from facility.platform import Platform


LAYOUT_REVISION = "named_depot_access_v3_20261028"
TURNBACK_SOURCES = {
    "sz20_eia": dict(
        url="https://www.szmc.net/SMARTC/upload/file/20200826/1598405736331043311.pdf",
        locator="Section 2.1.2.3, Fig. 2.1-7, printed page 40; terminal operating role",
        scope="Phase-I terminal locations; no intermediate turnback permission inferred"),
    "sz20_stabling_assumption": dict(
        url="https://www.szmc.net/SMARTC/upload/file/20200826/1598405736331043311.pdf",
        locator="Section 2.1.2.3, Figs. 2.1-5 and 2.1-6; inferred intermediate reversal",
        evidence="engineering_assumption",
        scope="Guozhan/Guozhan North shared stabling reversal is assumed, not verified as-built permission"),
    "sz1_luohu": dict(
        url="https://www.szmc.net/szmc_m/zhaobiaozhaoshang/zhaobiaogonggao/gongchenglei/201911/31240.html",
        locator="Section 1.1, Luohu turnback crossover renewal"),
    "sz1_dajuyuan": dict(
        url="https://www.sz.gov.cn/cn/xxgk/zfxxgj/zwdt/content/post_8474012.html",
        locator="2021-01-19, Airport East--Grand Theater temporary service"),
    "sz1_old_termini": dict(
        url="https://pnr.sz.gov.cn/attachment/1/1442/1442495/10537363.pdf",
        locator="Shenzhen Gazetteer, Line 1 history: 2004, 2009 and 2011 termini",
        scope="Historical turnback locations, cross-checked against legacy topology16 routes"),
    "sz1_xixiang": dict(
        url="https://www.szmc.net/szmc_m/home/xinweenzhongxin/gonggaoxinwen/201304/35885.html",
        locator="2013-04-17, Luohu--Xixiang short service"),
    "sz1_gushu": dict(
        url="https://www.people.com.cn/24hour/n/2014/0324/c25408-24714145.html",
        locator="2014-03-24, on-site reporting of Gushu temporary terminal operation",
        scope="Observed emergency operation, cross-checked against legacy topology16 routes"),
    "sz1_legacy_assumption": dict(
        path="linedata.py:_build_topology16",
        locator="huizhanzhongxin_turnback_*, zhuzilin_turnback_*, baoti_turnback routes",
        evidence="engineering_assumption",
        scope="Retained legacy facility locations; current route permissions unverified"),
    "sz10_eia": dict(
        url="https://www.szmc.net/public/cms/public/upload/infor/attach/2015-10-19/1445236278129.pdf",
        locator="Table 3, printed page 4: Liangmaoshan turnback; Mugu crossover; terminal roles",
        scope="Engineering report, not an as-built interlocking route table"),
    "sz10_auxiliary_assumption": dict(
        url="https://www.szmc.net/public/cms/public/upload/infor/attach/2015-10-19/1445236278129.pdf",
        locator="Table 3, printed page 4: Xuexiang North double stabling; Yabao auxiliary crossover",
        evidence="engineering_assumption",
        scope="Xuexiang North = Xuexiang; Yabao = former Innovation Park; reversal permissions inferred"),
    "sz6b_eia": dict(
        url="https://www.szmc.net/public/cms/public/upload/infor/attach/2018-02-24/1519442359039.pdf",
        locator="Section 2.1, Table 2.1-3, printed page 22: Cuihu and Zhongshan University",
        scope="Cuihu = Guangming; Zhongshan University = Zhongda"),
    "sz6b_phase2_eia": dict(
        url="https://www.szmc.net/SMARTC/upload/file/20200923/1600831740837038064.pdf",
        locator="Overview, printed page 1: Guangmingcheng crossovers and turnback"),
    "sz6b_terminus": dict(
        url="https://www.sz.gov.cn/cn/xxgk/zfxxgj/zwdt/content/post_10157093.html",
        locator="2022-10-09, Guangming--SUAT operating endpoints"),
    "sh16_layout": dict(
        url="https://umt1998.tongji.edu.cn/cn/article/pdf/preview/10.16037/j.1007-869x.2022.03.009.pdf",
        locator="Fig. 2 and adjacent text, printed page 45 (PDF page 4)",
        scope="Published engineering schematic; station-level crossover abstraction"),
}


LINE_SPECS = {
    101: dict(key="sz20", name="深圳地铁20号线", fleet=10,
              stations="机场北 国展南 国展 国展北 会展城".split(),
              source="https://www.szmc.net/home/xinwenzhongxin/gongsixinwen/202112/95508.html",
              turnbacks=(("机场北", "terminal", "sz20_eia"),
                         ("国展", "turnback", "sz20_stabling_assumption"),
                         ("国展北", "turnback", "sz20_stabling_assumption"),
                         ("会展城", "terminal", "sz20_eia")),
              shared_turnbacks={"T3_4": ("国展", "国展北")},
              pending_turnbacks=("国展/国展北共用停车线的折返权限为工程推定，待现状进路核验",)),
    102: dict(key="sz1", name="深圳地铁1号线", fleet=60,
              stations="罗湖 国贸 老街 大剧院 科学馆 华强路 岗厦 会展中心 购物公园 香蜜湖 车公庙 竹子林 侨城东 华侨城 世界之窗 白石洲 高新园 深大 桃园 大新 鲤鱼门 前海湾 新安 宝安中心 宝体 坪洲 西乡 固戍 后瑞 机场东".split(),
              source="https://jtys.sz.gov.cn/jtzx/wycx/dtcx/dtxl/content/post_12601089.html",
              turnbacks=(("罗湖", "terminal", "sz1_luohu"),
                         ("大剧院", "crossover", "sz1_dajuyuan"),
                         ("会展中心", "turnback", "sz1_legacy_assumption"),
                         ("竹子林", "turnback", "sz1_legacy_assumption"),
                         ("世界之窗", "crossover", "sz1_old_termini"),
                         ("深大", "turnback", "sz1_old_termini"),
                         ("宝体", "crossover", "sz1_legacy_assumption"),
                         ("西乡", "turnback", "sz1_xixiang"),
                         ("固戍", "turnback", "sz1_gushu"),
                         ("机场东", "terminal", "sz1_old_termini")),
              pending_turnbacks=("会展中心、竹子林、宝体按旧模型工程推定；前海湾/新安出入段线未作为直接折返启用",)),
    103: dict(key="sz10", name="深圳地铁10号线", fleet=48,
              stations="福田口岸 福民 岗厦 岗厦北 莲花村 冬瓜岭 孖岭 雅宝 南坑 光雅园 五和 坂田北 贝尔路 华为 岗头 雪象 甘坑 凉帽山 上李朗 木古 华南城 禾花 平湖 双拥街".split(),
              source="https://jtys.sz.gov.cn/jtzx/wycx/dtcx/dtxl/content/post_12601089.html",
              turnbacks=(("福田口岸", "terminal", "sz10_eia"),
                         ("雅宝", "crossover", "sz10_auxiliary_assumption"),
                         ("雪象", "turnback", "sz10_auxiliary_assumption"),
                         ("凉帽山", "turnback", "sz10_eia"),
                         ("木古", "crossover", "sz10_eia"),
                         ("双拥街", "terminal", "sz10_eia")),
              pending_turnbacks=("雅宝、雪象的折返权限为工程推定，待现状进路核验",)),
    104: dict(key="sz6b", name="深圳地铁6号线支线", fleet=14,
              stations="深理工 中大 圳美 光明 碧眼 虹桥公园 光明城".split(),
              source="https://jtys.sz.gov.cn/jtzx/wycx/dtcx/dtxl/content/post_12601089.html",
              turnbacks=(("深理工", "terminal", "sz6b_terminus"),
                         ("中大", "turnback", "sz6b_eia"),
                         ("光明", "crossover", "sz6b_eia"),
                         ("光明城", "terminal", "sz6b_phase2_eia")),
              pending_turnbacks=()),
    105: dict(key="sh16", name="上海地铁16号线", fleet=26,
              stations="龙阳路 华夏中路 罗山路 周浦东 鹤沙航城 航头东 新场 野生动物园 惠南 惠南东 书院 临港大道 滴水湖".split(),
              source="https://umt1998.tongji.edu.cn/cn/article/pdf/preview/10.16037/j.1007-869x.2022.03.009.pdf",
              turnbacks=(("龙阳路", "terminal", "sh16_layout"),
                         ("罗山路", "crossover", "sh16_layout"),
                         ("周浦东", "crossover", "sh16_layout"),
                         ("航头东", "crossover", "sh16_layout"),
                         ("新场", "crossover", "sh16_layout"),
                         ("野生动物园", "crossover", "sh16_layout"),
                         ("惠南东", "crossover", "sh16_layout"),
                         ("书院", "crossover", "sh16_layout"),
                         ("临港大道", "crossover", "sh16_layout"),
                         ("滴水湖", "terminal", "sh16_layout")),
              pending_turnbacks=()),
}

# Access stations are evidence-based. These durations are experiment inputs,
# including the external Line-6 movement, rather than measured running times.
DEPOT_SPECS = {
    101: (
        dict(id="D0", name="机场北车辆段", english_name="Jichangbei depot",
             stations=("国展南",), lead_seconds=300,
             source=dict(url="https://www.szmc.net/SMARTC/upload/file/20240202/1706841517912063272.pdf",
                         locator="Section 2.4.1: existing phase-I depot lead, 2.6 km, south of Guozhan South")),
    ),
    102: (
        dict(id="D0", name="竹子林车辆段", english_name="Zhuzilin depot",
             stations=("竹子林",), lead_seconds=120,
             source=dict(url="https://download.gongkong.com/fsfiles/technicalData/201609/2016092713374700001.pdf",
                         locator="Sections 2.1-2.2: Zhuzilin station and depot approach tracks")),
        dict(id="D1", name="前海车辆段", english_name="Qianhai depot",
             stations=("前海湾", "新安"), lead_seconds=180,
             source=dict(url="https://www.szmc.net/shentiejianshe/ditiebaohu/anbaoqu/202003/77099.html",
                         locator="Line-1 protection-area drawings, Qianhaiwan/Xin'an depot approaches",
                         corroboration="https://zh.wikipedia.org/wiki/前海车辆段")),
    ),
    103: (
        dict(id="D0", name="益田停车场", english_name="Yitian stabling yard",
             stations=("福田口岸",), lead_seconds=120,
             source=dict(url="https://www.szmc.net/shentiejianshe/ditiebaohu/anbaoqu/202008/83723.html",
                         locator="Futian Checkpoint--Yitian stabling yard protection-area drawing")),
        dict(id="D1", name="凉帽山车辆段", english_name="Liangmaoshan depot",
             stations=("甘坑", "凉帽山"), lead_seconds=180,
             source=dict(url="https://ggzyfw.beijing.gov.cn/cmsbj/u/cms/cn.gov.bjggzyfw.www/202510/8944555287104.pdf",
                         locator="Contractor's Line-10 project description: Gankeng and Liangmaoshan access leads")),
    ),
    104: (
        dict(id="D0", name="长圳车辆段", english_name="Changzhen depot",
             stations=("光明",), lead_seconds=900, external=True, via="Line 6",
             source=dict(url="https://www.szmc.net/public/cms/public/upload/infor/attach/2018-02-24/1519442359039.pdf",
                         locator="Section 2.1, printed page 22: Cuihu (Guangming) connecting crossover to Line 6 and Changzhen depot")),
    ),
    105: (
        dict(id="D0", name="川杨河停车场", english_name="Chuanyanghe yard",
             stations=("罗山路",), lead_seconds=180,
             source=dict(url="https://umt1998.tongji.edu.cn/cn/article/pdf/preview/10.16037/j.1007-869x.2022.03.009.pdf",
                         locator="Fig. 2, printed page 45: Chuanyanghe approach at Luoshan Road")),
        dict(id="D1", name="治北停车场", english_name="Zhibei yard",
             stations=("惠南东",), lead_seconds=180,
             source=dict(url="https://www.cnssce.org/32/201205/1478.html",
                         locator="2012-05-18 project report: Zhibei yard introduced from East Huinan",
                         corroboration="https://umt1998.tongji.edu.cn/cn/article/pdf/preview/10.16037/j.1007-869x.2022.03.009.pdf")),
    ),
}


def depot_sites(tid):
    """Named storage facilities and station-level access, shared by figures and execution."""
    spec = LINE_SPECS[tid]
    n = len(spec["stations"])
    result = []
    for index, raw in enumerate(DEPOT_SPECS[tid]):
        site = copy.deepcopy(raw)
        indices = [spec["stations"].index(name) for name in site["stations"]]
        platforms = [direction * n + i + 1 for i in indices for direction in (0, 1)]
        site.update(node_id=2*n+index+1, station_indices=indices,
                    entry_node_ids=platforms, exit_node_ids=platforms,
                    external=site.get("external", False),
                    access_basis="station-level bidirectional access abstraction, not surveyed interlocking permissions",
                    duration_basis="assumed aggregate one-way seconds, excluding preparation and inspection")
        result.append(site)
    return result


def depot_segment_times(tid):
    return {pair: site["lead_seconds"] for site in depot_sites(tid)
            for pair in ([(node, site["node_id"]) for node in site["entry_node_ids"]]
                         + [(site["node_id"], node) for node in site["exit_node_ids"]])}


def turnback_sites(tid):
    """Return the enabled macroscopic facilities and their engineering provenance."""
    spec = LINE_SPECS[tid]
    n = len(spec["stations"])
    shared = {}
    for resource, names in spec.get("shared_turnbacks", {}).items():
        for name in names:
            if name in shared:
                raise ValueError(f"Turnback station assigned to multiple facilities: {name}")
            shared[name] = resource
    sites, seen = [], set()
    for name, kind, source in spec["turnbacks"]:
        if name not in spec["stations"] or name in seen:
            raise ValueError(f"Invalid or duplicate turnback station: {name}")
        if kind not in {"terminal", "crossover", "turnback"}:
            raise ValueError(f"Unknown turnback facility type: {kind}")
        i = spec["stations"].index(name)
        if (kind == "terminal") != (i in (0, n - 1)):
            raise ValueError(f"Terminal classification disagrees with station order: {name}")
        seen.add(name)
        directions = (2,) if i == 0 else (1,) if i == n - 1 else (1, 2)
        sites.append(dict(station_name=name, station_index=i, node_id=i + 1,
                          platform_ids=(i + 1, n + i + 1),
                          incoming_directions=directions, kind=kind,
                          resource_id=shared.get(name, f"T{i + 1}"), capacity=1,
                          capacity_basis="one conflicting maneuver in the station-level abstraction",
                          evidence=TURNBACK_SOURCES[source].get("evidence", "published_facility"),
                          route_scope="assumed macroscopic reversal, not a reproduced interlocking route",
                          source=copy.deepcopy(TURNBACK_SOURCES[source])))
    if not {spec["stations"][0], spec["stations"][-1]} <= seen:
        raise ValueError("Both operating termini need explicit turnback facilities")
    if not set(shared) <= seen:
        raise ValueError("Shared turnback resource refers to an undeclared facility")
    return sorted(sites, key=lambda site: site["station_index"])


def line_id(value):
    if str(value).isdigit() and int(value) in LINE_SPECS:
        return int(value)
    for tid, spec in LINE_SPECS.items():
        if value == spec["key"]:
            return tid
    raise ValueError(f"Unknown named line: {value}")


def platform_names(tid):
    n = len(LINE_SPECS[tid]["stations"])
    return {f"s{direction}_{i}" for direction in (1, 2) for i in range(n)}


def build_line(tid):
    from topology.route import Route, RouteType
    from topology.node_type import NodeType
    spec = LINE_SPECS[tid]
    n = len(spec["stations"])
    forward = [Platform(i + 1, f"s1_{i}") for i in range(n)]
    reverse = [Platform(n + i + 1, f"s2_{i}") for i in range(n)]
    sites = depot_sites(tid)
    depots = [Platform(site["node_id"], f"depot_{i}", dwell=0) for i, site in enumerate(sites)]
    for direction, row in ((1, forward), (2, reverse)):
        for i, p in enumerate(row):
            p.node_type = NodeType.PLATFORM
            p.station_name = spec["stations"][i]
            p.direction = direction
    for p, site in zip(depots, sites):
        p.node_type = NodeType.SWITCH
        p.station_name = site["name"]
    routes = [Route(1, "forward_service", forward, route_type=RouteType.NON_CIRCULAR),
              Route(2, "reverse_service", list(reversed(reverse)), route_type=RouteType.NON_CIRCULAR)]
    for site in turnback_sites(tid):
        i = site["station_index"]
        for direction in site["incoming_directions"]:
            a, b = (forward[i], reverse[i]) if direction == 1 else (reverse[i], forward[i])
            routes.append(Route(len(routes) + 1, f"turnback_{a.id}_{b.id}", [a, b],
                                route_type=RouteType.NON_CIRCULAR))
    by_id = {p.id: p for p in [*forward, *reverse]}
    for depot, site in zip(depots, sites):
        for node in site["entry_node_ids"]:
            routes.append(Route(len(routes) + 1, f"depot_entry_{depot.id}_{node}",
                                [by_id[node], depot], route_type=RouteType.NON_CIRCULAR))
        for node in site["exit_node_ids"]:
            routes.append(Route(len(routes) + 1, f"depot_exit_{depot.id}_{node}",
                                [depot, by_id[node]], route_type=RouteType.NON_CIRCULAR))
    return [*forward, *reverse, *depots], routes, list(zip(forward, reverse))


def dispatch_configuration(tid, fleet=None):
    """Fixed initial stock; two reserves use the line's actual storage facilities."""
    spec = LINE_SPECS[tid]
    n = len(spec["stations"])
    fleet = spec["fleet"] if fleet is None else int(fleet)
    if not 2 <= fleet <= 2 * n + 2:
        raise ValueError("Named-line initialization needs 2..2N+2 vehicles")
    sites = depot_sites(tid)
    depots = {site["id"]: dict(site, exit_node_id=site["exit_node_ids"][0], capacity=max(4, fleet),
              preparation_seconds=60, inspection_seconds=300, inspection_capacity=1) for site in sites}
    service_count = fleet - 2
    vehicles = {f"agent_{i}": dict(initial_node=1 + (i * (2 * n)) // max(1, service_count))
                for i in range(service_count)}
    for i in range(2):
        vehicles[f"agent_{service_count + i}"] = dict(role="reserve", depot=sites[i % len(sites)]["id"])
    tasks = {"forward_replacement": dict(stops=list(range(1, n + 1)), purpose="replacement"),
             "reverse_replacement": dict(stops=list(range(2 * n, n, -1)), purpose="replacement")}
    turns = sorted(site["node_id"] for site in turnback_sites(tid))
    for first, last in combinations(turns, 2):
        if (first, last) == (1, n):
            continue
        forward = list(range(first, last + 1))
        reverse = list(range(n + last, n + first - 1, -1))
        # Reusable patterns, not a single shared vehicle assignment. Each
        # selected vehicle receives its own rotated, persistent service task.
        for direction, stops in (("forward", forward + reverse + [first]),
                                  ("reverse", reverse + forward + [n + last])):
            tasks[f"section_{first}_{last}_{direction}"] = dict(stops=stops, purpose="section_service",
                required=False, pattern=True, repeat=True)
    return dict(depots=depots, vehicles=vehicles,
        service_tasks=tasks, reserve_policy="contingency",
        clearance_seconds=30, coupling_seconds=60, turnback_seconds=60, tow_multiplier=2.,
        metadata=dict(line_key=spec["key"], station_source=spec["source"],
            layout_revision=LAYOUT_REVISION, turnback_sites=turnback_sites(tid),
            depot_sites=sites,
            pending_turnbacks=list(spec["pending_turnbacks"]),
            track_layout="two directed tracks; explicit turnbacks; named depot access stations; aggregated external access where declared",
            parameters="synthetic fleet, original seeded segment-time generation, 30s dwell, generated passenger demand",
            reference="station-level abstraction of cited facilities; not a surveyed or complete interlocking track plan"))


def event_resources(tid):
    _, routes, _ = build_line(tid)
    n = len(LINE_SPECS[tid]["stations"])
    sites = {site["node_id"]: site for site in turnback_sites(tid)}
    turnbacks = {}
    switches = {}
    for r in routes:
        a, b = r.node_list[0].id, r.node_list[-1].id
        if r.name.startswith("turnback"):
            station = min(a, b)
            site = sites[station]
            facility = turnbacks.setdefault(site["resource_id"], dict(
                segments=[], capacity=site["capacity"], station_names=[],
                facility_type=site["kind"], evidence=site["evidence"],
                source=copy.deepcopy(site["source"])))
            facility["segments"].append(f"{a}-{b}")
            if site["station_name"] not in facility["station_names"]:
                facility["station_names"].append(site["station_name"])
            switch = switches.setdefault(f"S{station}", dict(node_id=station, route_ids=[],
                route_entry_nodes={}, station_name=site["station_name"]))
            switch["route_ids"].append(r.id)
            switch["route_entry_nodes"][str(r.id)] = a
    depots = {site["id"]: dict(name=site["name"],
                entry_segments=[f"{node}-{site['node_id']}" for node in site["entry_node_ids"]],
                exit_segments=[f"{site['node_id']}-{node}" for node in site["exit_node_ids"]])
              for site in depot_sites(tid)}
    shared_routes = {f"depot_{key}_access": dict(capacity=1,
                        segments=value["entry_segments"] + value["exit_segments"],
                        name=value["name"], capacity_basis="one aggregate conflicting access movement")
                     for key, value in depots.items()}
    return dict(turnbacks=turnbacks, switches=switches, depots=depots, shared_routes=shared_routes)
