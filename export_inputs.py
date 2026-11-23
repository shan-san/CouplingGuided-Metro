"""Export or verify the twelve seeded evaluation cases; no simulation is run."""
import argparse
import json
from pathlib import Path

from scenarios.prepare_dispatch import events_for, read_scenarios


def build_catalogue():
    cases = []
    lines = {}
    for line in ("sz10", "sz6b", "sz20", "sh16"):
        recipe, entries = read_scenarios(
            line, "test", ["combination_B", "combination_D", "combination_F"],
            repeats=1, seed=111,
        )
        lines[line] = {
            "topology_id": recipe["topology"],
            "line": recipe["line"]["name"],
            "generator_recipe_sha256": recipe["sha256"],
        }
        for entry in entries:
            cases.append({**entry, "line": line, "events": events_for(entry)})
    return {
        "description": "Seeded simulation inputs; event times are seconds after 06:00.",
        "base_seed": 111, "split": "test", "repeat": 0,
        "episode_seconds": recipe["episode_seconds"],
        "decision_seconds": recipe["decision_seconds"],
        "physical_tick_seconds": recipe["physical_tick_seconds"],
        "lines": lines, "cases": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare with the bundled JSON")
    args = parser.parse_args()
    path = Path(__file__).resolve().parent / "data" / "evaluation_cases.json"
    catalogue = build_catalogue()
    if args.check:
        if json.loads(path.read_text(encoding="utf-8")) != catalogue:
            raise SystemExit("Evaluation inputs differ from the current generator.")
        print(f"Verified {len(catalogue['cases'])} evaluation cases.")
    else:
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(catalogue, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(path)
