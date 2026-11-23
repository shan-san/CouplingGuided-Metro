"""Run and replay a short Shenzhen Line 10 scenario B example without API keys."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("rule", "coordination"), default="rule")
    parser.add_argument("--output", type=Path, help="New directory; defaults to runs/demo-*")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.output:
        out = args.output.resolve()
        out.mkdir(parents=True, exist_ok=False)
    else:
        (root / "runs").mkdir(exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix="demo-", dir=root / "runs"))
    env = {**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1",
           "OPENAI_BASE_URL": "http://127.0.0.1:1/v1", "HF_HUB_OFFLINE": "1"}
    env.pop("OPENAI_API_KEY", None)

    def run(*argv):
        subprocess.run([sys.executable, "-m", *map(str, argv)], cwd=out, env=env, check=True)

    results = out / "results"
    if args.method == "rule":
        run("baseline.benchmark", "--line", "sz10", "--source-line", "sz10",
            "--cases", "B", "--methods", "rule_based", "--steps", 2,
            "--allow-short-test", "--horizon", 1, "--horizon-policy", "fixed",
            "--output", results)
        case = results / "sz10_test_0_combination_B" / "rule_based"
    else:
        run("LLM.main_hierarchical", "--mode", "benchmark", "--topology", "sz10",
            "--dispatch-scenario", "combination_B", "--scenario-seed", 111,
            "--device", "cpu", "--n_steps", 2, "--rl_model_path", root / "MARL_model/sz10.pt",
            "--joint-guidance-mode", "no_guidance", "--ablate_rag",
            "--milp_horizon", 1, "--fixed-planning-horizon", "--milp_time_limit", 5,
            "--coordination-budget", 10, "--output_dir", results)
        case = results
    summary = json.loads((case / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete" or summary["completed_stages"] != 2:
        raise SystemExit(f"Incomplete short run. Inspect {case}")
    run("experiment_results", "--replay", case / "manifest.json")
    print(f"Completed and replayed two stages. New results: {case}")


if __name__ == "__main__":
    main()
