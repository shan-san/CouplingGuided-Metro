# Coupling-guided metro rescheduling

Runnable supporting material for *A Coupling-Guided Framework for Real-Time Metro Rescheduling under Diverse Disturbances and Disruptions*. This release contains the simulator, dispatching methods, seeded evaluation inputs and a Shenzhen Line 10 MARL checkpoint. It contains **no historical experiment outputs**. The examples generate and replay their own decisions.

## Quick start

Use Python 3.11. In a terminal, change to the directory containing this README, then run:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python export_inputs.py --check
python run_demo.py
```

The default example runs the rule-based baseline for two 30-second decisions on Shenzhen Line 10, scenario B. It then replays those decisions from the initial state and checks the objective. A successful run ends with **“Completed and replayed two stages”** and prints the fresh output directory under `runs/`. No GPU, API key or language-model service is needed. For Windows, activate the environment with `.venv\Scripts\activate`.

To exercise the **MARL and coordination** entry point with the included checkpoint:

```sh
python run_demo.py --method coordination
```

This second example disables retrieval and language-model guidance and uses a short computation budget. Both examples cover the first minute, before the disruption starts; they verify execution, not the performance of the full method. Use `--output /path/to/new-directory` to place the results elsewhere. Each run creates its own directory; existing runs are not overwritten.

## Experiment inputs

| Evaluation | Lines | Scenarios |
| --- | --- | --- |
| Main comparison | Shenzhen Line 10 (`sz10`) | B |
| Cross-line | Shenzhen Line 6 Branch (`sz6b`), Shenzhen Line 20 (`sz20`), Shanghai Line 16 (`sh16`) | B |
| Cross-scenario | Shenzhen Line 10 | D, F |
| Combined transfer | The three target lines above | D, F |

[`data/evaluation_cases.json`](data/evaluation_cases.json) contains all 12 setup records: line, case ID, seed, event targets, onset and clearance times, and the times at which reports become available. Times are in seconds after 06:00. The evaluation uses base seed 111, test repeat 0, and 240 decision intervals over two hours. The simulator generates the inputs from the same recipe at runtime; `python export_inputs.py --check` independently regenerates the catalogue and compares it with the included file. To write a new copy, omit `--check`.

[`topology/metro_lines.py`](topology/metro_lines.py) defines stations, facilities and fleet assumptions and identifies the infrastructure sources. Passenger demand and event realizations are simulation inputs, not raw fare-card records. [`scenarios/prepare_dispatch.py`](scenarios/prepare_dispatch.py) defines the scenario combinations and split rules. The included [policy checkpoint](MARL_model/sz10.pt) is an **input** for evaluation; specify it explicitly when evaluating on another line.

## Longer experiments

**Baseline.** To run a two-hour rule-based evaluation, use the native benchmark directly:

```sh
python -m baseline.benchmark --line sz10 --source-line sz10 --cases B \
  --methods rule_based --steps 240 --horizon 5 --horizon-policy fixed \
  --budget 120 --output runs/sz10_b_rule
```

For a small solver-dependent check (not a paper-scale experiment), use:

```sh
python -m baseline.benchmark --line sz10 --source-line sz10 --cases B \
  --methods rh_milp --steps 1 --allow-short-test --n-agents 4 \
  --horizon 1 --horizon-policy fixed --budget 5 --output runs/small_rh_milp
```

MILP methods use IBM CPLEX through DOcplex. A compatible CPLEX installation and entitlement appropriate to the model size are needed for paper-scale MILPs. No solver binary or license is bundled. The downloadable Python runtime depends on platform support.

**Full LLM-guided method.** Install `requirements-guidance.txt` in the same environment. Provide a compatible strategic model endpoint, a small-model endpoint with its matching local adapter/tokenizer, and a source-training case library. Set `OPENAI_API_KEY` through your environment if the strategic service requires it. The example uses environment variables for your own locations and model names:

```sh
python -m LLM.main_hierarchical --mode benchmark --topology sz10 \
  --dispatch-scenario combination_B --scenario-seed 111 --device cpu --n_steps 240 \
  --rl_model_path MARL_model/sz10.pt --joint-guidance-mode full --require-components \
  --llm_base_url "$OPENAI_BASE_URL" --llm_model "$LLM_MODEL" \
  --use_small_model --small-model-backend vllm --small_model_url "$SLM_URL" \
  --small_model_path "$SLM_ADAPTER" --small_model_model "$SLM_MODEL" \
  --rag_db_path "$RAG_DB" --milp_horizon 5 --milp_time_limit 120 \
  --coordination-budget 150 --output_dir runs/sz10_b_full
```

The model services, their weights, credentials, and the historical RAG database are not distributed here. `--require-components` checks the required components instead of treating an unconfigured run as the full method. The full configuration above was not executed as part of packaging. To generate your own training cases, see `python -m MILP.collect_dispatch_data --help` (use the training split, not the evaluation cases).

## Code and outputs

| Task | Entry point |
| --- | --- |
| Inspect scenario inputs and event timings | `python -m scenarios.prepare_dispatch --lines sz10 --scenarios combination_B --details` |
| Train the native MARL policy | `python -m RL.dispatch_training --help` |
| Run and compare online baselines | `python -m baseline.benchmark --help` |
| Run the time-limited full-information reference | `python -m MILP.offline_baseline --help` |
| Configure coordination, ablations and sensitivity runs | `python -m LLM.run_experiment_suite --help` |
| Inspect cost accounting and replay logic | [`dispatch_objective.py`](dispatch_objective.py), [`experiment_results.py`](experiment_results.py) |

The native online runners write `manifest.json` (configuration and provenance), `decisions.jsonl` (executed decisions), `metrics.csv`, and `summary.json`. To check a new run, use `python -m experiment_results --replay /path/to/manifest.json`. Replay checks the recorded trajectory and objective, not wall-clock decision times. No result tables, paper drafts, run logs, solver licenses or private service configuration are required to run the two examples.
