# AE4350 - Bio-Inspired Intelligence and Learning for Aerospace Applications - Reia Ramkumar (6446426)
- [`pipeline_classes.py`](./pipeline_classes.py) — core classes (env, SNN, SAC, interest map, reward). Imported by everything else, never run directly.
- [`train.py`](./train.py) — generates the 60-case training set -> `all_cases.yaml`. Run first.
- [`sensitivity.py`](./sensitivity.py) — 16-parameter sensitivity sweep -> `snn/sac_sensitivity.yaml` + plots. Run second.
- [`best_config.py`](./best_config.py) — trains/validates/tests base vs best config, deploys winner, generates diagnostics. Run last.
                      (or)

- [`main.py`](./main.py) - runs all files in order

sensitivity plots -> `pipeline_outputs/sensitivity/plots/`
base > best config --> simulation + performance plots for base -> `pipeline_outputs/cases/`, summary -> `pipeline_outputs/best_config/best_config_summary.yaml`

*Bug fix:*
`compute_reward(predicted_state, true_state, ..)`
