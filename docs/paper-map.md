# Paper result index

Use named result groups rather than table numbers, which can change during camera-ready layout. Public references are under `results/`; recomputations go to matching `outputs/` directories.

| Result | Command / implementation | Aggregate references |
|---|---|---|
| Name-context convergence, per-trait levels, bias and repeatability | `run_analysis.py --stage name`; `name_sweep_metrics.py`, `contrast_metrics.py` | `metrics/convergence*.csv`, `directional_bias.csv`, `reliability_icc*.csv`, `framing_contrasts.csv` |
| Speech levels, evidence contrasts and cross-model comparisons | `run_analysis.py --stage speech`; `speech_metrics.py`, `speech_model_contrasts.py` | `metrics/speech_convergence.csv`, `evidence_contrasts.csv`, `speech_model_contrasts.csv` |
| Recognition and unrecognized-speech convergence | `run_analysis.py`; audited original per-record labels | `records/recognition.jsonl`, `metrics/reid_recognition*_bridge.csv`, `anon_by_recognition.csv` |
| Name-only/context-only controls | `identity_baselines.py` | `identity_baselines/comparison_*.csv` |
| Ceiling estimate and sensitivity | `expert_ceiling.py`, `ceiling_sensitivity.py` | `expert_ceiling.json`, `metrics/ceiling_sensitivity.csv` |
| Language-proxy sensitivity | `language_checks.py` | `language/*.csv` |
| Ideology-adjusted criterion | `ideology.py` | `ideology/adjusted_convergence_*.csv` |
| Metadata ridge baseline, ideology gradients and uniqueness | `ideology.py --metadata` | `ideology/metadata_baseline.csv`, `ideology_gradients.csv`, `metadata_identity_uniqueness.csv` |
| Prominence tertiles and gap | `prominence_analysis.py` (bundled historical snapshot); `fetch_prominence.py` for explicitly selected fresh measurements | `fame_gradient/fame_gradient_*.csv` |
| Desirability shift by prominence | `desirability_by_prominence.py` | `fame_gradient/desirability_by_prominence.csv` |
| Statistical figures | `make figures` | aggregate references above; output layouts in `outputs/figures/` |

The original reference split identity-baseline files by open versus closed models; the new script emits a combined table covering the same six raters. Tests align by model and trait rather than assuming identical file layout. Most internal aggregate references retain `G/O/K` for framing and `name` for name-context; public response exports use the clearer `E/neutral/P` and `name-context` labels, with an explicit adapter.

The ideology reference is the September analysis and preserves its original bootstrap random-number order. Prominence reproduction uses the bundled historical measurements; fresh retrievals can differ from that snapshot and require an explicit input override. See [prominence.md](prominence.md) for source definitions and licensing. Language results reproduce the original country-based proxy groups. See [statistics.md](statistics.md) for the precise interpretation and these limits.
