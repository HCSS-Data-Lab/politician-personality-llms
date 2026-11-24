# Reproducing the analyses

Run commands from the repository root. Offline commands never call a model API or deploy a GPU. Fresh inference is optional and explicitly requires `--execute` and your own credentials.

## Environment and tests

```bash
pip install -e '.[test,figures]'
python -m pytest
python scripts/audit_release.py
```

`requirements-tested.txt` records the environment used for offline validation of this release, separately from the historical model-serving versions in the Modal configuration. Most tests are synthetic, mocked, or operate on the public model responses. Integration tests requiring external data skip by default with an explicit reason.

## Download and prepare external data

Follow [data-access.md](docs/data-access.md). Download the public NEGex 2.0 candidate and election tables into `data/NEGex_2.0/`, then:

```bash
python scripts/build_inputs.py
```

This creates local `data/criterion.csv` for the fixed 203-politician membership list. It redistributes no criterion values. Source candidate slots are resolved through election-table names, not candidate rank. The private expert response file is not required for applying this fixed membership; it would be required to independently re-establish the original per-item coverage filter.

## Recompute statistics from original model responses

```bash
python scripts/run_analysis.py --stage recognition
python scripts/run_analysis.py
python scripts/identity_baselines.py
python scripts/ideology.py
python scripts/language_checks.py
python scripts/ceiling_sensitivity.py
```

Recognition needs no external data. The remaining commands use locally rebuilt NEGex values. Results are written to ignored `outputs/`, leaving `records/` and `results/` unchanged. The main analysis prepares internal compatibility records in `outputs/` from the cleaned public exports, then recomputes convergence, bias, ICC, paired framing/evidence/model contrasts, and recognition-conditioned convergence. Original audited recognition labels are attached to their original responses only.

The identity-baseline and ideology scripts read the public response schema directly. The latter needs the downloaded candidate/election tables to load the public supplied ideology-adjusted scores. See [statistics.md](docs/statistics.md) for scoring, bootstrap settings, missing data and interpretation. Rebuild aggregate ceilings directly from public NEGex with `python scripts/expert_ceiling.py`; a reference aggregate estimate is also supplied in `results/expert_ceiling.json`.

After the main analysis, validate all recomputed primary tables against their frozen references:

```bash
CHECK_RECOMPUTED=1 python -m pytest tests/test_reproduction.py
```

Optional full-source tests also independently rebuild and compare the criterion, identity-baseline and September ideology results:

```bash
NEGEX_LOCAL_DIR=data/NEGex_2.0 python -m pytest tests/test_ideology.py tests/test_identity_baselines.py
```

These tests use the reference aggregate tables. The criterion-builder's separate opt-in integration test additionally accepts `NEGEX_LOCAL_REFERENCE` pointing to a locally held original criterion file; such a file is deliberately not published.

## Speech input reconstruction and new model runs

For offline reanalysis of the speech-rating results, use the frozen model responses and separately reconstructed NEGex criterion; speech transcripts are not required. To run models anew, separately download the historical GPD transcript archive and extract its text files into `data/raw_speeches/`:

```bash
python scripts/prepare_speeches.py
```

The authored preprocessing recipes contain source-character ranges and editing instructions. They require the original source files and verify SHA-256 for each source, cleaned input and scrubbed input before writing. No transcript body is distributed. The recipe reproduces frozen cleaning and substitution decisions without repeating the model-based entity detector. Re-identification demonstrates that these scrubbed texts remain identifiable.

Follow [inference.md](docs/inference.md) for optional OpenRouter/vLLM queries and Modal serving. Each model writes to its own run directory. After all five complete with every condition and recognition probe, combine them and analyze without overwriting the original records:

```bash
python scripts/merge_runs.py outputs/runs/qwen-8b outputs/runs/qwen-32b outputs/runs/gemma-12b outputs/runs/gemma-27b outputs/runs/gpt-5.5 --output outputs/runs/combined
python scripts/run_analysis.py --records-dir outputs/runs/combined
python scripts/identity_baselines.py --records outputs/runs/combined/ratings.jsonl
python scripts/ideology.py --records outputs/runs/combined/ratings.jsonl
```

The main fresh analysis requires the full five-model matrix, not a small pilot. New guesses use their newly generated automatic labels and need a new audit; the original correction table is not transferred. Fresh model outputs need not match the historical responses. Run reference-comparison tests only on analyses of the supplied frozen records.

## Prominence and figures

The historical Wikipedia/Wikidata measurements for the 203-politician public sample are supplied in `measurements/prominence_2026-07-06.csv`. Their recorded snapshot date is 6 July 2026. After rebuilding the local NEGex criterion, run:

```bash
python scripts/prominence_analysis.py
python scripts/desirability_by_prominence.py
python scripts/ideology.py --metadata
```

These commands default to the frozen measurements and write results to ignored `outputs/`. Recompute ceiling sensitivity as above, then use `CHECK_SUPPLEMENTARY=1 python -m pytest tests/test_supplementary_reproduction.py` to compare the supplementary outputs with their references. Source definitions, the CC0 data licence, and the distinction between snapshot date and election-year pageviews are documented in [prominence.md](docs/prominence.md).

The supplementary gates cover full prominence/desirability tables and the metadata ridge baseline, plus the criterion/name gradients and non-office uniqueness rows. Two pre-existing optional-metadata differences remain: additional GPT context-only rows change their multiple-testing adjustments, and blank offices are counted differently in two uniqueness rows. See [prominence.md](docs/prominence.md) for details; these are not silently treated as successful full-branch reproduction.

For a new measurement, `python scripts/fetch_prominence.py` queries public Wikimedia APIs and writes `data/prominence.csv` with the actual retrieval date. Select it explicitly with `--prominence data/prominence.csv` on the three analysis commands. Current counts may differ from the historical snapshot; a fresh retrieval is not historical reproduction and must not overwrite the bundled file. No paid model inference is needed for either workflow.

Recreate the two statistical figure layouts from recomputed aggregates, or from bundled aggregate references when recomputations are absent:

```bash
make figures
```

Figures go to `outputs/figures/`. `results/` contains machine-readable aggregate sources for the tables. Paper-specific rounding and table references are indexed in [paper-map.md](docs/paper-map.md).

## Publication audit and packaging

```bash
python scripts/audit_release.py
python scripts/package_release.py
```

After intentional, reviewed source changes, run `python scripts/update_manifest.py` before the final audit.

The archive at `outputs/politician-personality-llms.tar.gz` contains exactly the hash-verified files in `RELEASE_MANIFEST.json`, plus the manifest itself. It excludes downloaded datasets, local criterion tables, fresh runs, account configuration, caches and generated outputs, even when these are present in a working checkout. Do not replace this with a recursive archive of the working directory.
