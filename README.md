# Politician personality ratings with LLMs

Companion to **Reading the Speech or Recalling the Speaker? LLM Personality Ratings of Politicians Against Expert Judgment**, Jesse Kommandeur and Alessandro Dell'Orto. Accepted for *Findings of the Association for Computational Linguistics: EACL 2027*; bibliographic details are provisional. See [CITATION.bib](CITATION.bib).

This package provides statistical analysis code, model-query runners, aggregate reference results, and cleaned model-generated responses for the **NEGex 2.0** analysis: 203 politicians, five models, five evidence conditions, and every framing actually executed. The speech comparison contains 192 speeches by 50 politicians.

It also includes the frozen Wikipedia/Wikidata prominence measurements for those 203 politicians, recorded as **6 July 2026**, so the historical prominence analyses can be rerun without fetching current counts. See [prominence documentation](docs/prominence.md) for definitions and CC0 licensing.

## Install and check

Python 3.11 or newer is required. Run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test,figures]'
python -m pytest
python scripts/audit_release.py
```

The tests and release audit make no model/API calls. Offline reanalysis needs no API key or cloud account. Tests requiring external data or completed recomputations skip by default with an explicit reason. See [VALIDATION.md](VALIDATION.md) for what has been checked.

## Reproduce results from the stored model answers

Speaker-recognition totals require no external data:

```bash
python scripts/run_analysis.py --stage recognition
```

For convergence with experts and the other primary analyses, obtain the public NEGex 2.0 candidate and election files following [data access](docs/data-access.md), then run:

```bash
python scripts/build_inputs.py
python scripts/run_analysis.py
python scripts/identity_baselines.py
python scripts/ideology.py
python scripts/language_checks.py
python scripts/ceiling_sensitivity.py
```

The builder reconstructs the expert criterion locally for the fixed 203-politician sample. Speech-rating reanalysis uses the stored model answers and that criterion; speech transcripts are not needed. Statistical reanalysis is deterministic given the required inputs and specified random seeds. The language checks preserve the original country-to-dominant-language proxy; see [statistical conventions](docs/statistics.md) for its interpretation.

For historical prominence and desirability analyses, the commands use the bundled July measurements by default:

```bash
python scripts/prominence_analysis.py
python scripts/desirability_by_prominence.py
```

The optional `python scripts/ideology.py --metadata` branch uses the same snapshot. Some context-only significance adjustments and office-based uniqueness counts differ from its historical references; the matching portions and remaining differences are documented in [prominence.md](docs/prominence.md).

Statistical outputs go to ignored `outputs/`, preserving the supplied records and references. See [REPRODUCING.md](REPRODUCING.md) for numerical comparison gates, figures, and packaging; [paper-map.md](docs/paper-map.md) maps analyses to result files.

## Collect new model answers

Fresh inference is a separate, optional workflow. It requires your own credentials and explicit `--execute` authorization. The runners support GPT-5.5 through OpenRouter and the four open-weight models through an authenticated vLLM endpoint, including optional Modal deployment.

Read [inference.md](docs/inference.md) for dry runs, dependencies, checkpoint settings, credentials, costs, and output handling. Speech inference additionally requires separately downloaded GPD transcripts and local input reconstruction. New answers need not match the historical responses: hosted aliases, backend changes, and sampling can change outputs. No live model inference or GPU deployment is established by the offline tests.

## What is included

| Directory | Contents |
|---|---|
| `records/` | Original model item ratings and speaker-recognition guesses selected for public release |
| `measurements/` | The frozen, eight-column Wikimedia prominence snapshot for the 203-politician sample |
| `manifests/` | Fixed sample identifiers, selected speeches, preprocessing recipes, and recognition corrections |
| `results/` | Aggregate statistical references; no politician-level expert scores |
| `pinf/` and `scripts/` | Instrument, scoring, analysis, data preparation, model-query, and deployment code |
| `tests/` | Statistical, reconstruction, inference, and release-boundary checks |

`RELEASE_MANIFEST.json` lists every publication file with its size and integrity hash. Downloaded datasets and reconstructed criterion values belong in ignored `data/`; fresh runs, recomputations, and release archives belong in ignored `outputs/`.

## Scope, licensing, and citation

This package covers NEGex 2.0 only. NEGex 3.0 follows the same data structure and is expected to be released in the future.

We do not redistribute NEGex datasets, individual expert responses, politician-level expert criterion values, speech transcripts, raw API caches, or credentials. The criterion for the fixed sample can be reconstructed from public inputs, but independently repeating its original per-item expert-coverage selection requires restricted inputs that are not supplied. The stored model answers do not reconstruct original inference costs and timings.

Our software is MIT-licensed. The Wikimedia measurements are reused under CC0; other external datasets, instruments, and model checkpoints retain their original terms. See [provenance and intended use](docs/provenance.md) and [data access](docs/data-access.md). Model responses are observations about model behavior, not established personality facts about individuals.

For the provisional paper citation, use [CITATION.bib](CITATION.bib) or [CITATION.cff](CITATION.cff). Final publication details will be added when available.
