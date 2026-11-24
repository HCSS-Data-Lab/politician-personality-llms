# Release validation

Validated on 23 November 2026. No paid model requests, live Modal deployments or live prominence retrievals were performed during validation.

The full 120-test integration suite passed using independently acquired local NEGex inputs. Nine subsequently added numerical regression checks also passed: scalar-equivalence tests for batched bootstrap arithmetic, supplementary reference comparisons, and the full name-context model-contrast table. The default test suite skips checks requiring separately acquired data or explicit recomputation; it still runs the exported-record, schema, coverage, privacy, mocked inference and statistical unit tests.

Verified numerical reproduction includes:

- All 4,872 reconstructed expert mean/SD/count cells, including missingness, exactly match the original local criterion. No criterion values are distributed.
- Convergence, directional bias, test-retest reliability, framing/evidence contrasts and recognition-conditioned convergence match the primary aggregate references.
- All 495 name-context model contrasts and 1,320 speech model contrasts match their original references numerically.
- All six identity-baseline aggregate rows and 48 trait rows match exactly.
- All 71 original September ideology aggregate cells match within numerical tolerance; the current six-row paper table matches at displayed precision.
- All three language-proxy reference tables match, including their categorical extrema labels.
- Ceiling sensitivity and prominence/desirability tables match the historical aggregate references using the original numerical measurements. The public sample's frozen Wikimedia counts are now supplied in `measurements/prominence_2026-07-06.csv`; a fresh API retrieval cannot guarantee historical snapshot parity. See [prominence.md](docs/prominence.md) for source definitions and CC0 licensing.
- All 192 original cleaned and scrubbed speech inputs reconstruct with matching SHA-256 digests from separately obtained source transcripts and the supplied offset recipes.

The public-response audit verifies exactly 29,625 rating records, 960 recognition probes and 52 sample-specific label corrections, with no missing/duplicate experimental cells. It enforces field allowlists and rejects added expert fields, unknown models/conditions, invalid item values, credential patterns, private paths, account endpoints, unlisted files and publication symlinks. Fresh-run merging uses the same complete-grid validator while permitting newly generated recognition labels.

Independent reviews covered methodology, data reconstruction, inference behavior, documentation and release privacy. Findings corrected before release included first-run omission of GPT speech records, language aliases, inadequate fresh-merge validation and transcript fragments in an early recipe format. The final recipes contain only source offsets, whitespace and single-character inserts. No transcript sentences or token literals are embedded.

The two statistical figure scripts execute successfully, and the Python project builds as a wheel. This does not establish current hosted-model availability or live serving compatibility; optional inference and deployment remain separately documented and were tested with mocks/source inspection.

`RELEASE_MANIFEST.json` is the publication boundary. Neither the local validation datasets nor generated criterion/analysis/cache files are included. Packaging rechecks every listed file's hash and includes only the audited allowlist.

## Historical Wikimedia snapshot addition — 24 November 2026

The approved snapshot contains exactly 203 rows and eight identifier/date/count
columns, matching the original numerical inputs without rounding or refetching.
An independent review verified source-export equality, sample scope, the
publication boundary, and historical versus fresh-input behavior. The README,
frozen model responses, existing aggregate references, and language checks were
left unchanged. No new model inference or deployment was performed.

Fresh execution using the snapshot reproduced all 36 prominence-tertile rows,
12 tertile-gap rows, and 36 desirability rows exactly: 816 numeric cells. The
36-row metadata ridge baseline matches within floating-point tolerance. The
criterion/name ideology gradients and five non-office uniqueness rows also
match their references; the paper's country/party/year uniqueness remains 197.

Two pre-existing optional-metadata differences were identified and documented:
48 additional GPT context-only gradient rows change the multiple-testing
families (195 overlapping adjusted q-values differ), and two office-based
uniqueness rows count 84 blank strings as complete rather than missing. This
addition does not change those calculations or their historical references.
Supplementary gates explicitly cover the matching portions, not the entire
optional metadata branch. See [prominence.md](docs/prominence.md).
