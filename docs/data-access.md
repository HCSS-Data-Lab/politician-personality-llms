# External data and local reconstruction

This package distributes analysis code, the authors' model-generated responses,
aggregate results, and minimal sample/preprocessing manifests. It does not
redistribute external source datasets, expert personality criterion values,
individual expert responses, or speech transcripts. Keep downloaded inputs and
reconstructed data under the ignored `data/` directory.

## NEGex 2.0

Obtain the April 2020 NEGex 2.0 release from the
[official project page](https://www.alessandro-nai.com/negative-campaigning-comparative-data)
or its [OSF record](https://doi.org/10.17605/OSF.IO/MEKWG).
Read its usage disclaimer and cite the dataset as requested:

> Nai, Alessandro. 2023. *The Negative Campaigning Comparative Expert Survey
> (NEGex), Release 2.0.* University of Amsterdam.
> https://doi.org/10.17605/OSF.IO/MEKWG

Place these files locally:

```text
data/NEGex_2.0/NEGex_2.0_candidates.dta
data/NEGex_2.0/NEGex_2.0_elections.dta
```

Then run:

```bash
python scripts/build_inputs.py
```

This creates `data/criterion.csv` for the fixed 203-politician primary sample.
The elections file is necessary: identifiers such as `electID#cand2` refer to
original candidate slots in the wide elections table. Candidate rank or row
order cannot safely substitute for those slots. The builder resolves names
within each election and fails on missing or ambiguous joins.

The sample membership was established using individual expert responses
supplied privately by the dataset author. Those responses are not distributed;
`manifests/sample.csv` preserves the selected source identifiers without
expert counts or personality values. Public battery-completer counts differ
from the per-item coverage used for original selection. The builder therefore
reconstructs the criterion for the frozen membership; it does not independently
repeat that restricted-data selection calculation.

Trait means across repeated elections use the released battery-completer
counts as weights. Trait SDs use the executed study's component approximation,
`sqrt(sd_item_a² + sd_item_b²) / 2`, averaged across usable elections with the
same counts. Undefined SDs remain missing. These are reconstructed locally,
not included in the release. The ideology robustness analysis reads the
release's supplied `adj_*` fields; it does not refit the dataset authors'
original adjustment from individual responses.

## Global Populism Database speeches

Obtain the historical Global Populism Database v2.1 build dated 2025-11-20 from
[Harvard Dataverse](https://doi.org/10.7910/DVN/LFTQEZ). Select the historical
version in the record's version history rather than silently substituting the
latest release. Consult its accompanying license and documentation.

Extract the transcript files into `data/raw_speeches/`. The fixed
`manifests/speeches.csv` lists the 192 speech identifiers used for the 50
politicians in the speech analysis. Follow the speech-preparation instructions
for generating local cleaned and scrubbed inputs. Source transcripts and their
cleaned/scrubbed derivatives are not bundled here. Scrubbing names does not
make these speeches reliably anonymous.

Suggested source citation: Hawkins, Kirk A., Rosario Aguilar, Bruno Castanho
Silva, Erin K. Jenne, Bojana Kocijan, and Cristóbal Rovira Kaltwasser. 2019.
*Measuring Populist Discourse: The Global Populism Database.* Use the Dataverse
record's citation for the downloaded version as well.

## Prominence inputs

The public sample's frozen Wikipedia pageview totals and Wikidata sitelink
counts are supplied in `measurements/prominence_2026-07-06.csv`, under CC0.
These eight allowlisted identifier/date/count columns contain no expert values
or raw API responses. See [prominence.md](prominence.md) for definitions,
licensing, and historical reanalysis. Fresh retrievals belong under ignored
`data/`; they may differ from the frozen counts and must retain their actual
retrieval dates.

## Version verification

`configs/source_checksums.json` records SHA-256 digests of the candidate and election tables used for the study. Compare downloaded files with these digests when exact historical reproduction is intended. The speech recipe separately verifies every selected source transcript and both reconstructed inputs.
