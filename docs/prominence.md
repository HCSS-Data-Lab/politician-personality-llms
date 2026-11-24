# Frozen Wikipedia/Wikidata prominence measurements

`measurements/prominence_2026-07-06.csv` preserves the exact numerical inputs
used for the public sample's prominence analyses, with the historical snapshot
date recorded as 6 July 2026. It contains 203 rows, one per politician in
`manifests/sample.csv`; no expert responses or personality criterion values
are included. The export contains eight explicitly allowlisted columns:

| Column | Meaning |
|---|---|
| `leader_id` | Joins the fixed public sample and model responses |
| `source_stratum` | `c1s`, the public analysis sample |
| `qid` | Wikidata entity identifier |
| `enwiki_title` | English Wikipedia article title used for pageview retrieval |
| `election_year` | Calendar year defining the pageview window |
| `sitelinks` | Count of Wikipedia language-edition links for the entity |
| `pageviews_election_year` | Sum of January–December monthly English Wikipedia pageviews |
| `snapshot_date` | Recorded collection date, not the pageview observation year |

Pageviews use the `en.wikipedia/all-access/all-agents` per-article monthly
endpoint; they include automated traffic and should not be described as counts
of distinct people or human-only readers. Counts concern visits to articles,
not identifiable visitors. Wikipedia sitelinks exclude non-Wikipedia projects.
Missing traffic must not be treated as zero; the snapshot retains the original
observed zero where present.

The source crosswalk preserves article-title and entity corrections made in
the original analysis, including historical article names that differ from
current titles. Source identifiers match `manifests/wiki_sources.csv`. The
historical values were filtered to the public sample without rounding or
refetching. No raw API responses or private matching notes are distributed.

## Licence and provenance

The measurements are reused under **CC0 1.0**, separate from the package's MIT
software licence. Wikidata's structured data and Wikimedia Analytics API data
are available under CC0:

- [Wikidata data licensing](https://www.wikidata.org/wiki/Wikidata:Licensing)
- [Wikimedia Analytics API data licensing](https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/documentation/access-policy.html)
- [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)

Sources: Wikidata `wbgetentities` sitelinks and the Wikimedia Analytics per-article
pageviews API. The recorded date identifies the frozen study input; the file
is not a new measurement collected when this companion was prepared.

## Recompute the historical results

After separately obtaining NEGex 2.0 and rebuilding the local criterion:

```bash
python scripts/prominence_analysis.py
python scripts/desirability_by_prominence.py
python scripts/ideology.py --metadata
```

All three commands default to the frozen CSV. They write only to ignored
`outputs/`. The first two reproduce the original prominence/desirability
aggregates; the optional metadata branch uses the same frozen measurements.
The supplied measurements do not reconstruct the original restricted-data
sample-selection process or provide expert criterion values.

The optional metadata branch has two pre-existing differences from its frozen
references. It now includes GPT context-only gradient rows absent from those
references, changing the corresponding Benjamini–Hochberg adjustment families.
Its two office-based uniqueness rows count blank office strings as complete;
the historical references excluded them (203 versus 119 complete records).
The metadata ridge baseline, criterion/name gradients, and five non-office
uniqueness rows reproduce their references, including the paper's 197 unique
country/party/year combinations. The supplementary tests check those portions
explicitly; passing them does not establish parity of the entire metadata
branch. The historical references and statistical calculations are preserved.

For a fresh retrieval, `python scripts/fetch_prominence.py` writes current
measurements to ignored `data/prominence.csv`, with their actual retrieval date.
Analyze those explicitly with `--prominence data/prominence.csv` on any of the
three commands. Fresh counts can change group membership and statistics; they
must not replace the frozen CSV or be described as historical reproduction.
