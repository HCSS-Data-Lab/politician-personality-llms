# Public record schemas

Each JSONL line is one object. Ratings contain exactly `leader_id`, `model`, `condition`, `framing`, `repeat`, `seed`, `speech_id`, and `ratings`. `ratings` has exactly sixteen integer keys `s1` through `s16`, each on 0–4. Speech ID is null for identity-only conditions. Seeds equal the repeat index (0, 1, 2); request retries in the original runs could use bumped sampling seeds, so the exported seed denotes the planned repeat seed. Original retry/cache bookkeeping is not published.

| Evidence condition | Framings | Records |
|---|---|---:|
| name-context | E, neutral, P | 9,135 |
| name-only | neutral | 3,045 |
| context-only | neutral | 3,045 |
| named-speech | E, neutral, P | 8,640 |
| anon-speech | E, neutral | 5,760 |

Recognition records contain `leader_id`, `model`, `speech_id`, `guess` (name, country, confidence), `auto_label`, `label`, and `partial_kind`. Confidence is the model-reported number, not calibrated certainty. Label is correct, partial or wrong; partial_kind is name or country for partial labels and null otherwise. The audited labels are frozen for the original guesses only.

The sample manifest contains public/source identifiers and the identity/context strings used in the prompts, without expert scores or expert response counts. Source identifiers use the original elections table's candidate slot, **not election rank**. `scripts/build_inputs.py` checks the join through the downloaded elections and candidates tables.
