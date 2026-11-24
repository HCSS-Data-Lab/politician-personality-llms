# Companion maintenance

This repository publishes authored code, prompts, aggregate statistical outputs, minimal source/selection identifiers, and field-allowlisted model-generated responses. External datasets and politician-level expert criterion values must remain under ignored `data/`; fresh runs and local results remain under ignored `outputs/`. Never package a working checkout recursively.

The explicitly approved exception is `measurements/prominence_2026-07-06.csv`: eight allowlisted identifier/date/count columns of CC0 Wikimedia measurements for the fixed 203-politician sample. Preserve its historical values and recorded date; fresh measurements belong under ignored `data/` and require explicit analysis overrides. Do not add source API caches, free-text matching notes, expert values, or other external datasets to this exception.

Preserve the executed experimental matrix, questionnaire item bindings, within-battery ordering, scoring, missing-value handling and bootstrap conventions. Original recognition corrections belong only to their original guesses. Fresh inference must remain opt-in, authenticated, independently cached and separated from frozen public records. Tests must never make paid inference calls or deploy cloud resources.

Before a release, run the tests, validate numerical reproduction with separately acquired inputs when available, inspect changes, regenerate the publication manifest, and run the release audit. Package only manifest-listed files. A static scan supplements, rather than replaces, review of content and provenance.
