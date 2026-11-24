# Running new model queries

The frozen exports under `records/` contain the model-generated item responses
used for the public analysis sample. New queries are optional and are written
under `outputs/`; they never overwrite the frozen exports.

Install the optional client dependencies:

```bash
python -m pip install -e '.[inference]'
```

## Plan a run without making calls

```bash
python scripts/run_models.py --model qwen-8b --backend vllm
python scripts/run_models.py --model gpt-5.5 --backend openrouter
```

These commands report the planned number of associated rows. They do not create
an API client, require a key, read speech text, deploy a server, or make network
requests. `--limit 3` restricts the selected associated rows. `--conditions` can
select one or more evidence conditions. With no selection, the complete executed
matrix is planned:

| Evidence condition | Framings | Repeats |
|---|---|---|
| `name-context` | `E`, `neutral`, `P` | 3 |
| `name-only` | `neutral` | 3 |
| `context-only` | `neutral` | 3 |
| `named-speech` | `E`, `neutral`, `P` | 3 |
| `anon-speech` | `E`, `neutral` | 3 |

The public framing names map to the executed code's `G`, `O`, `K` respectively.
Add `--recognition` to append one separate speaker-recognition probe per speech.
For the public sample, a complete plan is 5,925 rating associations per model,
plus 192 recognition probes when selected. Identical context prompts share one
request per repeat while retaining an association for every politician. Speech
slots remain independent even when their input text is identical.

## OpenRouter

Set `OPENROUTER_API_KEY` in your own environment without putting its value in the
repository. To run a small live name-context sample:

```bash
python scripts/run_models.py \
  --model gpt-5.5 --backend openrouter \
  --conditions name-context --limit 3 \
  --output outputs/runs/gpt55-example --execute
```

`--execute` explicitly permits network requests and the resulting charges. The
runner uses the original `openai/gpt-5.5` alias, disables reasoning, requests the
strict JSON schema, and sends no temperature parameter for GPT-5.5. This alias
is unversioned: new outputs may differ from those obtained during the study, and
its future availability is outside this package's control.

## Open-weight models with an existing vLLM endpoint

Serve the relevant model revision and settings listed in `scripts/modal_app.py`.
Set `VLLM_API_KEY` to the authentication key for your own endpoint, then run:

```bash
python scripts/run_models.py \
  --model qwen-8b --backend vllm \
  --base-url https://YOUR-ENDPOINT/v1 \
  --conditions name-context --limit 3 \
  --output outputs/runs/qwen8-example --execute
```

Replace `YOUR-ENDPOINT` with your own URL. The runner sends temperature 0.7,
max_tokens 512, sampling seed equal to repeat index, and the strict output
schema. Qwen's thinking mode is disabled through its chat-template setting.
The same instrument and item shuffle are used across backends. Runtime and
hardware differences can still affect generated outputs.

## Optional Modal deployment

Install the deployment dependency and authenticate your own Modal account:

```bash
python -m pip install -e '.[inference,modal]'
modal setup
```

In your Modal account, create a secret named `politician-personality-llms` with
`HF_TOKEN` and `VLLM_API_KEY`. Use your own Hugging Face token, accept the relevant
model access terms, and choose a strong endpoint authentication key. The serving
code requires both variables and does not enable an unauthenticated endpoint.
No account identifiers or credential values are bundled here.

```bash
modal deploy scripts/modal_app.py
modal run scripts/modal_app.py
```

The second command resolves your deployed endpoint URLs. Pass the relevant URL
to `run_models.py` as above. The deployment pins the original model revisions,
precisions, vLLM 0.21.0, A100 sizes, and 20,480-token context length. The
instrumentation compatibility patch is checked against its expected source.
Deploying and querying the endpoints incurs GPU and storage charges in your own
account. Containers scale down after ten idle minutes; explicitly stop the app
when finished:

```bash
modal app stop politician-personality-llms
```

## Speech inputs and output handling

The package does not distribute source transcripts. Before executing speech
conditions, prepare local files as described in the data-access and preprocessing
documentation. The runner expects the exact manifest filenames at:

```text
data/speeches/clean/<speech_id>
data/speeches/anon/<speech_id>
```

It caps each input at 2,028 whitespace-delimited words while preserving whitespace
up to the cut. The complete selected input set is checked before the first paid
request. Existing frozen responses do not require these files for reanalysis.

New output directories contain `ratings.jsonl`, optionally `recognition.jsonl`,
`run_config.json`, request-association hashes, and a parsed-response cache. They
contain no raw API envelopes or copies of the request prompts. All are local
runtime artifacts excluded from the public release manifest. Keep these output
directories private unless you separately audit a new export.

Rerunning the same command resumes completed associated records and cached
requests. The cache fingerprint includes endpoint, model, prompt, seed, decoding
options, and an independent cell identifier for speech requests. Changing an
associated record's prompt or endpoint requires a new output directory. On a
malformed response, the runner retries with sampling seeds incremented by
100,000, leaving the item-order prompt unchanged. It permits up to three parsing
attempts, with bounded retries for transient transport failures.

Fresh recognition guesses receive the automatic scorer's label only. The
historical corrections for the original guesses are not applied to fresh runs;
new guesses require their own audit.

## Verification boundary

`tests/test_runner.py` uses mock clients only. It checks default dry-run behavior,
the condition matrix, source-derived prompt hashes, schema/settings, context
request sharing, independent speech slots, resumption, retry seeds, credential
redaction on failures, and authenticated argv-based serving. No paid inference
or live Modal deployment is performed by the test suite.
