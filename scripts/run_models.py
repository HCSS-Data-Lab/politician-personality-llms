"""Run the published prompt matrix on your own endpoint; dry-run unless --execute.

No source datasets or account configuration are bundled. New calls are separate from
records/ and do not inherit corrections made to the original recognition guesses.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pinf import instrument as I

MODELS = {
    "qwen-8b": "Qwen/Qwen3-8B",
    "qwen-32b": "Qwen/Qwen3-32B",
    "gemma-12b": "google/gemma-3-12b-it",
    "gemma-27b": "google/gemma-3-27b-it",
    "gpt-5.5": "openai/gpt-5.5",
}
FRAMINGS = {"E": "G", "neutral": "O", "P": "K"}
MATRIX = {
    "name-context": ("E", "neutral", "P"),
    "name-only": ("neutral",),
    "context-only": ("neutral",),
    "named-speech": ("E", "neutral", "P"),
    "anon-speech": ("E", "neutral"),
}
MAX_TOKENS = 512
WORD_CAP = 2028


@dataclass(frozen=True)
class Task:
    leader_id: str
    model: str
    condition: str
    framing: str
    repeat: int
    speech_id: str | None = None

    @property
    def seed(self):
        return self.repeat

    @property
    def identity(self):
        return (self.leader_id, self.model, self.condition, self.framing,
                self.repeat, self.speech_id)


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def plan(sample: list[dict], speeches: list[dict], model: str,
         conditions: Iterable[str] = tuple(MATRIX), recognition: bool = False) -> list[Task]:
    if model not in MODELS:
        raise ValueError("Unknown model")
    ids = {row["leader_id"] for row in sample}
    if len(ids) != len(sample):
        raise ValueError("Duplicate sample identifiers")
    by_leader = {row["leader_id"]: [] for row in sample}
    seen = set()
    for row in speeches:
        if row["leader_id"] not in ids or row["speech_id"] in seen:
            raise ValueError("Invalid speech manifest association")
        if Path(row["speech_id"]).name != row["speech_id"]:
            raise ValueError("Speech identifier must be a filename")
        seen.add(row["speech_id"])
        by_leader[row["leader_id"]].append(row["speech_id"])
    tasks = []
    for condition in conditions:
        if condition not in MATRIX:
            raise ValueError(f"Unknown condition: {condition}")
        for row in sample:
            speech_ids = by_leader[row["leader_id"]] if condition.endswith("speech") else [None]
            for sid in speech_ids:
                for framing in MATRIX[condition]:
                    for repeat in range(3):
                        tasks.append(Task(row["leader_id"], model, condition, framing, repeat, sid))
    if recognition:
        tasks.extend(Task(row["leader_id"], model, "recognition", "neutral", 0, row["speech_id"])
                     for row in speeches)
    return tasks


def cap_words(text: str, n: int = WORD_CAP) -> str:
    if n < 0:
        raise ValueError("Word cap must be nonnegative")
    if n == 0:
        return ""
    spans = list(re.finditer(r"\S+", text))
    return text if len(spans) <= n else text[:spans[n - 1].end()]


def context_only(row: dict) -> str:
    prefix = row["full_name"] + ", "
    if not row["identity"].startswith(prefix):
        raise ValueError(f"Identity lacks expected name prefix for {row['leader_id']}")
    return row["identity"][len(prefix):]


def build_prompt(task: Task, person: dict, root: Path = ROOT) -> str:
    framing = FRAMINGS[task.framing]
    if task.condition == "name-context":
        return I.build_prompt(person["identity"], framing, task.seed)
    if task.condition == "name-only":
        return I.build_prompt(person["full_name"], framing, task.seed)
    if task.condition == "context-only":
        return I.build_prompt(context_only(person), framing, task.seed)
    if task.condition in ("named-speech", "anon-speech", "recognition"):
        kind = "clean" if task.condition == "named-speech" else "anon"
        path = root / "data" / "speeches" / kind / str(task.speech_id)
        text = cap_words(path.read_text(encoding="utf-8"))
        if task.condition == "named-speech":
            return I.build_named_speech_prompt(person["identity"], text, framing, task.seed)
        if task.condition == "anon-speech":
            return I.build_anon_speech_prompt(text, framing, task.seed)
        return I.build_reid_prompt(text)
    raise ValueError("Unknown condition")


def request_options(task: Task, prompt: str, backend: str, effective_seed: int | None = None) -> dict:
    if backend not in {"vllm", "openrouter"}:
        raise ValueError("Unknown backend")
    hf_id = MODELS[task.model]
    is_reid = task.condition == "recognition"
    if backend == "openrouter":
        extra = {
            "response_format": {"type": "json_schema", "json_schema": I.REID_SCHEMA if is_reid else I.RATINGS_SCHEMA},
            "reasoning": {"enabled": False},
        }
    else:
        extra = I.reid_extra_body(hf_id) if is_reid else I.vllm_extra_body(hf_id)
    options = {
        "model": hf_id,
        "messages": [{"role": "user", "content": prompt}],
        "seed": task.seed if effective_seed is None else effective_seed,
        "max_tokens": MAX_TOKENS,
        "extra_body": extra,
    }
    if task.model != "gpt-5.5":
        options["temperature"] = I.TEMPERATURE
    return options


def request_hash(options: dict, base_url: str, task: Task | None = None) -> str:
    value = {"base_url": base_url.rstrip("/"), "request": options}
    # Speech slots were independent experimental calls, including the two slots
    # containing identical text. Only the identity/context baselines share calls.
    if task is not None and task.speech_id is not None:
        value["speech_cell"] = [task.condition, task.speech_id, task.framing, task.repeat]
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _validate_value(task: Task, value: object) -> dict:
    text = json.dumps(value, ensure_ascii=False)
    return I.parse_reid_response(text) if task.condition == "recognition" else I.parse_response(text)


def obtain(client, task: Task, prompt: str, backend: str, base_url: str,
           cache_dir: Path, sleep: Callable[[float], None] = time.sleep) -> dict:
    options = request_options(task, prompt, backend)
    fingerprint = request_hash(options, base_url, task)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (fingerprint + ".json")
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("request_sha256") != fingerprint:
            raise ValueError("Cache fingerprint mismatch")
        return _validate_value(task, cached["value"])
    for parse_attempt in range(3):
        effective_seed = task.seed + 100_000 * parse_attempt
        current = request_options(task, prompt, backend, effective_seed)
        response = None
        for transport_attempt in range(4):
            try:
                response = client.chat.completions.create(**current)
                break
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                retryable = status in {408, 409, 429, 500, 502, 503, 504} or (isinstance(exc, (TimeoutError, ConnectionError)) or type(exc).__name__ in {"APIConnectionError", "APITimeoutError"})
                if not retryable or transport_attempt == 3:
                    raise RuntimeError("Model request failed; inspect endpoint availability and credentials") from None
                sleep(2 ** transport_attempt)
        text = response.choices[0].message.content or ""
        try:
            value = I.parse_reid_response(text) if task.condition == "recognition" else I.parse_response(text)
        except I.ResponseError:
            if parse_attempt == 2:
                raise RuntimeError("Model response failed validation after three sampling seeds") from None
            continue
        payload = {"request_sha256": fingerprint, "effective_seed": effective_seed, "value": value}
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(cache_path)
        return value
    raise RuntimeError("No valid response")


def result_row(task: Task, value: dict, person: dict) -> dict:
    if task.condition == "recognition":
        from pinf.anonymize import score_reid_guess
        label = score_reid_guess(f"{value['name']} {value['country']}", person["full_name"], country=person["country"])
        partial_kind = None
        if label == "partial":
            name_label = score_reid_guess(value["name"], person["full_name"])
            partial_kind = "name" if name_label == "partial" else "country"
        return {"leader_id": task.leader_id, "model": task.model, "speech_id": task.speech_id,
                "guess": value, "auto_label": label, "label": label, "partial_kind": partial_kind}
    return {"leader_id": task.leader_id, "model": task.model, "condition": task.condition,
            "framing": task.framing, "repeat": task.repeat, "seed": task.seed,
            "speech_id": task.speech_id, "ratings": value}


def _row_identity(row: dict):
    if "ratings" not in row:
        return (row["leader_id"], row["model"], "recognition", "neutral", 0, row["speech_id"])
    return (row["leader_id"], row["model"], row["condition"], row["framing"], row["repeat"], row["speech_id"])


def run(tasks: list[Task], sample: list[dict], *, root: Path = ROOT,
        backend: str, base_url: str, output: Path, execute: bool = False,
        client_factory=None, sleep: Callable[[float], None] = time.sleep) -> dict:
    """Run a plan; dependency injection keeps tests entirely offline."""
    # The public frozen exports are never destinations for fresh output.
    output = output.resolve()
    if not output.is_relative_to((root / "outputs").resolve()):
        raise ValueError("Fresh runs must be written below outputs/")
    if not execute:
        return {"planned_associations": len(tasks), "executed_requests": 0, "dry_run": True}
    people = {r["leader_id"]: r for r in sample}
    # Validate every selected input before creating a client or performing paid work.
    prompts = [build_prompt(task, people[task.leader_id], root) for task in tasks]
    signature = {"backend": backend, "base_url": base_url.rstrip("/"),
                 "models": sorted({t.model for t in tasks}), "max_tokens": MAX_TOKENS,
                 "runner_schema": 1}
    output.mkdir(parents=True, exist_ok=True)
    meta = output / "run_config.json"
    if meta.exists() and json.loads(meta.read_text(encoding="utf-8")) != signature:
        raise ValueError("Existing run has incompatible endpoint or model settings; choose a new output directory")
    meta.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
    prior = {}
    for filename in ("ratings.jsonl", "recognition.jsonl"):
        path = output / filename
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                identity = _row_identity(row)
                if identity in prior:
                    raise ValueError("Duplicate associated record in existing run")
                prior[identity] = row
    if client_factory is None:
        from openai import OpenAI
        key_name = "OPENROUTER_API_KEY" if backend == "openrouter" else "VLLM_API_KEY"
        key = os.environ.get(key_name)
        if not key:
            raise ValueError(f"Set {key_name} in your environment")
        client_factory = lambda: OpenAI(base_url=base_url, api_key=key, max_retries=0, timeout=180)
    hashes_path = output / "association_requests.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8")) if hashes_path.exists() else {}
    for task, prompt in zip(tasks, prompts):
        association = json.dumps(task.identity, separators=(",", ":"))
        fingerprint = request_hash(request_options(task, prompt, backend), base_url, task)
        if task.identity in prior and hashes.get(association) != fingerprint:
            raise ValueError("Existing associated record has a different or unverified request; choose a new output directory")
    client = None
    written = 0
    fingerprints = set()
    for task, prompt in zip(tasks, prompts):
        if task.identity in prior:
            _validate_value(task, prior[task.identity].get("ratings", prior[task.identity].get("guess")))
            continue
        if client is None:
            client = client_factory()
        fingerprints.add(request_hash(request_options(task, prompt, backend), base_url, task))
        value = obtain(client, task, prompt, backend, base_url, output / "cache", sleep=sleep)
        row = result_row(task, value, people[task.leader_id])
        filename = "recognition.jsonl" if task.condition == "recognition" else "ratings.jsonl"
        with (output / filename).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        prior[task.identity] = row
        hashes[json.dumps(task.identity, separators=(",", ":"))] = request_hash(request_options(task, prompt, backend), base_url, task)
        hashes_path.write_text(json.dumps(hashes, sort_keys=True) + "\n", encoding="utf-8")
        written += 1
    return {"planned_associations": len(tasks), "written_associations": written,
            "distinct_selected_request_hashes": len(fingerprints), "dry_run": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--backend", choices=("vllm", "openrouter"), required=True)
    parser.add_argument("--base-url", help="Your OpenAI-compatible endpoint, including /v1")
    parser.add_argument("--conditions", nargs="+", choices=MATRIX, default=list(MATRIX))
    parser.add_argument("--recognition", action="store_true", help="Also run one independent probe per speech")
    parser.add_argument("--limit", type=int, help="Limit associated rows, before executing")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "runs" / "new-run")
    parser.add_argument("--execute", action="store_true", help="Permit paid API/GPU requests")
    args = parser.parse_args(argv)
    if (args.model == "gpt-5.5") != (args.backend == "openrouter"):
        parser.error("Use OpenRouter for GPT-5.5 and vLLM for the open-weight panel")
    base_url = args.base_url or ("https://openrouter.ai/api/v1" if args.backend == "openrouter" else None)
    if args.execute and not base_url:
        parser.error("--base-url is required for a live vLLM run")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    sample = read_csv(ROOT / "manifests" / "sample.csv")
    speeches = read_csv(ROOT / "manifests" / "speeches.csv")
    tasks = plan(sample, speeches, args.model, args.conditions, args.recognition)
    if args.limit:
        tasks = tasks[:args.limit]
    result = run(tasks, sample, backend=args.backend, base_url=base_url or "dry-run",
                 output=args.output, execute=args.execute)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
