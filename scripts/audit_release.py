"""Audit the publication allowlist and the contents of the public response export.

This is a conservative static check, not a guarantee that all sensitive information
can be detected automatically. Local datasets and computed outputs are outside the
publication allowlist and must never be included in an archive.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = frozenset({"qwen-8b", "qwen-32b", "gemma-12b", "gemma-27b", "gpt-5.5"})
CONDITIONS = {
    "name-context": ("E", "neutral", "P"),
    "name-only": ("neutral",),
    "context-only": ("neutral",),
    "named-speech": ("E", "neutral", "P"),
    "anon-speech": ("E", "neutral"),
}
RATING_KEYS = frozenset({"leader_id", "model", "condition", "framing", "repeat", "seed", "speech_id", "ratings"})
RECOGNITION_KEYS = frozenset({"leader_id", "model", "speech_id", "guess", "auto_label", "label", "partial_kind"})
SAMPLE_KEYS = frozenset({"leader_id", "full_name", "country", "office", "arm", "identity", "negex_ids", "is_bridge", "election_year", "election_type"})
SPEECH_KEYS = frozenset({"leader_id", "speech_id", "occasion", "language", "original_id"})
LABELS = frozenset({"correct", "partial", "wrong"})
PROMINENCE_FILE = "measurements/prominence_2026-07-06.csv"
PROMINENCE_KEYS = frozenset({"leader_id", "source_stratum", "qid", "enwiki_title",
                            "election_year", "sitelinks", "pageviews_election_year", "snapshot_date"})
RUNTIME_DIRS = frozenset({"data", "outputs", ".venv", "venv", ".git", ".pytest_cache", ".ruff_cache", "__pycache__", "build", "dist"})
RUNTIME_SUFFIXES = (".egg-info",)
TEXT_SUFFIXES = frozenset({".py", ".md", ".toml", ".json", ".jsonl", ".csv", ".yaml", ".yml", ".cff", ".bib", ".txt", ".lock"})
SECRET_PATTERNS = {
    "private user path": re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+"),
    "private temporary path": re.compile(r"/(?:tmp|mnt|media)/[A-Za-z0-9_.-]+"),
    "API token": re.compile(r"\b(?:sk-(?:or-v1-)?[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b"),
    "bearer credential": re.compile(r"\bBearer\s+[A-Za-z0-9_-]{20,}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "personal email": re.compile(r"\b[A-Za-z0-9._%+-]+@(?:outlook|gmail|hotmail|hcss|uva|vu|tudelft)\.[A-Za-z.]+\b", re.I),
    "account endpoint": re.compile(r"https?://[A-Za-z0-9-]+--[A-Za-z0-9-]+\.modal\.run"),
}
EXTENSION_PATTERN = re.compile(r"NEGex\s*[_ -]?3(?:[.]0|_0)|\bn3x[_][_]|roster[_]ext34|criterion[_]237|prominence[_]237", re.I)


class AuditError(ValueError):
    """The candidate release violates its publication contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        require(len(fields) == len(set(fields)), f"duplicate CSV columns: {path.name}")
        return fields, list(reader)


def jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            require(bool(line.strip()), f"blank record: {path.name}:{number}")
            row = json.loads(line)
            require(isinstance(row, dict), f"record must be an object: {path.name}:{number}")
            yield row


def runtime_path(relative: Path) -> bool:
    return any(part in RUNTIME_DIRS or part.endswith(RUNTIME_SUFFIXES) for part in relative.parts)


def audit_manifest(root: Path) -> dict:
    manifest_path = root / "RELEASE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["files"]
    require(isinstance(entries, list) and bool(entries), "publication manifest must list files")
    published = set()
    for entry in entries:
        rel = Path(entry["path"])
        require(not rel.is_absolute() and ".." not in rel.parts, "unsafe publication path")
        require(rel.as_posix() not in published, f"duplicate publication path: {rel}")
        require(not runtime_path(rel), f"runtime path in publication manifest: {rel}")
        path = root / rel
        require(not path.is_symlink() and path.is_file(), f"missing or symlinked publication file: {rel}")
        require(not any(parent.is_symlink() for parent in path.parents if parent != root.parent), f"symlinked publication parent: {rel}")
        require(path.stat().st_size == entry["size"], f"size mismatch: {rel}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], f"hash mismatch: {rel}")
        published.add(rel.as_posix())
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if (path.is_file() or path.is_symlink())
        and not runtime_path(path.relative_to(root))
        and path.name != "RELEASE_MANIFEST.json"
    }
    require(actual == published, f"unlisted/missing publication files: {sorted(actual ^ published)}")
    return {"publication_files": len(published), "publication_bytes": sum(e["size"] for e in entries)}


def audit_text(root: Path) -> None:
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if runtime_path(rel) or not path.is_file():
            continue
        require(not path.is_symlink(), f"symlink in release: {rel}")
        require(path.suffix.lower() not in {".dta", ".xlsx", ".xls", ".docx", ".pdf", ".zip", ".tgz", ".gz", ".parquet", ".pickle", ".pkl"}, f"binary dataset/archive in release: {rel}")
        require(not path.name.startswith(".env") or path.name == ".env.example", f"environment file in release: {rel}")
        if path.suffix in TEXT_SUFFIXES or path.name in {"LICENSE", "Makefile", ".gitignore", ".env.example"}:
            content = path.read_text(encoding="utf-8")
            require("\x00" not in content, f"binary content in text file: {rel}")
            for label, pattern in SECRET_PATTERNS.items():
                require(pattern.search(content) is None, f"{label} detected in {rel}")
            if rel.as_posix() != "README.md":
                require(EXTENSION_PATTERN.search(content) is None, f"out-of-scope dataset reference in {rel}")
        if path.suffix == ".csv" and rel.parts[0] == "results":
            fields, rows = read_csv(path)
            require(not ({"leader_id", "full_name", "identity", "speech_id"} & set(fields)), f"per-person records in aggregate table: {rel}")
            # An aggregate metric such as expert_prediction_r is our result,
            # whereas a per-person expert_<trait>_mean/sd/n is criterion data.
            require(not any(re.fullmatch(r"expert_.+_(?:mean|sd|n)", k)
                            or k.startswith(("bfi_", "triad_", "adj_"))
                            for k in fields), f"criterion columns in aggregate table: {rel}")
            if "stratum" in fields:
                require(all(row["stratum"] == "c1s" or row["stratum"].startswith("bridge:")
                            for row in rows), f"out-of-scope aggregate stratum: {rel}")


def audit_records(root: Path, records_dir: Path | None = None, expected_label_changes: int | None = 52) -> dict:
    records_dir = Path(records_dir) if records_dir is not None else root / "records"
    fields, sample = read_csv(root / "manifests" / "sample.csv")
    require(set(fields) == SAMPLE_KEYS, "sample manifest contains missing or unapproved columns")
    ids = {r["leader_id"] for r in sample}
    require(len(sample) == len(ids) == 203, "sample must contain 203 distinct politicians")
    require(all(x.startswith("c1s__") for x in ids), "sample contains an out-of-scope politician")
    bridge = {r["leader_id"] for r in sample if r["is_bridge"].lower() in {"1", "true", "1.0"}}
    require(len(bridge) == 50, "speech bridge must contain 50 politicians")
    fields, speeches = read_csv(root / "manifests" / "speeches.csv")
    require(set(fields) <= SPEECH_KEYS and {"leader_id", "speech_id", "occasion", "language"} <= set(fields), "unapproved speech manifest columns")
    sid_to_leader = {r["speech_id"]: r["leader_id"] for r in speeches}
    require(len(speeches) == len(sid_to_leader) == 192, "speech manifest must contain 192 distinct speeches")
    require(set(sid_to_leader.values()) == bridge, "speech manifest does not match sample bridge")
    expected = set()
    for condition, framings in CONDITIONS.items():
        units = sid_to_leader.items() if condition in {"named-speech", "anon-speech"} else [(None, leader) for leader in ids]
        for speech, leader in units:
            for model in MODELS:
                for framing in framings:
                    for repeat in range(3):
                        expected.add((leader, model, condition, framing, repeat, speech))
    seen = set()
    counts = Counter()
    for row in jsonl(records_dir / "ratings.jsonl"):
        require(set(row) == RATING_KEYS, "rating record contains missing or unapproved fields")
        require(row["leader_id"] in ids and row["model"] in MODELS, "unknown rating politician/model")
        require(row["condition"] in CONDITIONS and row["framing"] in CONDITIONS[row["condition"]], "unexecuted condition/framing combination")
        require(type(row["repeat"]) is int and row["repeat"] in range(3), "invalid repeat")
        require(type(row["seed"]) is int and row["seed"] == row["repeat"], "seed must equal the executed repeat")
        rating = row["ratings"]
        require(isinstance(rating, dict) and set(rating) == {f"s{i}" for i in range(1, 17)}, "invalid item labels")
        require(all(type(value) is int and 0 <= value <= 4 for value in rating.values()), "ratings must be integers on 0..4")
        key = (row["leader_id"], row["model"], row["condition"], row["framing"], row["repeat"], row["speech_id"])
        require(key not in seen, "duplicate experimental rating cell")
        require(key in expected, "rating cell is outside the executed matrix")
        seen.add(key)
        counts[row["condition"]] += 1
    require(seen == expected and len(seen) == 29625, f"rating coverage mismatch: missing {len(expected - seen)}, unexpected {len(seen - expected)}")
    expected_recognition = {(sid, model) for sid in sid_to_leader for model in MODELS}
    seen_recognition = set()
    changes = 0
    for row in jsonl(records_dir / "recognition.jsonl"):
        require(set(row) == RECOGNITION_KEYS, "recognition record contains missing or unapproved fields")
        require(row["speech_id"] in sid_to_leader and row["leader_id"] == sid_to_leader[row["speech_id"]], "recognition speech/politician mismatch")
        require(row["model"] in MODELS, "unknown recognition model")
        guess = row["guess"]
        require(isinstance(guess, dict) and set(guess) == {"name", "country", "confidence"}, "unapproved guess fields")
        require(isinstance(guess["name"], str) and isinstance(guess["country"], str), "guess name and country must be strings")
        confidence = guess["confidence"]
        require(type(confidence) in {int, float} and 0 <= confidence <= 1, "confidence outside 0..1")
        require(row["auto_label"] in LABELS and row["label"] in LABELS, "unknown recognition label")
        require(row["partial_kind"] in {None, "name", "country"}, "invalid partial-kind annotation")
        require(row["label"] == "partial" or row["partial_kind"] is None, "non-partial guess has partial annotation")
        key = row["speech_id"], row["model"]
        require(key not in seen_recognition, "duplicate recognition cell")
        seen_recognition.add(key)
        changes += row["auto_label"] != row["label"]
    require(seen_recognition == expected_recognition and len(seen_recognition) == 960, "recognition coverage mismatch")
    if expected_label_changes is not None:
        require(changes == expected_label_changes, f"expected {expected_label_changes} analysis-scope label changes, got {changes}")
    return {"politicians": len(ids), "bridge_politicians": len(bridge), "speeches": len(sid_to_leader), "ratings": len(seen), "ratings_by_condition": dict(sorted(counts.items())), "recognition_probes": len(seen_recognition), "recognition_label_changes": changes}


def audit_prominence_snapshot(root: Path) -> dict:
    """Allow only the approved public Wikimedia numeric snapshot and sample."""
    fields, rows = read_csv(root / PROMINENCE_FILE)
    require(set(fields) == PROMINENCE_KEYS, "prominence snapshot has missing or unapproved columns")
    _, sample = read_csv(root / "manifests/sample.csv")
    _, crosswalk = read_csv(root / "manifests/wiki_sources.csv")
    expected = {row["leader_id"]: row for row in sample}
    sources = {row["leader_id"]: row for row in crosswalk}
    ids = {row["leader_id"] for row in rows}
    require(len(rows) == len(ids) == len(expected) == 203 and ids == set(expected),
            "prominence snapshot must match the 203-politician sample exactly")
    for row in rows:
        leader = row["leader_id"]
        require(leader.startswith("c1s__") and row["source_stratum"] == "c1s",
                "out-of-scope prominence politician")
        require(row["snapshot_date"] == "2026-07-06", "incorrect historical snapshot date")
        require(re.fullmatch(r"Q[1-9]\d*", row["qid"]) is not None and bool(row["enwiki_title"]),
                "invalid prominence source identifier")
        require(leader in sources and all(row[key] == sources[leader][key] for key in ("qid", "enwiki_title")),
                "prominence source differs from the published source crosswalk")
        require(row["election_year"] == expected[leader]["election_year"], "prominence election year differs from sample")
        for key in ("sitelinks", "pageviews_election_year"):
            require(re.fullmatch(r"0|[1-9]\d*", row[key]) is not None,
                    "prominence counts must be nonnegative integers, not missing values")
    actual = {p.relative_to(root).as_posix() for p in (root / "measurements").rglob("*") if p.is_file()}
    require(actual == {PROMINENCE_FILE}, "unapproved file in measurements directory")
    return {"historical_prominence_rows": len(rows)}


def audit(root: Path, *, manifest: bool = True) -> dict:
    root = root.resolve()
    report = audit_manifest(root) if manifest else {}
    audit_text(root)
    report.update(audit_records(root))
    report.update(audit_prominence_snapshot(root))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--without-manifest", action="store_true", help="development-only content check before the publication manifest is generated")
    args = parser.parse_args()
    try:
        report = audit(args.root, manifest=not args.without_manifest)
    except (AuditError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Release audit failed: {exc}\n")
    print(json.dumps({"status": "passed", **report}, indent=2))


if __name__ == "__main__":
    main()
