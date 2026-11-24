"""Retrieve current Wikipedia/Wikidata prominence measurements into local data.

The shipped crosswalk contains source identifiers only. Retrieval dates describe
these requests, not the historical measurements used for the published paper.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Callable
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UA = "politician-personality-llms/0.1 (academic replication)"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
PAGEVIEWS_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents"
NON_WIKIPEDIA = {
    "commonswiki", "specieswiki", "metawiki", "mediawikiwiki", "wikidatawiki",
    "wikimaniawiki", "sourceswiki", "outreachwiki", "incubatorwiki", "foundationwiki",
}


def get_json(url: str, *, user_agent: str = DEFAULT_UA, attempts: int = 3) -> dict:
    """Fetch JSON with bounded retries. Missing or failed data are never zero."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object")
            return result
        except urllib.error.HTTPError as error:
            if error.code not in {408, 429, 500, 502, 503, 504} or attempt + 1 == attempts:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt + 1 == attempts:
                raise
        time.sleep(2**attempt)
    raise RuntimeError("Request retries exhausted")


def count_wikipedia_sitelinks(entity: dict) -> int:
    if "missing" in entity or "sitelinks" not in entity:
        raise ValueError("Missing Wikidata entity or sitelinks")
    links = entity["sitelinks"]
    if not isinstance(links, dict):
        raise ValueError("Invalid sitelink object")
    return sum(key.endswith("wiki") and key not in NON_WIKIPEDIA for key in links)


def election_year_pageviews(payload: dict, year: int) -> int:
    """Require all twelve distinct monthly buckets for the specified year."""
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("No monthly pageview data")
    months = {}
    for item in items:
        stamp, views = str(item.get("timestamp", "")), item.get("views")
        if not re.fullmatch(r"\d{10}", stamp) or stamp[:4] != str(year):
            raise ValueError("Unexpected pageview month")
        month = int(stamp[4:6])
        if month not in range(1, 13) or month in months:
            raise ValueError("Duplicate or invalid pageview month")
        if not isinstance(views, int) or isinstance(views, bool) or views < 0:
            raise ValueError("Invalid pageview count")
        months[month] = views
    if set(months) != set(range(1, 13)):
        raise ValueError("Incomplete election-year pageviews; require January through December")
    return sum(months.values())


def fetch_prominence(sample: pd.DataFrame, crosswalk: pd.DataFrame, *,
                     fetch: Callable[[str], dict] = get_json, delay: float = 0.2,
                     allow_missing: bool = False) -> pd.DataFrame:
    """Retrieve values for the manifest, retaining failures as explicit missing values."""
    if delay < 0:
        raise ValueError("delay must be non-negative")
    required = {"leader_id", "qid", "enwiki_title"}
    if required - set(crosswalk):
        raise ValueError("Wiki crosswalk lacks leader_id, qid or enwiki_title")
    if not {"leader_id", "election_year"} <= set(sample):
        raise ValueError("Sample manifest lacks identifiers or election years")
    if sample.leader_id.duplicated().any() or crosswalk.leader_id.duplicated().any():
        raise ValueError("Duplicate politician identifiers")
    if set(sample.leader_id) != set(crosswalk.leader_id):
        raise ValueError("Crosswalk membership differs from the sample")
    merged = sample[["leader_id", "election_year"]].merge(
        crosswalk[list(required)], on="leader_id", validate="one_to_one"
    ).sort_values("leader_id")
    if not merged.qid.map(lambda q: isinstance(q, str) and bool(re.fullmatch(r"Q[1-9]\d*", q))).all():
        raise ValueError("Crosswalk contains an invalid Wikidata identifier")
    if merged.enwiki_title.isna().any() or merged.enwiki_title.eq("").any():
        raise ValueError("Crosswalk contains an empty Wikipedia title")
    if not merged.election_year.map(lambda y: pd.notna(y) and float(y).is_integer()).all():
        raise ValueError("Invalid election year")

    fetched_date = datetime.now(timezone.utc).date().isoformat()
    entities = {}
    failures = {}
    qids = sorted(set(merged.qid))
    for start in range(0, len(qids), 50):
        batch = qids[start:start + 50]
        url = WIKIDATA_API + "?" + urllib.parse.urlencode({
            "action": "wbgetentities", "ids": "|".join(batch),
            "props": "sitelinks", "format": "json",
        })
        try:
            payload = fetch(url)
            if not isinstance(payload.get("entities"), dict):
                raise ValueError("No Wikidata entities returned")
            for qid in batch:
                try:
                    entities[qid] = count_wikipedia_sitelinks(payload["entities"].get(qid, {}))
                except ValueError as error:
                    failures[qid] = str(error)
        except (OSError, ValueError) as error:
            for qid in batch:
                failures[qid] = type(error).__name__
        if delay:
            time.sleep(delay)

    rows = []
    for r in merged.itertuples(index=False):
        year = int(r.election_year)
        encoded = urllib.parse.quote(r.enwiki_title.replace(" ", "_"), safe="")
        url = f"{PAGEVIEWS_API}/{encoded}/monthly/{year}010100/{year}123100"
        views = None
        pageviews_status = "ok"
        try:
            views = election_year_pageviews(fetch(url), year)
        except (OSError, ValueError) as error:
            pageviews_status = type(error).__name__
        rows.append({
            "leader_id": r.leader_id, "source_stratum": "c1s",
            "qid": r.qid, "enwiki_title": r.enwiki_title, "election_year": year,
            "sitelinks": entities.get(r.qid), "pageviews_election_year": views,
            "sitelinks_status": "ok" if r.qid in entities else failures.get(r.qid, "missing"),
            "pageviews_status": pageviews_status, "snapshot_date": fetched_date,
        })
        if delay:
            time.sleep(delay)
    out = pd.DataFrame(rows)
    missing = out[["sitelinks", "pageviews_election_year"]].isna().any(axis=1)
    if missing.any() and not allow_missing:
        raise ValueError(
            f"Prominence retrieval incomplete for {int(missing.sum())} politicians; "
            "no output written. Retry or use --allow-missing to preserve explicit missing values."
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests" / "sample.csv")
    parser.add_argument("--crosswalk", type=Path, default=ROOT / "manifests" / "wiki_sources.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "prominence.csv")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between requests in seconds")
    parser.add_argument("--user-agent", default=os.environ.get("WIKIMEDIA_USER_AGENT", DEFAULT_UA),
                        help="Set your own identifiable User-Agent if required by Wikimedia")
    args = parser.parse_args()
    sample = pd.read_csv(args.manifest)
    crosswalk = pd.read_csv(args.crosswalk)
    try:
        result = fetch_prominence(
            sample, crosswalk, fetch=lambda url: get_json(url, user_agent=args.user_agent),
            delay=args.delay, allow_missing=args.allow_missing,
        )
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Retrieved prominence for {len(result)} politicians; date {result.snapshot_date.iloc[0]}.")
    print("These are fresh measurements; they are not the paper's historical snapshot.")


if __name__ == "__main__":
    main()
