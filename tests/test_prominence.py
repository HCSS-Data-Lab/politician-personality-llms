"""Prominence retrieval tests use synthetic API responses and never the network."""
from datetime import datetime, timezone
import io
import json
import urllib.error
import urllib.parse

import pandas as pd
import pytest

from scripts import fetch_prominence as fp


def months(year=2020, views=10):
    return {"items": [{"timestamp": f"{year}{m:02d}0100", "views": views} for m in range(1, 13)]}


def inputs():
    sample = pd.DataFrame([{"leader_id": "person_one", "election_year": 2020}])
    links = pd.DataFrame([{"leader_id": "person_one", "qid": "Q123", "enwiki_title": "Alex Example/Person"}])
    return sample, links


def entity():
    return {"entities": {"Q123": {"sitelinks": {
        "enwiki": {"title": "Alex Example"}, "frwiki": {},
        "commonswiki": {}, "specieswiki": {}, "metawiki": {},
        "enwiktionary": {}, "wikidatawiki": {},
    }}}}


def test_only_wikipedia_sitelinks_are_counted():
    assert fp.count_wikipedia_sitelinks(entity()["entities"]["Q123"]) == 2
    assert fp.count_wikipedia_sitelinks({"sitelinks": {}}) == 0
    with pytest.raises(ValueError):
        fp.count_wikipedia_sitelinks({"missing": ""})


def test_full_year_url_encoding_and_actual_retrieval_date():
    sample, links = inputs()
    requests = []
    def fetch(url):
        requests.append(url)
        return entity() if "wbgetentities" in url else months()
    result = fp.fetch_prominence(sample, links, fetch=fetch, delay=0)
    assert result.loc[0, "sitelinks"] == 2
    assert result.loc[0, "pageviews_election_year"] == 120
    assert result.loc[0, "snapshot_date"] == datetime.now(timezone.utc).date().isoformat()
    assert result.loc[0, "source_stratum"] == "c1s"
    assert requests[1].endswith("Alex_Example%2FPerson/monthly/2020010100/2020123100")


@pytest.mark.parametrize("payload", [
    {}, {"items": []}, {"items": months()["items"][:-1]},
    {"items": months()["items"] + [months()["items"][0]]}, months(2019),
    {"items": [{"timestamp": "2020010100", "views": -1}]},
])
def test_missing_incomplete_or_invalid_traffic_is_not_zero(payload):
    with pytest.raises(ValueError):
        fp.election_year_pageviews(payload, 2020)


def test_genuine_zero_traffic_requires_twelve_observed_buckets():
    assert fp.election_year_pageviews(months(views=0), 2020) == 0


def test_http_failure_default_fails_and_optional_missing_remains_null():
    sample, links = inputs()
    def fetch(url):
        if "wbgetentities" in url:
            return entity()
        raise urllib.error.HTTPError(url, 404, "Not found", None, None)
    with pytest.raises(ValueError, match="incomplete"):
        fp.fetch_prominence(sample, links, fetch=fetch, delay=0)
    result = fp.fetch_prominence(sample, links, fetch=fetch, delay=0, allow_missing=True)
    assert pd.isna(result.loc[0, "pageviews_election_year"])
    assert result.loc[0, "pageviews_status"] == "HTTPError"
    assert result.loc[0, "sitelinks"] == 2


def test_missing_wikidata_entity_never_becomes_zero():
    sample, links = inputs()
    def fetch(url):
        return {"entities": {"Q123": {"missing": ""}}} if "wbgetentities" in url else months()
    result = fp.fetch_prominence(sample, links, fetch=fetch, delay=0, allow_missing=True)
    assert pd.isna(result.loc[0, "sitelinks"])
    assert result.loc[0, "pageviews_election_year"] == 120


def test_crosswalk_must_match_manifest_and_identifiers_are_valid():
    sample, links = inputs()
    for changed in [links.assign(leader_id="another"), links.assign(qid="not-a-qid"),
                    links.assign(enwiki_title=""), pd.concat([links, links])]:
        with pytest.raises(ValueError):
            fp.fetch_prominence(sample, changed, fetch=lambda _: pytest.fail("Unexpected request"), delay=0)


def test_http_client_uses_caller_user_agent(monkeypatch):
    seen = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return json.dumps({"ok": True}).encode()
    def urlopen(request, timeout):
        seen.append((request, timeout))
        return Response()
    monkeypatch.setattr(fp.urllib.request, "urlopen", urlopen)
    assert fp.get_json("https://example.invalid/test", user_agent="my-research-client") == {"ok": True}
    assert seen[0][0].get_header("User-agent") == "my-research-client"
    assert seen[0][1] == 30


def test_http_retry_is_bounded_and_404_is_not_retried(monkeypatch):
    calls = []
    def failure(request, timeout):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 404, "Not found", None, None)
    monkeypatch.setattr(fp.urllib.request, "urlopen", failure)
    monkeypatch.setattr(fp.time, "sleep", lambda _: pytest.fail("404 must not be retried"))
    with pytest.raises(urllib.error.HTTPError):
        fp.get_json("https://example.invalid/test")
    assert len(calls) == 1
