"""Rebuild the analysis criterion from a locally obtained NEGex 2.0 release.

The public manifest freezes membership and source-card identifiers. This builder
never needs individual expert responses and does not attempt to repeat the
restricted-data sample-selection calculation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAITS = {
    "extraversion": ("bfi_extrav", "bfi_c1", "bfi_c6", "numrespt_bfi"),
    "agreeableness": ("bfi_agree", "bfi_c7", "bfi_c2", "numrespt_bfi"),
    "conscientiousness": ("bfi_consc", "bfi_c3", "bfi_c8", "numrespt_bfi"),
    "emotional_stability": ("bfi_emot", "bfi_c9", "bfi_c4", "numrespt_bfi"),
    "openness": ("bfi_open", "bfi_c5", "bfi_c10", "numrespt_bfi"),
    "narcissism": ("triad_narciss", "triad_c1", "triad_c4", "numrespt_triad"),
    "psychopathy": ("triad_psycho", "triad_c2", "triad_c5", "numrespt_triad"),
    "machiavellianism": ("triad_machiav", "triad_c3", "triad_c6", "numrespt_triad"),
}
MANIFEST_COLUMNS = [
    "leader_id", "full_name", "country", "office", "arm", "identity",
    "negex_ids", "is_bridge", "election_year", "election_type",
]


def normalize_actor(value: object) -> str:
    text = "".join(
        c for c in unicodedata.normalize("NFKD", str(value).lower())
        if not unicodedata.combining(c)
    )
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def map_source_cards(candidates: pd.DataFrame, elections: pd.DataFrame) -> pd.DataFrame:
    """Resolve source slots by election and normalized name, never candidate rank."""
    required = {"electID", "actor"}
    missing = required - set(candidates)
    if missing:
        raise ValueError(f"Candidate file lacks columns: {sorted(missing)}")
    if "electID" not in elections:
        raise ValueError("Election file lacks electID")
    wide_rows = []
    for row in elections.to_dict("records"):
        for slot in range(1, 11):
            value = row.get(f"candidate{slot}")
            if isinstance(value, str) and value.strip():
                wide_rows.append({
                    "electID": str(row["electID"]),
                    "actor_n": normalize_actor(value),
                    "source_id": f"{row['electID']}#cand{slot}",
                })
    if not wide_rows:
        raise ValueError("Election file has no candidate1..candidate10 names")
    wide = pd.DataFrame(wide_rows)
    cand = candidates.copy()
    cand["electID"] = cand["electID"].astype(str)
    cand["actor_n"] = cand["actor"].map(normalize_actor)
    mapped = wide.merge(cand, on=["electID", "actor_n"], how="left", validate="one_to_one")
    if mapped["source_id"].duplicated().any():
        raise ValueError("Ambiguous source-card identifiers")
    return mapped.set_index("source_id", drop=False)


def trait_statistics(cards: pd.DataFrame, trait: str) -> tuple[float, float, float]:
    """Match the executed criterion's mean, component-SD approximation and count.

    Counts are public battery-completer counts, not the restricted per-item
    counts used when membership was originally selected.
    """
    mean_col, item_a, item_b, count_col = TRAITS[trait]
    needed = [mean_col, count_col, f"sd{item_a}", f"sd{item_b}"]
    absent = set(needed) - set(cards)
    if absent:
        raise ValueError(f"Candidate file lacks columns: {sorted(absent)}")
    subset = cards.loc[cards[mean_col].notna()]
    if subset.empty:
        return np.nan, np.nan, np.nan
    counts = pd.to_numeric(subset[count_col], errors="coerce").fillna(0.0).astype(float)
    weights = counts.copy()
    if weights.sum() <= 0:
        weights[:] = 1.0
    means = pd.to_numeric(subset[mean_col], errors="raise").astype(float)
    mean = round(float((means * weights).sum() / weights.sum()), 6)
    n = int(round(float(counts.sum())))
    sd_a = pd.to_numeric(subset[f"sd{item_a}"], errors="coerce").astype(float)
    sd_b = pd.to_numeric(subset[f"sd{item_b}"], errors="coerce").astype(float)
    card_sd = np.sqrt(sd_a**2 + sd_b**2) / 2
    valid = card_sd.notna() & (counts > 1)
    sd = (
        round(float(np.average(card_sd[valid], weights=counts[valid])), 6)
        if n > 1 and valid.any() else np.nan
    )
    return mean, sd, n


def build_criterion(manifest: pd.DataFrame, candidates: pd.DataFrame,
                    elections: pd.DataFrame) -> pd.DataFrame:
    missing = set(MANIFEST_COLUMNS) - set(manifest)
    if missing:
        raise ValueError(f"Sample manifest lacks columns: {sorted(missing)}")
    if manifest["leader_id"].duplicated().any():
        raise ValueError("Sample manifest has duplicate leader identifiers")
    source = map_source_cards(candidates, elections)
    rows = []
    for r in manifest[MANIFEST_COLUMNS].to_dict("records"):
        if not isinstance(r["negex_ids"], str) or not r["negex_ids"].strip():
            raise ValueError(f"No source cards for {r['leader_id']}")
        ids = r["negex_ids"].split(";")
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate source card for {r['leader_id']}")
        missing_ids = set(ids) - set(source.index)
        if missing_ids:
            raise ValueError(f"Unmatched source cards: {sorted(missing_ids)}")
        cards = source.loc[ids]
        if cards["actor"].isna().any():
            raise ValueError(f"Candidate name could not be resolved for {r['leader_id']}")
        for trait in TRAITS:
            mean, sd, n = trait_statistics(cards, trait)
            r.update({f"expert_{trait}_mean": mean, f"expert_{trait}_sd": sd,
                      f"expert_{trait}_n": n})
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negex-dir", type=Path, default=ROOT / "data" / "NEGex_2.0")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests" / "sample.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "criterion.csv")
    args = parser.parse_args()
    inputs = [args.negex_dir / "NEGex_2.0_candidates.dta",
              args.negex_dir / "NEGex_2.0_elections.dta"]
    missing = [p.name for p in inputs if not p.is_file()]
    if missing:
        parser.error("Obtain the NEGex 2.0 release first; missing: " + ", ".join(missing)
                     + ". See docs/data-access.md.")
    criterion = build_criterion(
        pd.read_csv(args.manifest),
        pd.read_stata(inputs[0], convert_categoricals=False),
        pd.read_stata(inputs[1], convert_categoricals=False),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    criterion.to_csv(args.output, index=False)
    print(f"Rebuilt {len(criterion)} politicians into the local criterion file.")


if __name__ == "__main__":
    main()
