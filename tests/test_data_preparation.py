"""Synthetic tests of source-card mapping and criterion reconstruction."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.build_inputs import (TRAITS, build_criterion, map_source_cards,
                                 normalize_actor, trait_statistics)


def manifest(ids="E1#cand2;E2#cand1"):
    return pd.DataFrame([{
        "leader_id": "synthetic_politician", "full_name": "Alex Example", "country": "Example",
        "office": "President", "arm": "office-holder",
        "identity": "Alex Example, President of Example", "negex_ids": ids,
        "is_bridge": 1, "election_year": 2020, "election_type": "presidential",
    }])


def candidate(election, name, count, value, sd=1.0, rank=1):
    out = {"electID": election, "actor": name, "c_rank": rank,
           "numrespt_bfi": float(count), "numrespt_triad": float(count)}
    for mean, a, b, _ in TRAITS.values():
        out[mean] = float(value)
        out[f"sd{a}"] = float(sd)
        out[f"sd{b}"] = float(sd)
    return out


def sources():
    # First card is slot 2 but rank 1: joining by rank would choose the wrong person.
    cand = pd.DataFrame([
        candidate("E1", "Other Person", 20, 0, rank=2),
        candidate("E1", "Alex Éxample", 2, 1, sd=1, rank=1),
        candidate("E2", "Alex Example", 6, 3, sd=2, rank=2),
    ])
    elect = pd.DataFrame([
        {"electID": "E1", "candidate1": "Other Person", "candidate2": "Alex Example"},
        {"electID": "E2", "candidate1": "Alex Example", "candidate2": ""},
    ])
    return cand, elect


def test_wide_slot_mapping_and_weighted_composite_sd():
    cand, elect = sources()
    out = build_criterion(manifest(), cand, elect).iloc[0]
    for trait in TRAITS:
        assert out[f"expert_{trait}_mean"] == 2.5
        assert out[f"expert_{trait}_sd"] == round((2 / np.sqrt(2) + 6 * np.sqrt(2)) / 8, 6)
        assert out[f"expert_{trait}_n"] == 8
    assert normalize_actor("  Álex-Éxample ") == "alex example"


def test_missing_trait_uses_only_its_available_cards():
    cand, elect = sources()
    cand.loc[cand.electID == "E2", "bfi_extrav"] = np.nan
    out = build_criterion(manifest(), cand, elect).iloc[0]
    assert out.expert_extraversion_mean == 1
    assert out.expert_extraversion_n == 2
    assert out.expert_agreeableness_mean == 2.5


def test_missing_weights_fall_back_only_if_all_are_missing():
    cand, _ = sources()
    cards = cand.iloc[1:].copy()
    cards["numrespt_bfi"] = [np.nan, 6]
    assert trait_statistics(cards, "extraversion")[0] == 3
    cards["numrespt_bfi"] = np.nan
    mean, sd, n = trait_statistics(cards, "extraversion")
    assert mean == 2 and n == 0 and np.isnan(sd)


def test_single_expert_or_missing_component_does_not_fabricate_sd():
    cards = pd.DataFrame([candidate("E1", "Alex", 1, 2)])
    assert np.isnan(trait_statistics(cards, "extraversion")[1])
    cards["numrespt_bfi"] = 4
    cards["sdbfi_c1"] = np.nan
    assert np.isnan(trait_statistics(cards, "extraversion")[1])


def test_missing_trait_is_missing_not_zero():
    cards = pd.DataFrame([candidate("E1", "Alex", 4, 2)])
    cards["bfi_extrav"] = np.nan
    assert all(np.isnan(v) for v in trait_statistics(cards, "extraversion"))


def test_ambiguous_name_join_fails():
    cand, elect = sources()
    with pytest.raises(pd.errors.MergeError):
        map_source_cards(pd.concat([cand, cand.iloc[[1]]]), elect)


@pytest.mark.parametrize("source_ids", ["E3#cand1", "E1#cand3", "E1#cand2;E1#cand2"])
def test_bad_source_ids_fail(source_ids):
    cand, elect = sources()
    with pytest.raises(ValueError):
        build_criterion(manifest(source_ids), cand, elect)


def test_cli_rebuilds_from_synthetic_stata_files(tmp_path):
    cand, elect = sources()
    cand.to_stata(tmp_path / "NEGex_2.0_candidates.dta", write_index=False, version=118)
    elect.to_stata(tmp_path / "NEGex_2.0_elections.dta", write_index=False, version=118)
    manifest().to_csv(tmp_path / "manifest.csv", index=False)
    destination = tmp_path / "criterion.csv"
    result = subprocess.run([
        sys.executable, "scripts/build_inputs.py", "--negex-dir", str(tmp_path),
        "--manifest", str(tmp_path / "manifest.csv"), "--output", str(destination),
    ], text=True, capture_output=True, check=True)
    assert "Rebuilt 1 politicians" in result.stdout
    rebuilt = pd.read_csv(destination)
    assert rebuilt.loc[0, "expert_extraversion_mean"] == 2.5


def test_local_release_reconstructs_reference_when_explicitly_supplied():
    """Opt-in local check; neither the dataset nor the criterion is distributed."""
    source = os.getenv("NEGEX_LOCAL_DIR")
    reference = os.getenv("NEGEX_LOCAL_REFERENCE")
    if not source or not reference:
        pytest.skip("Set NEGEX_LOCAL_DIR and NEGEX_LOCAL_REFERENCE for the local integration check")
    root = Path(__file__).resolve().parents[1]
    rebuilt = build_criterion(
        pd.read_csv(root / "manifests/sample.csv"),
        pd.read_stata(Path(source) / "NEGex_2.0_candidates.dta", convert_categoricals=False),
        pd.read_stata(Path(source) / "NEGex_2.0_elections.dta", convert_categoricals=False),
    ).set_index("leader_id")
    expected = pd.read_csv(reference).set_index("leader_id").loc[rebuilt.index]
    columns = [f"expert_{t}_{stat}" for t in TRAITS for stat in ("mean", "sd", "n")]
    np.testing.assert_allclose(rebuilt[columns], expected[columns], atol=1e-6, rtol=0, equal_nan=True)
