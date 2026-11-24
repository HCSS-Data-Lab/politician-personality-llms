"""Historical reference checks must not reject explicitly selected fresh inputs."""
import pandas as pd
import pytest
from scripts import desirability_by_prominence as desirability


def different_results():
    return pd.DataFrame([dict(proxy=proxy, model=model, metric=metric,
                              high_minus_low=9., ci_low=8., ci_high=10.)
                         for proxy, model, metric in desirability.REPORTED])


def test_explicit_alternative_measurements_do_not_require_historical_parity(tmp_path, capsys):
    desirability.verify_historical_results(different_results(), tmp_path / 'fresh.csv')
    assert 'historical comparison not applied' in capsys.readouterr().out


def test_bundled_snapshot_still_enforces_historical_parity():
    with pytest.raises(SystemExit) as exc:
        desirability.verify_historical_results(different_results(), desirability.PROMINENCE)
    assert exc.value.code == 1
