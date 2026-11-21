import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from bayes.data import calce
from bayes.regression import evaluate, hierarchical as hier

FEATURES = ["a", "b"]


@pytest.fixture(scope="module")
def cells() -> pd.DataFrame:
    generator = np.random.default_rng(1)
    rows = []
    for chemistry, level, rate in (("X", 2.5, 1.0), ("Y", 1.2, 3.0)):
        for index in range(3):
            cycle = np.arange(10, 910, 30, dtype=float)
            eta = level + 0.1 * index - 0.8 * (1 - np.exp(-rate * cycle / 1000.0))
            soh = 100 * hier.CEILING / (1 + np.exp(-eta)) * (1 + 0.01 * generator.standard_normal(len(cycle)))
            rows.append(
                pd.DataFrame(
                    {
                        "cell": f"{chemistry}{index}",
                        "CellType": chemistry,
                        "cycle": cycle,
                        "soh": soh,
                        "a": generator.standard_normal(len(cycle)),
                        "b": generator.standard_normal(len(cycle)),
                    }
                )
            )
    frame = pd.concat(rows, ignore_index=True)
    frame["y"] = hier.state_of_health(frame["soh"])
    return frame


def test_the_bands_are_ordered_and_have_the_length_of_the_rows():
    draws = np.random.default_rng(0).standard_normal((500, 7))
    lower, median, upper = evaluate.bands(draws)
    assert lower.shape == median.shape == upper.shape == (7,)
    assert (lower < median).all() and (median < upper).all()


def test_the_score_of_a_perfect_narrow_band_is_zero_error_and_full_coverage():
    frame = pd.DataFrame({"soh": [80.0, 70.0], "median": [80.0, 70.0], "lower": [79.0, 69.0], "upper": [81.0, 71.0]})
    result = evaluate.score(frame)
    assert (
        result["mae"] == 0.0
        and result["coverage"] == 1.0
        and result["width"] == 2.0
        and result["interval"] == 2.0
        and result["n"] == 2
    )


def test_the_score_counts_an_observation_outside_the_band_against_coverage_and_the_interval_score():
    frame = pd.DataFrame({"soh": [80.0, 90.0], "median": [80.0, 80.0], "lower": [79.0, 79.0], "upper": [81.0, 81.0]})
    result = evaluate.score(frame, alpha=0.1)
    assert result["coverage"] == 0.5 and result["mae"] == 5.0
    assert result["interval"] == pytest.approx((2.0 + 2.0 + 20.0 * 9.0) / 2)


def test_cold_start_scores_every_setting_on_the_same_window_and_reuses_stored_fits(cells, tmp_path):
    held = cells[cells["cell"].isin(["X2", "Y2"])]
    train = cells[~cells["cell"].isin(["X2", "Y2"])].reset_index(drop=True)
    scaler = StandardScaler().fit(train[FEATURES])
    tiny = {"draws": 20, "tune": 20, "chains": 2}
    scores, diagnostics, curves = evaluate.cold_start(
        train, held, scaler, FEATURES, (0, 150), 60, tiny, tiny, tmp_path, draws=30
    )
    assert set(scores["strategy"]) == set(evaluate.STRATEGY) and set(scores["cycles_seen"]) == {0, 150}
    assert scores.groupby("strategy")["n"].nunique().eq(1).all()  # the scored window does not depend on the cycles seen
    assert scores["n"].iloc[0] == int((held["cycle"] > 150 + 60).sum())
    assert len(diagnostics) == len(scores) and {"r_hat", "ess_bulk", "divergences"} <= set(diagnostics.columns)
    assert len(curves) == len(evaluate.STRATEGY) * len(held)
    assert len(list(tmp_path.glob("*.nc"))) == len(scores)
    again, _, _ = evaluate.cold_start(train, held, scaler, FEATURES, (0, 150), 60, tiny, tiny, tmp_path, draws=30)
    assert again["mae"].round(6).tolist() == scores["mae"].round(6).tolist()


def test_cold_start_requires_the_first_setting_to_be_zero(cells, tmp_path):
    scaler = StandardScaler().fit(cells[FEATURES])
    with pytest.raises(ValueError):
        evaluate.cold_start(cells, cells, scaler, FEATURES, (100, 200), 50, {}, {}, tmp_path)


def test_the_embargo_removes_the_cycles_next_to_the_last_one_seen(cells):
    held = cells[cells["cell"] == "X0"]
    window = held["cycle"] > 150 + 60
    assert held.loc[window, "cycle"].min() > 210 and held.loc[~window, "cycle"].max() <= 210


def test_load_keeps_every_nth_cycle_of_each_cell_and_scales_the_state_of_health(cells, tmp_path):
    source = tmp_path / "calce.parquet"
    raw = cells.assign(BatteryID="CALCE_" + cells["cell"]).drop(columns=["cell"])
    raw.to_parquet(source)
    loaded = calce.load(source, FEATURES, every=3)
    assert set(loaded["cell"]) == set(cells["cell"])
    assert (
        loaded.groupby("cell").size()
        == len(cells[cells["cell"] == "X0"]) // 3 + (len(cells[cells["cell"] == "X0"]) % 3 > 0)
    ).all()
    assert ((loaded["y"] > 0) & (loaded["y"] < 1)).all()
    with pytest.raises(ValueError):
        calce.load(source, FEATURES, every=0)


def test_split_separates_the_held_out_cells_and_rejects_an_unknown_one(cells):
    train, held = calce.split(cells, ["X2"])
    assert "X2" not in set(train["cell"]) and set(held["cell"]) == {"X2"}
    with pytest.raises(ValueError):
        calce.split(cells, ["Z9"])
