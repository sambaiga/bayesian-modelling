import numpy as np
import pandas as pd
import pytest

from bayes import eol_study, survival

TINY = {"draws": 60, "tune": 80, "chains": 2}


def cells() -> pd.DataFrame:
    rows = []
    for cell, chemistry, a, b in (
        ("A_1", "A", 3.9, 2.4),
        ("A_2", "A", 3.8, 2.2),
        ("A_3", "A", 3.85, 2.3),
        ("B_1", "B", 3.2, 1.1),
        ("B_2", "B", 3.1, 1.0),
        ("B_3", "B", 3.15, 1.05),
    ):
        cycles = np.arange(8, 1600, 8, dtype=float)
        loss = np.exp(a) * (cycles / 1000.0) ** b + np.random.default_rng(len(cell) + int(a * 10)).normal(
            0, 0.3, len(cycles)
        )
        rows.append(pd.DataFrame({"cell": cell, "CellType": chemistry, "cycle": cycles, "soh": 100.0 - loss}))
    return pd.concat(rows, ignore_index=True)


def test_weibull_predictions_cover_the_running_cells_only_and_stay_above_the_stop(tmp_path):
    table = survival.end_of_life(cells())
    stop = float(table["eol"].median())
    draws = eol_study.weibull_predictions(table, stop, tmp_path, TINY, draws=500)
    running = set(table.loc[table["eol"] > stop, "cell"])
    assert set(draws.columns) == running and 0 < len(running) < len(table)
    assert draws.to_numpy().min() > stop and draws.shape[0] == 500


def test_weibull_predictions_are_empty_when_no_cell_is_running(tmp_path):
    table = survival.end_of_life(cells())
    assert eol_study.weibull_predictions(table, float(table["eol"].max()) + 1, tmp_path, TINY, draws=50).shape[1] == 0


def test_path_predictions_follow_the_cells_that_run_on_and_report_what_was_dropped(tmp_path):
    frame = cells()
    table = survival.end_of_life(frame)
    stop = float(table["eol"].median())
    draws, dropped = eol_study.path_predictions(frame, table, stop, tmp_path, TINY)
    running = set(table.loc[table["eol"] > stop, "cell"])
    assert set(draws.columns) == running and set(dropped["cell"]) == running
    assert dropped["dropped"].between(0.0, 1.0).all()
    assert np.nanmin(draws.to_numpy()) > stop


def test_score_predictions_gives_one_row_for_each_cell_with_the_observed_end_of_life():
    table = survival.end_of_life(cells())
    cell = table["cell"].iloc[0]
    truth = float(table.set_index("cell").loc[cell, "eol"])
    draws = pd.DataFrame({cell: np.linspace(truth - 50, truth + 80, 201), "pad": np.nan})
    scored = eol_study.score_predictions(draws[[cell]], table)
    assert list(scored["cell"]) == [cell] and scored["eol"].iloc[0] == truth
    assert scored["covered"].iloc[0] == 1.0 and scored["error"].iloc[0] == pytest.approx(15.0, abs=1.0)
    padded = pd.DataFrame({cell: np.concatenate([np.linspace(truth - 50, truth + 80, 201), [np.nan] * 20])})
    assert eol_study.score_predictions(padded, table)["median"].iloc[0] == pytest.approx(scored["median"].iloc[0])
