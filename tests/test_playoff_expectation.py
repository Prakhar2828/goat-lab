from pathlib import Path

import pandas as pd

from goatlab.models.playoff_expectation import (
    cross_fit_series_overperformance,
)


def test_season_grouped_cross_fit(
    tmp_path: Path,
) -> None:
    rows = []

    for season_number in range(8):
        season = (
            f"{2000 + season_number}-"
            f"{str(2001 + season_number)[-2:]}"
        )

        for series_number in range(3):
            strength = (
                season_number
                + series_number
            )

            rows.extend(
                [
                    {
                        "SEASON": season,
                        "SERIES_ID": (
                            f"{season}-{series_number}"
                        ),
                        "TEAM_WON_SERIES": 1,
                        "TEAM_SRS": strength + 2,
                        "OPP_SRS": strength,
                        "TEAM_NET_RATING": (
                            strength + 1
                        ),
                        "OPP_NET_RATING": strength,
                        "HOME_COURT": 1,
                    },
                    {
                        "SEASON": season,
                        "SERIES_ID": (
                            f"{season}-{series_number}"
                        ),
                        "TEAM_WON_SERIES": 0,
                        "TEAM_SRS": strength,
                        "OPP_SRS": strength + 2,
                        "TEAM_NET_RATING": strength,
                        "OPP_NET_RATING": (
                            strength + 1
                        ),
                        "HOME_COURT": 0,
                    },
                ]
            )

    frame = pd.DataFrame(rows)

    scored, report = (
        cross_fit_series_overperformance(
            frame,
            n_splits=4,
            artifact_path=(
                tmp_path / "model.joblib"
            ),
        )
    )

    assert scored[
        "EXPECTED_SERIES_WIN_PROB"
    ].notna().all()

    assert scored[
        "EXPECTED_SERIES_WIN_PROB"
    ].between(0, 1).all()

    assert (
        scored.groupby("SEASON")[
            "CV_FOLD"
        ].nunique()
        == 1
    ).all()

    assert (
        scored["PREDICTION_SOURCE"]
        == "season_grouped_out_of_fold"
    ).all()

    assert report.folds == 4
    assert (
        report.evaluation_method
        == "season_grouped_out_of_fold"
    )
