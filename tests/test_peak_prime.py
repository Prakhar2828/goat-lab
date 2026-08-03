import pandas as pd

from goatlab.models.peak_prime import summarize_peak_prime_longevity


def test_peak_summary() -> None:
    frame = pd.DataFrame(
        {
            "PLAYER_NAME": ["A"] * 5,
            "SEASON_TYPE": ["Regular Season"] * 5,
            "CAREER_YEAR": [1, 2, 3, 4, 5],
            "SEASON_VALUE_0_100": [70, 80, 90, 85, 75],
            "MIN": [1000] * 5,
        }
    )
    summary = summarize_peak_prime_longevity(frame)
    assert summary.iloc[0]["BEST_SEASON"] == 90
    assert summary.iloc[0]["TOP_3_PEAK"] == 85
