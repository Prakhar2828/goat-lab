from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goatlab.models.playoff_game_evidence import (
    PlayoffGamePolicy,
    add_game_metrics,
    add_season_relative_metrics,
    add_series_state_flags,
    bootstrap_series_comparison,
    core_stat_coverage,
    match_candidate_games,
    normalize_player_statistics,
    summarize_player_games,
)


def _raw_game(
    *,
    first_name: str = "Michael",
    last_name: str = "Jordan",
    player_id: int = 23,
    game_id: int = 1,
    game_date: str = "1991-04-20",
    team_id: int = 1,
    opponent_id: int = 2,
    win: int = 1,
    series_game: int = 1,
    points: float = 30.0,
) -> dict[str, object]:
    return {
        "firstName": first_name,
        "lastName": last_name,
        "personId": player_id,
        "gameId": game_id,
        "gameDate": game_date,
        "gameType": "Playoffs",
        "gameLabel": "East - First Round",
        "gameSubLabel": f"Game {series_game}",
        "seriesGameNumber": series_game,
        "win": win,
        "home": 1,
        "numMinutes": 40.0,
        "points": points,
        "assists": 5.0,
        "reboundsTotal": 6.0,
        "reboundsOffensive": 1.0,
        "reboundsDefensive": 5.0,
        "steals": 2.0,
        "blocks": 1.0,
        "turnovers": 3.0,
        "foulsPersonal": 3.0,
        "fieldGoalsMade": 12.0,
        "fieldGoalsAttempted": 24.0,
        "threePointersMade": 1.0,
        "threePointersAttempted": 3.0,
        "freeThrowsMade": 6.0,
        "freeThrowsAttempted": 8.0,
        "plusMinusPoints": 5.0,
        "playerteamId": team_id,
        "opponentteamId": opponent_id,
    }


def _normalized_games() -> pd.DataFrame:
    raw = pd.DataFrame(
        [
            _raw_game(game_id=1, game_date="1991-04-20", win=1, series_game=1),
            _raw_game(game_id=2, game_date="1991-04-22", win=0, series_game=2),
            _raw_game(game_id=3, game_date="1991-04-24", win=1, series_game=3),
            _raw_game(game_id=4, game_date="1991-04-26", win=1, series_game=4),
        ]
    )
    return add_game_metrics(normalize_player_statistics(raw))


def _series_row(series_games: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "SERIES_ID": "1990-91-1-2",
                "PLAYER_NAME": "Michael Jordan",
                "SEASON": "1990-91",
                "TEAM_ID": 1,
                "OPP_TEAM_ID": 2,
                "TEAM_WON_SERIES": 1,
                "TEAM_SERIES_WINS": 3,
                "OPP_SERIES_WINS": 1,
                "SERIES_GAMES": series_games,
                "SERIES_START_DATE": "1991-04-20",
                "SERIES_END_DATE": "1991-04-26",
                "ROUND": "First Round",
            }
        ]
    )


def test_normalization_builds_name_and_playoff_season() -> None:
    result = normalize_player_statistics(pd.DataFrame([_raw_game()]))
    assert result.loc[0, "PLAYER_NAME"] == "Michael Jordan"
    assert result.loc[0, "SEASON"] == "1990-91"
    assert result.loc[0, "GAME_ID"] == 1


def test_game_metrics_match_transparent_formula() -> None:
    result = add_game_metrics(
        normalize_player_statistics(pd.DataFrame([_raw_game()]))
    )
    assert result.loc[0, "TRUE_SHOOTING_PCT"] == pytest.approx(
        30.0 / (2.0 * (24.0 + 0.44 * 8.0))
    )
    assert result.loc[0, "GAME_SCORE"] == pytest.approx(21.4)
    assert result.loc[0, "GAME_SCORE_PER36"] == pytest.approx(19.26)


def test_season_relative_game_score_is_centered() -> None:
    raw = pd.DataFrame(
        [
            _raw_game(game_id=1, points=20.0),
            _raw_game(game_id=2, points=30.0),
            _raw_game(game_id=3, points=40.0),
        ]
    )
    games = add_game_metrics(normalize_player_statistics(raw))
    result = add_season_relative_metrics(games)
    assert result["GAME_SCORE_SEASON_Z"].mean() == pytest.approx(0.0)
    assert result["GAME_SCORE_PERCENTILE"].between(0, 1).all()


def test_exact_series_match_and_state_flags() -> None:
    result = match_candidate_games(_normalized_games(), _series_row())
    assert len(result) == 4
    assert result["SERIES_ID"].nunique() == 1
    assert result["ELIMINATION_GAME"].sum() == 0
    assert result["CLOSEOUT_OPPORTUNITY"].sum() == 1
    assert result["SERIES_CLINCH_GAME"].sum() == 1


def test_series_match_rejects_wrong_game_count() -> None:
    with pytest.raises(ValueError, match="game counts did not match"):
        match_candidate_games(_normalized_games(), _series_row(series_games=5))


def test_elimination_flag_uses_pre_game_state() -> None:
    frame = _normalized_games().copy()
    frame["PLAYER_NAME"] = "Michael Jordan"
    frame["SERIES_ID"] = "x"
    frame["TEAM_SERIES_WINS"] = 3
    frame["OPP_SERIES_WINS"] = 2
    frame["WIN"] = [0, 1, 0, 1]
    result = add_series_state_flags(frame)
    assert result["ELIMINATION_GAME"].tolist() == [False, False, False, True]


def test_missing_core_stat_is_not_converted_to_zero() -> None:
    games = _normalized_games()
    games.loc[0, "ASSISTS"] = np.nan
    coverage = core_stat_coverage(games)
    assists = coverage[coverage["STAT"].eq("ASSISTS")].iloc[0]
    assert assists["OBSERVATIONS"] == 3
    assert assists["COVERAGE_RATE"] == pytest.approx(0.75)
    assert pd.isna(games.loc[0, "ASSISTS"])


def test_player_summary_is_diagnostic_only() -> None:
    games = add_season_relative_metrics(_normalized_games())
    games = match_candidate_games(games, _series_row())
    summary = summarize_player_games(games)
    assert summary["ADDITIONAL_CENTRAL_WEIGHT"].eq(0.0).all()
    assert not summary["PRIMARY_MODEL_ELIGIBLE"].any()


def test_series_bootstrap_is_reproducible() -> None:
    series = pd.DataFrame(
        {
            "PLAYER_NAME": [
                "Michael Jordan",
                "Michael Jordan",
                "LeBron James",
                "LeBron James",
            ],
            "GAME_SCORE": [20.0, 22.0, 21.0, 25.0],
        }
    )
    first = bootstrap_series_comparison(
        series,
        metrics=["GAME_SCORE"],
        repetitions=200,
        seed=23,
    )
    second = bootstrap_series_comparison(
        series,
        metrics=["GAME_SCORE"],
        repetitions=200,
        seed=23,
    )
    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "CLUSTER_UNIT"] == "playoff_series"
    assert first.loc[0, "ADDITIONAL_CENTRAL_WEIGHT"] == 0.0


def test_policy_keeps_final_simulation_blocked() -> None:
    policy = PlayoffGamePolicy()
    assert policy.additional_central_weight == 0.0
    assert policy.final_simulation_allowed is False
