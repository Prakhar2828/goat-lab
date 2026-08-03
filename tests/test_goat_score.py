import pandas as pd

from goatlab.models.goat_score import score_players, validate_weights


def test_validate_weights_normalizes() -> None:
    weights = validate_weights({"peak": 2, "longevity": 1})
    assert abs(sum(weights.values()) - 1) < 1e-12
    assert weights["peak"] == 2 / 3


def test_score_players_reweights_missing_categories() -> None:
    frame = pd.DataFrame(
        [
            {"PLAYER_NAME": "A", "peak": 100.0, "longevity": 50.0},
            {"PLAYER_NAME": "B", "peak": 80.0, "longevity": 100.0},
        ]
    )
    scored = score_players(frame, {"peak": 0.5, "longevity": 0.5})
    assert scored.iloc[0]["PLAYER_NAME"] == "B"
