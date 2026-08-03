from __future__ import annotations

import numpy as np
import pandas as pd


CATEGORIES = [
    "peak",
    "prime",
    "longevity",
    "regular_season",
    "playoffs",
    "winning_context",
    "offense",
    "defense",
    "cultural_impact",
]


def run_weight_simulation(
    category_scores: pd.DataFrame,
    simulations: int = 250_000,
    alpha: float | list[float] = 1.0,
    random_seed: int = 23,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample defensible weight systems from a Dirichlet distribution."""
    players = category_scores["PLAYER_NAME"].tolist()
    matrix = category_scores.set_index("PLAYER_NAME")[CATEGORIES].astype(float)
    if matrix.isna().any().any():
        matrix = matrix.apply(lambda row: row.fillna(row.mean()), axis=1)

    alpha_vector = np.repeat(float(alpha), len(CATEGORIES)) if np.isscalar(alpha) else np.asarray(alpha)
    rng = np.random.default_rng(random_seed)
    weights = rng.dirichlet(alpha_vector, size=simulations)
    score_matrix = weights @ matrix.T.to_numpy()
    winner_indices = np.argmax(score_matrix, axis=1)

    summary = pd.DataFrame(
        {
            "PLAYER_NAME": players,
            "WIN_RATE": [float(np.mean(winner_indices == index)) for index in range(len(players))],
            "MEAN_SCORE": score_matrix.mean(axis=0),
            "P05_SCORE": np.quantile(score_matrix, 0.05, axis=0),
            "P95_SCORE": np.quantile(score_matrix, 0.95, axis=0),
        }
    )

    if len(players) == 2:
        margin = score_matrix[:, 0] - score_matrix[:, 1]
        correlations = [float(np.corrcoef(weights[:, index], margin)[0, 1]) for index in range(len(CATEGORIES))]
        drivers = pd.DataFrame(
            {
                "CATEGORY": CATEGORIES,
                "MARGIN_CORRELATION_PLAYER_1": correlations,
                "MEAN_WEIGHT": weights.mean(axis=0),
            }
        ).sort_values("MARGIN_CORRELATION_PLAYER_1", key=np.abs, ascending=False)
    else:
        drivers = pd.DataFrame({"CATEGORY": CATEGORIES, "MEAN_WEIGHT": weights.mean(axis=0)})
    return summary, drivers


def two_category_boundary(
    category_scores: pd.DataFrame,
    category_x: str,
    category_y: str,
    grid_size: int = 101,
) -> pd.DataFrame:
    if len(category_scores) != 2:
        raise ValueError("Boundary visualization currently requires exactly two players.")
    players = category_scores["PLAYER_NAME"].tolist()
    rows: list[dict[str, float | str]] = []
    for x_weight in np.linspace(0, 1, grid_size):
        for y_weight in np.linspace(0, 1 - x_weight, grid_size):
            remaining = 1 - x_weight - y_weight
            other_categories = [c for c in CATEGORIES if c not in {category_x, category_y}]
            weight_map = {c: remaining / len(other_categories) for c in other_categories}
            weight_map[category_x] = x_weight
            weight_map[category_y] = y_weight
            scores = []
            for _, player in category_scores.iterrows():
                scores.append(sum(float(player[c]) * weight_map[c] for c in CATEGORIES))
            rows.append(
                {
                    "X_WEIGHT": x_weight,
                    "Y_WEIGHT": y_weight,
                    "WINNER": players[int(np.argmax(scores))],
                    "MARGIN": abs(scores[0] - scores[1]),
                }
            )
    return pd.DataFrame(rows)
