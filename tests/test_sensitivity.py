import pandas as pd

from goatlab.models.sensitivity import CATEGORIES, run_weight_simulation


def test_weight_simulation_outputs_probabilities() -> None:
    rows = []
    for player, base in [("A", 60), ("B", 40)]:
        row = {"PLAYER_NAME": player}
        row.update({category: base for category in CATEGORIES})
        rows.append(row)
    summary, drivers = run_weight_simulation(pd.DataFrame(rows), simulations=1000)
    assert abs(summary["WIN_RATE"].sum() - 1) < 1e-12
    assert summary.sort_values("WIN_RATE", ascending=False).iloc[0]["PLAYER_NAME"] == "A"
    assert not drivers.empty
