from goatlab.utils import season_range


def test_season_range_is_inclusive() -> None:
    assert season_range("2022-23", "2024-25") == ["2022-23", "2023-24", "2024-25"]
