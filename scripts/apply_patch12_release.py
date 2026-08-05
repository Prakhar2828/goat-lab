from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected exactly one occurrence of {old!r}; found {count}"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("UPDATED:", path)


def main() -> int:
    pyproject = Path("pyproject.toml")
    text = pyproject.read_text(encoding="utf-8")
    if 'version = "1.0.0"' not in text:
        replace_once(
            pyproject,
            'version = "0.1.0"',
            'version = "1.0.0"',
        )
    else:
        print("UNCHANGED:", pyproject)

    dictionary = Path("docs/DATA_DICTIONARY.md")
    section = """
## Public v1 dashboard package

The immutable Streamlit release reads publication-safe files from
`release/dashboard_data/` before falling back to local `data/processed/` outputs.

- `production_category_scores.parquet` — frozen nine-category production scores
- `production_hierarchy_scores.parquet` — frozen total GOAT scores and ranks
- `weight_simulation_summary.parquet` — 250,000-run win rates and score intervals
- `weight_simulation_drivers.parquet` — category-weight correlations with the margin
- `hierarchy_weight_simulation_group_audit.parquet` — realized group-cap validation
- `v1_release_gate.json` — 32-check frozen release-gate metadata
- `goat_player_season_values.parquet` — publication career-curve data
- `peak_prime_longevity.parquet` — publication peak/prime/longevity summaries
- `playoff_series_scored.parquet` — publication playoff-series context
"""

    content = dictionary.read_text(encoding="utf-8")
    if "## Public v1 dashboard package" not in content:
        dictionary.write_text(
            content.rstrip() + "\n\n" + section.strip() + "\n",
            encoding="utf-8",
        )
        print("UPDATED:", dictionary)
    else:
        print("UNCHANGED:", dictionary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
