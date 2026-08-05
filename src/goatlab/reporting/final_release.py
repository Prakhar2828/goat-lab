from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

TARGET_PLAYERS = ("LeBron James", "Michael Jordan")

FINAL_ARTIFACT_PATHS = (
    "data/processed/production_category_scores.parquet",
    "data/processed/production_hierarchy_scores.parquet",
    "data/processed/weight_simulation_summary.parquet",
    "data/processed/weight_simulation_drivers.parquet",
    "data/processed/hierarchy_weight_simulation_group_audit.parquet",
    "data/processed/v1_release_gate.json",
    "models/training_metadata.json",
)

FREEZE_INPUT_PATHS = (
    "configs/v1_final_model.json",
    "configs/category_hierarchy.json",
    "docs/V1_FINAL_PREREGISTRATION.md",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hierarchy_weights(config: dict[str, Any]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for group in config["groups"]:
        cap = float(group["cap"])
        for item in group["categories"]:
            category = str(item["name"])
            weights[category] = cap * float(item["within_group_weight"])

    if not weights:
        raise ValueError("The hierarchy contains no category weights.")
    if abs(sum(weights.values()) - 1.0) > 1e-10:
        raise ValueError("Frozen hierarchy weights must sum to one.")
    return weights


def weighted_scores(
    category_scores: pd.DataFrame,
    weights: dict[str, float],
) -> dict[str, float]:
    required = {"PLAYER_NAME", *weights}
    missing = required.difference(category_scores.columns)
    if missing:
        raise ValueError(
            f"Category score frame is missing columns: {sorted(missing)}"
        )

    target = category_scores[
        category_scores["PLAYER_NAME"].isin(TARGET_PLAYERS)
    ].copy()
    if set(target["PLAYER_NAME"].astype(str)) != set(TARGET_PLAYERS):
        raise ValueError("Both target players are required.")

    result: dict[str, float] = {}
    for row in target.itertuples(index=False):
        score = sum(
            float(getattr(row, category)) * weight
            for category, weight in weights.items()
        )
        result[str(row.PLAYER_NAME)] = float(score)
    return result


def build_scale_sensitivity(
    comparison: pd.DataFrame,
    current_category_scores: pd.DataFrame,
    final_config: dict[str, Any],
    hierarchy_config: dict[str, Any],
) -> list[dict[str, Any]]:
    scale_categories = [
        str(value)
        for value in final_config["production_scale_categories"]
    ]
    native_categories = [
        str(value)
        for value in final_config["native_scale_categories"]
    ]
    weights = hierarchy_weights(hierarchy_config)

    expected = set(weights)
    observed = set(scale_categories) | set(native_categories)
    if observed != expected:
        raise ValueError(
            "Scale and native category sets do not match the hierarchy."
        )

    native = current_category_scores[
        ["PLAYER_NAME", *native_categories]
    ].copy()

    rows: list[dict[str, Any]] = []
    scenarios = sorted(comparison["SCENARIO"].astype(str).unique())

    for scenario in scenarios:
        scenario_rows = comparison[
            comparison["SCENARIO"].astype(str).eq(scenario)
            & comparison["CATEGORY"].isin(scale_categories)
        ][
            ["PLAYER_NAME", "CATEGORY", "SCORE"]
        ].copy()

        pivot = scenario_rows.pivot(
            index="PLAYER_NAME",
            columns="CATEGORY",
            values="SCORE",
        ).reset_index()

        combined = pivot.merge(
            native,
            on="PLAYER_NAME",
            how="inner",
            validate="one_to_one",
        )
        scores = weighted_scores(combined, weights)

        lebron = scores["LeBron James"]
        jordan = scores["Michael Jordan"]
        margin = lebron - jordan

        if abs(margin) <= 1e-12:
            winner = "Tie"
        elif margin > 0:
            winner = "LeBron James"
        else:
            winner = "Michael Jordan"

        rows.append(
            {
                "scenario": scenario,
                "winner": winner,
                "lebron_score": lebron,
                "jordan_score": jordan,
                "lebron_minus_jordan": margin,
                "is_production": scenario
                == str(final_config["production_scale"]),
            }
        )

    if not rows:
        raise ValueError("No scaling scenarios were available.")
    return rows


def _player_records(
    frame: pd.DataFrame,
    value_columns: list[str],
    sort_column: str,
) -> list[dict[str, Any]]:
    target = frame[
        frame["PLAYER_NAME"].isin(TARGET_PLAYERS)
    ].copy()
    if len(target) != 2:
        raise ValueError("Expected exactly two target-player rows.")

    target = target.sort_values(sort_column, ascending=False)
    records: list[dict[str, Any]] = []
    for _, row in target.iterrows():
        record: dict[str, Any] = {
            "player": str(row["PLAYER_NAME"]),
        }
        for column in value_columns:
            value = row[column]
            if pd.isna(value):
                record[column.lower()] = None
            elif isinstance(value, (int, float)):
                record[column.lower()] = float(value)
            else:
                record[column.lower()] = value.item() if hasattr(value, "item") else value
        records.append(record)
    return records



def _required_config_leaf(
    config: dict[str, object],
    key: str,
) -> object:
    """Read one frozen value regardless of config nesting.

    Duplicate occurrences are allowed only when they contain
    exactly the same preregistered value.
    """
    matches: list[object] = []

    def walk(value: object) -> None:
        if isinstance(value, dict):
            if key in value:
                matches.append(value[key])

            for child in value.values():
                walk(child)

        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(config)

    if not matches:
        raise KeyError(
            f"Frozen final-model config is missing {key!r}."
        )

    first = matches[0]

    if any(value != first for value in matches[1:]):
        raise ValueError(
            f"Frozen final-model config contains conflicting "
            f"values for {key!r}: {matches!r}"
        )

    return first



def build_release_manifest(
    repository_root: str | Path = ".",
    *,
    git_commit: str,
    git_branch: str,
) -> dict[str, Any]:
    root = Path(repository_root)

    final_config = json.loads(
        (root / "configs/v1_final_model.json").read_text(encoding="utf-8")
    )
    hierarchy_config = json.loads(
        (root / "configs/category_hierarchy.json").read_text(
            encoding="utf-8"
        )
    )
    gate = json.loads(
        (root / "data/processed/v1_release_gate.json").read_text(
            encoding="utf-8"
        )
    )
    training = json.loads(
        (root / "models/training_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    cultural_audit = json.loads(
        (
            root
            / "data/processed/cultural_impact_sensitivity_audit.json"
        ).read_text(encoding="utf-8")
    )

    summary = pd.read_parquet(
        root / "data/processed/weight_simulation_summary.parquet"
    )
    drivers = pd.read_parquet(
        root / "data/processed/weight_simulation_drivers.parquet"
    )
    group_audit = pd.read_parquet(
        root
        / "data/processed/hierarchy_weight_simulation_group_audit.parquet"
    )
    production_categories = pd.read_parquet(
        root / "data/processed/production_category_scores.parquet"
    )
    production_hierarchy = pd.read_parquet(
        root / "data/processed/production_hierarchy_scores.parquet"
    )
    comparison = pd.read_parquet(
        root / "data/processed/category_scaling_comparison.parquet"
    )
    current_categories = pd.read_parquet(
        root / "data/processed/category_scores.parquet"
    )

    if not bool(gate["final_simulation_allowed"]):
        raise ValueError("The final release gate is not open.")
    if int(gate["release_blockers"]) != 0:
        raise ValueError("The final release gate has blockers.")
    if int(training["simulations"]) != int(_required_config_leaf(final_config, "simulations")):
        raise ValueError("Simulation count does not match the freeze.")
    if int(training["random_seed"]) != int(_required_config_leaf(final_config, "random_seed")):
        raise ValueError("Random seed does not match the freeze.")

    if abs(float(summary["WIN_RATE"].sum()) - 1.0) > 1e-12:
        raise ValueError("Simulation win rates must sum to one.")

    for row in group_audit.itertuples(index=False):
        cap = float(row.FROZEN_GROUP_CAP)
        if abs(float(row.MIN_REALIZED_GROUP_MASS) - cap) > 1e-12:
            raise ValueError("A simulated group minimum violated its cap.")
        if abs(float(row.MAX_REALIZED_GROUP_MASS) - cap) > 1e-12:
            raise ValueError("A simulated group maximum violated its cap.")

    central_records = _player_records(
        production_hierarchy,
        ["GOAT_SCORE", "RANK"],
        "GOAT_SCORE",
    )
    simulation_records = _player_records(
        summary,
        [
            "WIN_RATE",
            "MEAN_SCORE",
            "P05_SCORE",
            "P95_SCORE",
        ],
        "WIN_RATE",
    )

    central_margin = (
        float(central_records[0]["goat_score"])
        - float(central_records[1]["goat_score"])
    )
    win_rate_margin = (
        float(simulation_records[0]["win_rate"])
        - float(simulation_records[1]["win_rate"])
    )

    scale_sensitivity = build_scale_sensitivity(
        comparison,
        current_categories,
        final_config,
        hierarchy_config,
    )
    scale_winners = {
        row["winner"]
        for row in scale_sensitivity
        if row["winner"] != "Tie"
    }
    scale_robust = len(scale_winners) <= 1

    hashes: dict[str, str] = {}
    for relative in (*FINAL_ARTIFACT_PATHS, *FREEZE_INPUT_PATHS):
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"Missing release artifact: {relative}")
        hashes[relative] = sha256_file(path)

    category_records = production_categories.sort_values(
        "PLAYER_NAME"
    ).to_dict(orient="records")
    for record in category_records:
        for key, value in list(record.items()):
            if key == "PLAYER_NAME":
                continue
            record[key] = float(value)

    driver_records = drivers.to_dict(orient="records")
    for record in driver_records:
        for key, value in list(record.items()):
            if pd.isna(value):
                record[key] = None
            elif hasattr(value, "item"):
                record[key] = value.item()

    group_records = group_audit.to_dict(orient="records")
    for record in group_records:
        for key, value in list(record.items()):
            if hasattr(value, "item"):
                record[key] = value.item()

    manifest: dict[str, Any] = {
        "release": "v1",
        "source_commit": git_commit,
        "source_branch": git_branch,
        "result_classification": (
            "conditional_not_robust_across_approved_scaling_scenarios"
            if not scale_robust
            else "robust_across_approved_scaling_scenarios"
        ),
        "interpretation": (
            "Share of preregistered sampled value systems won; "
            "not an objective probability that a player is the GOAT."
        ),
        "model_freeze": {
            "production_scale": final_config["production_scale"],
            "production_scale_categories": final_config[
                "production_scale_categories"
            ],
            "native_scale_categories": final_config[
                "native_scale_categories"
            ],
            "simulations": int(_required_config_leaf(final_config, "simulations")),
            "random_seed": int(_required_config_leaf(final_config, "random_seed")),
            "within_group_concentration": float(
                _required_config_leaf(final_config, "within_group_concentration")
            ),
            "production_weights_frozen": bool(
                gate["production_weights_frozen"]
            ),
            "production_scale_frozen": bool(
                gate["production_scale_frozen"]
            ),
            "release_gate_checks": int(gate["checks"]),
            "release_gate_passed_checks": int(gate["passed_checks"]),
            "release_blockers": int(gate["release_blockers"]),
        },
        "central_result": {
            "winner": central_records[0]["player"],
            "margin_points": central_margin,
            "players": central_records,
        },
        "simulation_result": {
            "winner": simulation_records[0]["player"],
            "win_rate_margin": win_rate_margin,
            "players": simulation_records,
        },
        "production_category_scores": category_records,
        "scale_sensitivity": {
            "winner_robust": scale_robust,
            "distinct_winners": sorted(scale_winners),
            "scenarios": scale_sensitivity,
        },
        "cultural_weighting_sensitivity": {
            "winner_robust": bool(
                cultural_audit["winner_robust_across_grid"]
            ),
            "scenario_count": int(cultural_audit["scenarios"]),
            "winner_counts": cultural_audit["winner_counts"],
            "configured_crossover_attention_weight": cultural_audit[
                "configured_blend_crossover_attention_weight"
            ],
        },
        "simulation_drivers": driver_records,
        "group_cap_audit": group_records,
        "artifact_sha256": hashes,
        "limitations": [
            "The central winner changes across approved category-scaling scenarios.",
            "The cultural-impact ordering changes across reasonable weighting choices.",
            "Expert-film evidence was excluded from the primary model because no source met the frozen eligibility standard.",
            "Game-level playoff, impact-metric, and supporting-cast audits remain diagnostic and add zero central weight.",
            "The simulation varies within-group category weights around fixed group caps; it does not sample every possible GOAT philosophy.",
        ],
    }
    return manifest


def render_results_markdown(manifest: dict[str, Any]) -> str:
    central = manifest["central_result"]
    simulation = manifest["simulation_result"]
    scale = manifest["scale_sensitivity"]
    freeze = manifest["model_freeze"]

    central_by_player = {
        row["player"]: row
        for row in central["players"]
    }
    simulation_by_player = {
        row["player"]: row
        for row in simulation["players"]
    }

    lines = [
        "# GOAT Lab v1 Final Results",
        "",
        "## Headline",
        "",
        (
            f"Under the preregistered v1 production model, "
            f"**{central['winner']} ranks first by "
            f"{central['margin_points']:.6f} points**."
        ),
        "",
        (
            f"Across {freeze['simulations']:,} hierarchy-aware weight "
            f"simulations, **{simulation['winner']} wins "
            f"{simulation_by_player[simulation['winner']]['win_rate']:.4%}** "
            "of sampled value systems."
        ),
        "",
        ("> This is a conditional model result, not an objective probability "
        "that either player is the GOAT."),
        "",
        "## Frozen central result",
        "",
        "| Player | GOAT score | Rank |",
        "|---|---:|---:|",
    ]

    for player in ("LeBron James", "Michael Jordan"):
        row = central_by_player[player]
        lines.append(
            f"| {player} | {row['goat_score']:.6f} | "
            f"{int(row['rank'])} |"
        )

    lines.extend(
        [
            "",
            "## Preregistered simulation",
            "",
            "| Player | Win rate | Mean score | P05 | P95 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for player in ("LeBron James", "Michael Jordan"):
        row = simulation_by_player[player]
        lines.append(
            f"| {player} | {row['win_rate']:.4%} | "
            f"{row['mean_score']:.6f} | {row['p05_score']:.6f} | "
            f"{row['p95_score']:.6f} |"
        )

    lines.extend(
        [
            "",
            ("The win rate is the share of frozen-cap, within-group weight "
            "systems won. Group mass remains exactly 50% Performance Arc, "
            "40% Basketball Value, and 10% Broader Legacy."),
            "",
            "## Scale sensitivity",
            "",
            "| Scaling scenario | LeBron | Jordan | L-J margin | Winner |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in scale["scenarios"]:
        marker = " *(production)*" if row["is_production"] else ""
        lines.append(
            f"| {row['scenario']}{marker} | "
            f"{row['lebron_score']:.6f} | "
            f"{row['jordan_score']:.6f} | "
            f"{row['lebron_minus_jordan']:+.6f} | "
            f"{row['winner']} |"
        )

    lines.extend(
        [
            "",
            (
                "**Robustness conclusion:** the winner is not stable across "
                "the four approved scaling scenarios. The production result "
                "uses `bounded_logit_tail` because that method was frozen "
                "before the final simulation."
            ),
            "",
            "## Largest simulation drivers",
            "",
            "| Category | Frozen weight | Margin correlation | Interpretation |",
            "|---|---:|---:|---|",
        ]
    )

    for row in manifest["simulation_drivers"][:5]:
        correlation = row["MARGIN_CORRELATION_PLAYER_1"]
        if correlation is None:
            interpretation = "Fixed in this simulation."
            correlation_text = "—"
        elif correlation > 0:
            interpretation = "More weight generally favors LeBron."
            correlation_text = f"{correlation:+.6f}"
        else:
            interpretation = "More weight generally favors Jordan."
            correlation_text = f"{correlation:+.6f}"
        lines.append(
            f"| {row['CATEGORY']} | "
            f"{row['FROZEN_TOTAL_WEIGHT']:.3f} | "
            f"{correlation_text} | {interpretation} |"
        )

    lines.extend(
        [
            "",
            ("Defense is the strongest swing factor, while offense is the "
            "largest counterweight favoring LeBron."),
            "",
            "## Evidence treatment and limitations",
            "",
        ]
    )
    for limitation in manifest["limitations"]:
        lines.append(f"- {limitation}")

    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Source commit: `{manifest['source_commit']}`",
            f"- Source branch: `{manifest['source_branch']}`",
            f"- Production scale: `{freeze['production_scale']}`",
            f"- Simulations: `{freeze['simulations']}`",
            f"- Random seed: `{freeze['random_seed']}`",
            (
                "- Within-group concentration: "
                f"`{freeze['within_group_concentration']}`"
            ),
            (
                "- Release gate: "
                f"`{freeze['release_gate_passed_checks']}/"
                f"{freeze['release_gate_checks']}` checks passed"
            ),
            "- Artifact hashes: `release/v1_artifact_hashes.sha256`",
            "- Machine-readable manifest: `release/v1_release_manifest.json`",
            "",
        ]
    )
    return "\n".join(lines)


def write_release_bundle(
    manifest: dict[str, Any],
    repository_root: str | Path = ".",
) -> None:
    root = Path(repository_root)
    release_dir = root / "release"
    release_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = release_dir / "v1_release_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    hash_lines = [
        f"{digest}  {relative}"
        for relative, digest in sorted(
            manifest["artifact_sha256"].items()
        )
    ]
    (release_dir / "v1_artifact_hashes.sha256").write_text(
        "\n".join(hash_lines) + "\n",
        encoding="utf-8",
    )

    docs_path = root / "docs/V1_FINAL_RESULTS.md"
    docs_path.write_text(
        render_results_markdown(manifest),
        encoding="utf-8",
    )


def verify_release_bundle(
    repository_root: str | Path = ".",
) -> None:
    root = Path(repository_root)
    manifest_path = root / "release/v1_release_manifest.json"
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    for relative, expected in manifest["artifact_sha256"].items():
        actual = sha256_file(root / relative)
        if actual != expected:
            raise ValueError(
                f"Hash mismatch for {relative}: {actual} != {expected}"
            )

    markdown = render_results_markdown(manifest)
    actual_markdown = (
        root / "docs/V1_FINAL_RESULTS.md"
    ).read_text(encoding="utf-8")
    if actual_markdown != markdown:
        raise ValueError("The final-results document is not reproducible.")
