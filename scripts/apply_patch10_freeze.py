from __future__ import annotations

import json
from pathlib import Path


def replace_once(
    path: Path,
    old: str,
    new: str,
) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        print("Already patched:", path)
        return
    if old not in text:
        raise RuntimeError(
            f"Expected source block not found in {path}"
        )
    path.write_text(
        text.replace(old, new, 1),
        encoding="utf-8",
    )
    print("Patched:", path)


def main() -> None:
    hierarchy_module = Path(
        "src/goatlab/models/category_hierarchy.py"
    )
    replace_once(
        hierarchy_module,
        '''    if bool(config.get("final_simulation_allowed", False)):
        raise ValueError("This patch may not unlock the final simulation.")
''',
        '''    final_allowed = bool(
        config.get("final_simulation_allowed", False)
    )
    required_freezes = (
        bool(config.get("hierarchy_frozen", False)),
        bool(config.get("group_caps_frozen", False)),
        bool(config.get("production_weights_frozen", False)),
    )
    if final_allowed and not all(required_freezes):
        raise ValueError(
            "Final simulation requires the hierarchy, group caps, "
            "and production weights to be frozen."
        )
''',
    )

    pipeline = Path("src/goatlab/pipeline.py")
    replace_once(
        pipeline,
        '''from goatlab.models.sensitivity import run_weight_simulation
''',
        '''from goatlab.models.category_hierarchy import (
    load_hierarchy_config,
)
from goatlab.models.final_model import (
    load_final_model_config,
    run_hierarchy_weight_simulation,
    score_frozen_hierarchy,
)
from goatlab.models.release_gate import (
    assert_v1_release_gate,
)
''',
    )

    replace_once(
        pipeline,
        '''def train_models() -> None:
    settings.ensure_directories()
    category_scores = build_category_scores()
    if category_scores.drop(columns=["PLAYER_NAME"]).isna().any().any():
        missing = category_scores.set_index("PLAYER_NAME").columns[
            category_scores.set_index("PLAYER_NAME").isna().any()
        ].tolist()
        raise ValueError(
            "Complete the contextual category inputs before final simulation. "
            f"Missing categories: {missing}"
        )
    summary, drivers = run_weight_simulation(category_scores)
    write_parquet(summary, settings.processed_dir / "weight_simulation_summary.parquet")
    write_parquet(drivers, settings.processed_dir / "weight_simulation_drivers.parquet")
    metadata = {
        "simulations": 250000,
        "random_seed": settings.random_seed,
        "note": "Playoff expectation training requires data/manual/playoff_series.csv.",
    }
    (settings.model_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
''',
        '''def train_models() -> None:
    settings.ensure_directories()
    category_scores = build_category_scores()

    gate_metadata, production_scores = (
        assert_v1_release_gate(
            processed_dir=settings.processed_dir,
            category_scores=category_scores,
        )
    )

    final_config = load_final_model_config(
        "configs/v1_final_model.json"
    )
    hierarchy = load_hierarchy_config(
        final_config["hierarchy_config"]
    )
    central = score_frozen_hierarchy(
        production_scores,
        hierarchy,
    )

    simulation = final_config["simulation"]
    summary, drivers, group_audit = (
        run_hierarchy_weight_simulation(
            production_scores,
            hierarchy,
            simulations=int(
                simulation["simulations"]
            ),
            random_seed=int(
                simulation["random_seed"]
            ),
            within_group_concentration=float(
                simulation[
                    "within_group_concentration"
                ]
            ),
        )
    )

    write_parquet(
        production_scores,
        settings.processed_dir
        / "production_category_scores.parquet",
    )
    write_parquet(
        central,
        settings.processed_dir
        / "production_hierarchy_scores.parquet",
    )
    write_parquet(
        summary,
        settings.processed_dir
        / "weight_simulation_summary.parquet",
    )
    write_parquet(
        drivers,
        settings.processed_dir
        / "weight_simulation_drivers.parquet",
    )
    write_parquet(
        group_audit,
        settings.processed_dir
        / "hierarchy_weight_simulation_group_audit.parquet",
    )

    metadata = {
        "simulations": int(
            simulation["simulations"]
        ),
        "random_seed": int(
            simulation["random_seed"]
        ),
        "within_group_concentration": float(
            simulation[
                "within_group_concentration"
            ]
        ),
        "production_scale": final_config[
            "production_scale"
        ],
        "production_scale_categories": (
            final_config[
                "production_scale_categories"
            ]
        ),
        "native_scale_categories": (
            final_config[
                "native_scale_categories"
            ]
        ),
        "release_gate": gate_metadata,
        "note": (
            "Playoff expectation training requires "
            "data/manual/playoff_series.csv."
        ),
    }
    (
        settings.model_dir
        / "training_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\\n",
        encoding="utf-8",
    )
''',
    )

    hierarchy_path = Path(
        "configs/category_hierarchy.json"
    )
    hierarchy = json.loads(
        hierarchy_path.read_text(encoding="utf-8")
    )
    hierarchy["production_weights_frozen"] = True
    hierarchy["final_simulation_allowed"] = True
    hierarchy_path.write_text(
        json.dumps(
            hierarchy,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Updated:", hierarchy_path)

    checklist = Path(
        "docs/V1_RELEASE_CHECKLIST.md"
    )
    lines = checklist.read_text(
        encoding="utf-8"
    ).splitlines()
    entry = (
        "- [x] Final production scale, hierarchy weights, "
        "simulation seed, and release gate frozen"
    )
    if entry not in lines:
        index = next(
            (
                i
                for i, line in enumerate(lines)
                if (
                    "Cultural-impact weighting "
                    "sensitivity frozen"
                    in line
                )
            ),
            len(lines) - 1,
        )
        lines.insert(index + 1, entry)
        checklist.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        print("Updated:", checklist)
    else:
        print("Already updated:", checklist)


if __name__ == "__main__":
    main()
