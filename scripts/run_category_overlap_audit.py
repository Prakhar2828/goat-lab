from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from goatlab.models.category_hierarchy import (
    build_overlap_audit,
    build_weight_table,
    load_hierarchy_config,
    score_category_hierarchy,
)
from goatlab.settings import settings
from goatlab.utils import write_parquet


CONFIG_PATH = Path("configs/category_hierarchy.json")


def main() -> None:
    settings.ensure_directories()
    config = load_hierarchy_config(CONFIG_PATH)

    reference_path = (
        settings.processed_dir
        / "historical_career_reference.parquet"
    )
    category_path = (
        settings.processed_dir
        / "category_scores.parquet"
    )

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Missing {reference_path}. Build category scores first."
        )
    if not category_path.exists():
        raise FileNotFoundError(
            f"Missing {category_path}. Build category scores first."
        )

    reference = pd.read_parquet(reference_path)
    category_scores = pd.read_parquet(category_path)

    audit = build_overlap_audit(reference, config)
    weights = build_weight_table(config)
    scores = score_category_hierarchy(category_scores, config)

    audit_path = (
        settings.processed_dir
        / "category_overlap_audit.parquet"
    )
    weights_path = (
        settings.processed_dir
        / "category_hierarchy_weights.parquet"
    )
    scores_path = (
        settings.processed_dir
        / "category_hierarchy_scores.parquet"
    )
    metadata_path = (
        settings.processed_dir
        / "category_hierarchy_audit.json"
    )
    report_path = (
        settings.processed_dir
        / "category_hierarchy_audit.txt"
    )

    write_parquet(audit, audit_path)
    write_parquet(weights, weights_path)
    write_parquet(scores, scores_path)

    metadata = {
        "categories": int(weights["CATEGORY"].nunique()),
        "groups": int(weights["GROUP"].nunique()),
        "pair_count": int(len(audit)),
        "declared_dependencies": int(
            audit["DECLARED_DEPENDENCY"].sum()
        ),
        "high_overlap_advisories": int(
            audit["STATUS"].eq(
                "high_overlap_advisory"
            ).sum()
        ),
        "release_blockers": int(
            audit["RELEASE_BLOCKER"].sum()
        ),
        "hierarchy_frozen": bool(
            config["hierarchy_frozen"]
        ),
        "group_caps_frozen": bool(
            config["group_caps_frozen"]
        ),
        "production_weights_frozen": bool(
            config["production_weights_frozen"]
        ),
        "final_simulation_allowed": bool(
            config["final_simulation_allowed"]
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = "\n".join(
        [
            "GOAT Lab category hierarchy audit",
            "=" * 36,
            "",
            json.dumps(
                metadata,
                indent=2,
                sort_keys=True,
            ),
            "",
            "Provisional hierarchy weights",
            "-" * 31,
            weights.to_string(index=False),
            "",
            "Pairwise historical overlap audit",
            "-" * 33,
            audit.to_string(index=False),
            "",
            "Diagnostic hierarchy scores",
            "-" * 27,
            scores.to_string(index=False),
            "",
            "Final simulation remains blocked.",
            "",
        ]
    )
    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print(
        weights[
            [
                "GROUP",
                "CATEGORY",
                "GROUP_CAP",
                "WITHIN_GROUP_WEIGHT",
                "PROVISIONAL_TOTAL_WEIGHT",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        audit[
            [
                "CATEGORY_A",
                "CATEGORY_B",
                "ABS_SPEARMAN_CORRELATION",
                "STATUS",
                "RELEASE_BLOCKER",
            ]
        ].head(15).to_string(index=False)
    )
    print()
    print(scores.to_string(index=False))
    print()
    print(f"Wrote {audit_path}")
    print(f"Wrote {weights_path}")
    print(f"Wrote {scores_path}")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {report_path}")
    print()
    print("Final simulation remains blocked.")


if __name__ == "__main__":
    main()
