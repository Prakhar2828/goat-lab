from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from goatlab.models.category_hierarchy import (
    load_hierarchy_config,
)
from goatlab.models.final_model import (
    build_production_category_scores,
    load_final_model_config,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def evaluate_v1_release_gate(
    *,
    processed_dir: str | Path = "data/processed",
    category_scores: pd.DataFrame | None = None,
    hierarchy_config_path: str | Path = (
        "configs/category_hierarchy.json"
    ),
    final_model_config_path: str | Path = (
        "configs/v1_final_model.json"
    ),
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    processed = Path(processed_dir)
    final_config = load_final_model_config(
        final_model_config_path
    )
    hierarchy = load_hierarchy_config(
        hierarchy_config_path
    )

    checks: list[dict[str, Any]] = []

    def add_check(
        name: str,
        passed: bool,
        *,
        value: Any,
        expected: Any,
        severity: str = "blocker",
        detail: str = "",
    ) -> None:
        checks.append(
            {
                "CHECK": name,
                "PASSED": bool(passed),
                "SEVERITY": severity,
                "VALUE": json.dumps(
                    value,
                    sort_keys=True,
                    default=str,
                ),
                "EXPECTED": json.dumps(
                    expected,
                    sort_keys=True,
                    default=str,
                ),
                "DETAIL": detail,
                "RELEASE_BLOCKER": bool(
                    severity == "blocker"
                    and not passed
                ),
            }
        )

    add_check(
        "hierarchy_frozen",
        bool(hierarchy.get("hierarchy_frozen", False)),
        value=hierarchy.get("hierarchy_frozen"),
        expected=True,
    )
    add_check(
        "group_caps_frozen",
        bool(hierarchy.get("group_caps_frozen", False)),
        value=hierarchy.get("group_caps_frozen"),
        expected=True,
    )
    add_check(
        "production_weights_frozen",
        bool(
            hierarchy.get(
                "production_weights_frozen",
                False,
            )
        ),
        value=hierarchy.get(
            "production_weights_frozen"
        ),
        expected=True,
    )
    add_check(
        "hierarchy_unlock_enabled",
        bool(
            hierarchy.get(
                "final_simulation_allowed",
                False,
            )
        ),
        value=hierarchy.get(
            "final_simulation_allowed"
        ),
        expected=True,
    )

    scaling_path = (
        processed
        / "category_scaling_audit.json"
    )
    scaling_exists = scaling_path.exists()
    add_check(
        "category_scaling_audit_exists",
        scaling_exists,
        value=scaling_exists,
        expected=True,
    )
    if scaling_exists:
        scaling = _load_json(scaling_path)
        scenarios = scaling.get("scenarios", [])
        add_check(
            "production_scale_was_audited",
            final_config["production_scale"]
            in scenarios,
            value=final_config["production_scale"],
            expected=scenarios,
        )

    for filename in final_config["mandatory_audits"]:
        path = processed / filename
        exists = path.exists()
        add_check(
            f"mandatory_audit_exists:{filename}",
            exists,
            value=exists,
            expected=True,
        )
        if not exists:
            continue

        payload = _load_json(path)
        blocker_count = int(
            payload.get(
                "release_blockers",
                0,
            )
        )
        add_check(
            f"mandatory_audit_clear:{filename}",
            blocker_count == 0,
            value=blocker_count,
            expected=0,
        )

        if "central_scores_changed" in payload:
            add_check(
                f"central_scores_unchanged:{filename}",
                payload["central_scores_changed"]
                is False,
                value=payload[
                    "central_scores_changed"
                ],
                expected=False,
            )

        if (
            "additional_central_weight_total"
            in payload
        ):
            add_check(
                f"zero_additional_weight:{filename}",
                float(
                    payload[
                        "additional_central_weight_total"
                    ]
                )
                == 0.0,
                value=payload[
                    "additional_central_weight_total"
                ],
                expected=0.0,
            )

    for filename in final_config["advisory_audits"]:
        path = processed / filename
        exists = path.exists()
        add_check(
            f"advisory_audit_exists:{filename}",
            exists,
            value=exists,
            expected=True,
            severity="advisory",
        )
        if exists:
            payload = _load_json(path)
            add_check(
                f"advisory_audit_recorded:{filename}",
                True,
                value=payload.get(
                    "release_blockers",
                    0,
                ),
                expected=(
                    "documented advisory; "
                    "not primary-model eligible"
                ),
                severity="advisory",
                detail=(
                    "Expert evidence is excluded from the "
                    "central score and its blockers are "
                    "preserved as disclosure."
                ),
            )

    if category_scores is None:
        category_path = (
            processed
            / "category_scores.parquet"
        )
        exists = category_path.exists()
        add_check(
            "category_scores_exist",
            exists,
            value=exists,
            expected=True,
        )
        if exists:
            category_scores = pd.read_parquet(
                category_path
            )
    else:
        add_check(
            "category_scores_supplied",
            True,
            value=True,
            expected=True,
        )

    production_scores = pd.DataFrame()
    if category_scores is not None:
        try:
            production_scores = (
                build_production_category_scores(
                    category_scores,
                    final_config,
                )
            )
        except Exception as exc:
            add_check(
                "production_scores_valid",
                False,
                value=str(exc),
                expected=(
                    "complete finite 0-100 scores"
                ),
            )
        else:
            add_check(
                "production_scores_valid",
                True,
                value={
                    "rows": int(
                        len(production_scores)
                    ),
                    "players": sorted(
                        production_scores[
                            "PLAYER_NAME"
                        ].astype(str).tolist()
                    ),
                },
                expected={
                    "rows": 2,
                    "players": [
                        "LeBron James",
                        "Michael Jordan",
                    ],
                },
            )
            add_check(
                "production_player_set",
                set(
                    production_scores[
                        "PLAYER_NAME"
                    ].astype(str)
                )
                == {
                    "LeBron James",
                    "Michael Jordan",
                },
                value=sorted(
                    production_scores[
                        "PLAYER_NAME"
                    ].astype(str).tolist()
                ),
                expected=[
                    "LeBron James",
                    "Michael Jordan",
                ],
            )

    checks_frame = pd.DataFrame(checks)
    blocker_count = int(
        checks_frame["RELEASE_BLOCKER"].sum()
    )
    advisory_failures = int(
        (
            checks_frame["SEVERITY"].eq(
                "advisory"
            )
            & ~checks_frame["PASSED"]
        ).sum()
    )
    final_allowed = bool(
        blocker_count == 0
        and final_config[
            "final_simulation_allowed"
        ]
        and hierarchy[
            "final_simulation_allowed"
        ]
    )

    metadata = {
        "release_blockers": blocker_count,
        "advisory_failures": advisory_failures,
        "checks": int(len(checks_frame)),
        "passed_checks": int(
            checks_frame["PASSED"].sum()
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
        "production_weights_frozen": True,
        "production_scale_frozen": True,
        "simulations": int(
            final_config["simulation"][
                "simulations"
            ]
        ),
        "random_seed": int(
            final_config["simulation"][
                "random_seed"
            ]
        ),
        "within_group_concentration": float(
            final_config["simulation"][
                "within_group_concentration"
            ]
        ),
        "expert_evidence_treatment": (
            "advisory_only_not_primary_model_eligible"
        ),
        "final_simulation_allowed": final_allowed,
    }
    return checks_frame, metadata, production_scores


def assert_v1_release_gate(
    *,
    processed_dir: str | Path = "data/processed",
    category_scores: pd.DataFrame | None = None,
    hierarchy_config_path: str | Path = (
        "configs/category_hierarchy.json"
    ),
    final_model_config_path: str | Path = (
        "configs/v1_final_model.json"
    ),
) -> tuple[dict[str, Any], pd.DataFrame]:
    checks, metadata, production_scores = (
        evaluate_v1_release_gate(
            processed_dir=processed_dir,
            category_scores=category_scores,
            hierarchy_config_path=(
                hierarchy_config_path
            ),
            final_model_config_path=(
                final_model_config_path
            ),
        )
    )
    if not metadata["final_simulation_allowed"]:
        blockers = checks.loc[
            checks["RELEASE_BLOCKER"],
            ["CHECK", "VALUE", "EXPECTED"],
        ]
        raise RuntimeError(
            "Version 1 release gate failed:\n"
            + blockers.to_string(index=False)
        )
    return metadata, production_scores
