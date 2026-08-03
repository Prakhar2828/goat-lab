from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelReport:
    auc: float
    log_loss: float
    brier: float
    train_seasons: tuple[str, ...]
    test_seasons: tuple[str, ...]
    evaluation_method: str = "chronological_holdout"
    folds: int = 1


FEATURES = [
    "TEAM_SRS",
    "OPP_SRS",
    "TEAM_NET_RATING",
    "OPP_NET_RATING",
    "TEAM_SEED",
    "OPP_SEED",
    "HOME_COURT",
    "REST_ADVANTAGE",
    "TEAM_STAR_VALUE",
    "OPP_STAR_VALUE",
    "TEAM_SUPPORT_VALUE",
    "OPP_SUPPORT_VALUE",
]


def _available_features(
    series: pd.DataFrame,
) -> list[str]:
    return [
        column
        for column in FEATURES
        if column in series.columns
        and pd.to_numeric(
            series[column],
            errors="coerce",
        ).notna().any()
    ]


def _validate_series(
    series: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    required = {
        "SEASON",
        "TEAM_WON_SERIES",
    }

    missing = required.difference(
        series.columns
    )

    if missing:
        raise ValueError(
            "Series table is missing columns: "
            f"{sorted(missing)}"
        )

    if len(feature_columns) < 4:
        raise ValueError(
            "At least four contextual features "
            "are required to train the model."
        )

    labels = pd.to_numeric(
        series["TEAM_WON_SERIES"],
        errors="coerce",
    )

    if labels.isna().any():
        raise ValueError(
            "TEAM_WON_SERIES contains "
            "missing or invalid values."
        )

    if not set(labels.astype(int).unique()).issubset(
        {0, 1}
    ):
        raise ValueError(
            "TEAM_WON_SERIES must contain only 0 and 1."
        )


def build_pipeline(
    feature_columns: list[str],
) -> Pipeline:
    numeric_pipeline = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            (
                "scale",
                StandardScaler(),
            ),
        ]
    )

    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                numeric_pipeline,
                feature_columns,
            )
        ],
        remainder="drop",
    )

    return Pipeline(
        [
            (
                "preprocess",
                preprocess,
            ),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    max_iter=5_000,
                    random_state=23,
                ),
            ),
        ]
    )


def _build_report(
    labels: pd.Series,
    probabilities: np.ndarray,
    train_seasons: tuple[str, ...],
    test_seasons: tuple[str, ...],
    evaluation_method: str,
    folds: int,
) -> ModelReport:
    labels = pd.to_numeric(
        labels,
        errors="raise",
    ).astype(int)

    return ModelReport(
        auc=(
            float(
                roc_auc_score(
                    labels,
                    probabilities,
                )
            )
            if labels.nunique() > 1
            else float("nan")
        ),
        log_loss=float(
            log_loss(
                labels,
                probabilities,
                labels=[0, 1],
            )
        ),
        brier=float(
            brier_score_loss(
                labels,
                probabilities,
            )
        ),
        train_seasons=train_seasons,
        test_seasons=test_seasons,
        evaluation_method=evaluation_method,
        folds=folds,
    )


def _add_prediction_columns(
    series: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    result = series.copy()

    result[
        "EXPECTED_SERIES_WIN_PROB"
    ] = probabilities

    result["SERIES_OVERPERFORMANCE"] = (
        pd.to_numeric(
            result["TEAM_WON_SERIES"],
            errors="raise",
        ).astype(float)
        - result[
            "EXPECTED_SERIES_WIN_PROB"
        ]
    )

    eps = 1e-6

    labels = pd.to_numeric(
        result["TEAM_WON_SERIES"],
        errors="raise",
    ).astype(int)

    result["SURPRISE_LOG_SCORE"] = np.where(
        labels == 1,
        -np.log(
            np.clip(
                result[
                    "EXPECTED_SERIES_WIN_PROB"
                ],
                eps,
                1,
            )
        ),
        -np.log(
            np.clip(
                1
                - result[
                    "EXPECTED_SERIES_WIN_PROB"
                ],
                eps,
                1,
            )
        ),
    )

    return result


def train_playoff_series_model(
    series: pd.DataFrame,
    artifact_path: str | Path = (
        "models/playoff_series_logit.joblib"
    ),
) -> tuple[Pipeline, ModelReport]:
    """Train with a chronological holdout.

    This remains available for model diagnostics. Career scoring
    should use cross_fit_series_overperformance instead.
    """

    feature_columns = _available_features(
        series
    )

    _validate_series(
        series,
        feature_columns,
    )

    work = series.copy()

    work["SEASON"] = (
        work["SEASON"].astype(str)
    )

    seasons = sorted(
        work["SEASON"]
        .dropna()
        .unique()
    )

    if len(seasons) < 2:
        raise ValueError(
            "At least two seasons are required."
        )

    split_index = min(
        len(seasons) - 1,
        max(
            1,
            int(len(seasons) * 0.8),
        ),
    )

    train_seasons = tuple(
        seasons[:split_index]
    )

    test_seasons = tuple(
        seasons[split_index:]
    )

    train = work[
        work["SEASON"].isin(
            train_seasons
        )
    ]

    test = work[
        work["SEASON"].isin(
            test_seasons
        )
    ]

    model = build_pipeline(
        feature_columns
    )

    model.fit(
        train[feature_columns],
        train[
            "TEAM_WON_SERIES"
        ].astype(int),
    )

    probabilities = model.predict_proba(
        test[feature_columns]
    )[:, 1]

    report = _build_report(
        labels=test[
            "TEAM_WON_SERIES"
        ],
        probabilities=probabilities,
        train_seasons=train_seasons,
        test_seasons=test_seasons,
        evaluation_method=(
            "chronological_holdout"
        ),
        folds=1,
    )

    artifact_path = Path(
        artifact_path
    )

    artifact_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "features": feature_columns,
            "report": report,
        },
        artifact_path,
    )

    return model, report


def cross_fit_series_overperformance(
    series: pd.DataFrame,
    n_splits: int = 10,
    artifact_path: str | Path = (
        "models/playoff_series_logit.joblib"
    ),
) -> tuple[pd.DataFrame, ModelReport]:
    """Generate season-grouped out-of-fold probabilities.

    Every season is scored by a model that did not train on any
    series from that season. This prevents in-sample career scoring
    and applies the same evaluation method across eras.
    """

    feature_columns = _available_features(
        series
    )

    _validate_series(
        series,
        feature_columns,
    )

    work = (
        series.reset_index(
            drop=True
        )
        .copy()
    )

    work["SEASON"] = (
        work["SEASON"].astype(str)
    )

    unique_seasons = sorted(
        work["SEASON"].unique()
    )

    if len(unique_seasons) < 2:
        raise ValueError(
            "At least two seasons are required "
            "for grouped cross-fitting."
        )

    fold_count = min(
        max(2, int(n_splits)),
        len(unique_seasons),
    )

    splitter = GroupKFold(
        n_splits=fold_count
    )

    probabilities = np.full(
        len(work),
        np.nan,
        dtype="float64",
    )

    fold_ids = np.full(
        len(work),
        -1,
        dtype="int64",
    )

    labels = pd.to_numeric(
        work["TEAM_WON_SERIES"],
        errors="raise",
    ).astype(int)

    groups = work["SEASON"]

    for fold_id, (
        train_index,
        test_index,
    ) in enumerate(
        splitter.split(
            work[feature_columns],
            labels,
            groups=groups,
        ),
        start=1,
    ):
        fold_model = build_pipeline(
            feature_columns
        )

        fold_model.fit(
            work.loc[
                train_index,
                feature_columns,
            ],
            labels.loc[
                train_index
            ],
        )

        probabilities[
            test_index
        ] = fold_model.predict_proba(
            work.loc[
                test_index,
                feature_columns,
            ]
        )[:, 1]

        fold_ids[
            test_index
        ] = fold_id

    if np.isnan(probabilities).any():
        raise RuntimeError(
            "Some rows did not receive "
            "out-of-fold predictions."
        )

    result = _add_prediction_columns(
        work,
        probabilities,
    )

    result["CV_FOLD"] = fold_ids
    result["PREDICTION_SOURCE"] = (
        "season_grouped_out_of_fold"
    )

    report = _build_report(
        labels=labels,
        probabilities=probabilities,
        train_seasons=tuple(
            unique_seasons
        ),
        test_seasons=tuple(
            unique_seasons
        ),
        evaluation_method=(
            "season_grouped_out_of_fold"
        ),
        folds=fold_count,
    )

    # Fit a final all-data model for future prediction use.
    # The historical scored table above still uses only OOF values.
    final_model = build_pipeline(
        feature_columns
    )

    final_model.fit(
        work[feature_columns],
        labels,
    )

    artifact_path = Path(
        artifact_path
    )

    artifact_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": final_model,
            "features": feature_columns,
            "report": report,
        },
        artifact_path,
    )

    return result, report


def add_series_overperformance(
    series: pd.DataFrame,
    model: Pipeline,
    features: list[str],
) -> pd.DataFrame:
    """Score new series using a previously trained model."""

    probabilities = model.predict_proba(
        series[features]
    )[:, 1]

    return _add_prediction_columns(
        series,
        probabilities,
    )
