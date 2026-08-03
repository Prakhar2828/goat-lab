from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from goatlab.data.cultural import ingest_wikimedia
from goatlab.data.ingest_core import run_core_ingestion
from goatlab.data.manual_imports import import_manual_advanced, import_mvp_votes
from goatlab.models.playoff_expectation import (
    cross_fit_series_overperformance,
)
from goatlab.pipeline import build_features, train_models
from goatlab.settings import settings
from goatlab.utils import write_parquet

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("ingest-core")
def ingest_core(skip_game_logs: bool = False) -> None:
    """Download and cache NBA season, team, game-log, and award data."""
    run_core_ingestion(include_game_logs=not skip_game_logs)
    console.print("[green]Core ingestion complete.[/green]")


@app.command("ingest-cultural")
def ingest_cultural() -> None:
    """Fetch Wikimedia attention data."""
    ingest_wikimedia()
    console.print("[green]Cultural attention ingestion complete.[/green]")


@app.command("import-advanced")
def import_advanced(path: Path) -> None:
    import_manual_advanced(path)
    console.print("[green]Manual advanced metrics imported.[/green]")


@app.command("import-mvp-votes")
def import_votes(path: Path) -> None:
    import_mvp_votes(path)
    console.print("[green]MVP voting data imported.[/green]")


@app.command("build-features")
def build_features_command(skip_team_context: bool = False) -> None:
    build_features(include_team_context=not skip_team_context)
    console.print("[green]Feature tables built.[/green]")


@app.command("train-models")
def train_models_command() -> None:
    """Cross-fit playoff context and run the final simulations."""

    series_path = (
        settings.manual_dir
        / "playoff_series.csv"
    )

    if series_path.exists():
        series = pd.read_csv(
            series_path
        )

        scored, report = (
            cross_fit_series_overperformance(
                series,
                n_splits=10,
            )
        )

        output_path = (
            settings.processed_dir
            / "playoff_series_scored.parquet"
        )

        write_parquet(
            scored,
            output_path,
        )

        console.print(
            "[green]"
            "Playoff context cross-fit complete. "
            f"Folds={report.folds}, "
            f"AUC={report.auc:.3f}, "
            f"Brier={report.brier:.3f}"
            "[/green]"
        )
    else:
        console.print(
            "[yellow]"
            "Skipped playoff model: add "
            "data/manual/playoff_series.csv."
            "[/yellow]"
        )

    train_models()

    console.print(
        "[green]"
        "Model training complete."
        "[/green]"
    )


@app.command("all")
def run_all(skip_game_logs: bool = False) -> None:
    run_core_ingestion(include_game_logs=not skip_game_logs)
    build_features(include_team_context=not skip_game_logs)
    train_models_command()
