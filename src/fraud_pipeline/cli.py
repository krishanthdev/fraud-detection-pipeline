"""Command line entry point.

One command per pipeline stage, in the order the stages run:

    fraud ingest      stage 1
    fraud validate    stage 2
    fraud eda         exploration and feature selection
    fraud features    stage 3
    fraud train       stage 4
    fraud evaluate    stage 5
    fraud register    stage 6
    fraud serve       stage 7
    fraud run-all     stages 1 to 6 in sequence

Every command takes the same global options, so a single config file drives the run:

    fraud --config configs/config.yaml --set project.seed=7 train
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from fraud_pipeline import __version__
from fraud_pipeline import eda as eda_stage
from fraud_pipeline import ingest as ingest_stage
from fraud_pipeline import validation as validation_stage
from fraud_pipeline.config import Config, load_config
from fraud_pipeline.eda import EdaError
from fraud_pipeline.feature_selection import SelectionError
from fraud_pipeline.ingest import IngestError
from fraud_pipeline.logging_utils import setup_logging, stage_banner
from fraud_pipeline.seeding import set_global_seed
from fraud_pipeline.splits import SplitError
from fraud_pipeline.validation import ValidationError

console = Console()

app = typer.Typer(
    name="fraud",
    help="End to end credit card fraud detection pipeline.",
    add_completion=False,
    no_args_is_help=True,
)


class RunContext:
    """Shared state built once by the callback and read by every command."""

    def __init__(self, config_path: Path | None, overrides: list[str], log_level: str) -> None:
        self.log_level = log_level
        self.logger = setup_logging(log_level)
        self.config: Config = load_config(config_path, overrides)


def _context(ctx: typer.Context) -> RunContext:
    return ctx.ensure_object(RunContext)


def _not_built_yet(stage: str, branch: str) -> None:
    """Report honestly that a stage is scaffolded but not implemented."""
    console.print(f"[yellow]Stage '{stage}' is not implemented yet.[/yellow]")
    console.print(f"It arrives on branch [bold]{branch}[/bold].")
    raise typer.Exit(code=2)


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", "-c", help="Path to the config file. Defaults to configs/config.yaml."
        ),
    ] = None,
    set_: Annotated[
        list[str] | None,
        typer.Option(
            "--set", "-s", help="Override a config value, for example project.seed=7. Repeatable."
        ),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="DEBUG, INFO, WARNING or ERROR."),
    ] = "INFO",
) -> None:
    """Load the config and set up logging before any command runs."""
    ctx.obj = RunContext(config, list(set_ or []), log_level)


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"fraud-pipeline {__version__}")


@app.command("show-config")
def show_config(ctx: typer.Context) -> None:
    """Print the resolved configuration, after any --set overrides."""
    cfg = _context(ctx).config
    spec = cfg.dataset.spec()

    table = Table(title="Resolved configuration", show_lines=False)
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    rows = [
        ("project.name", cfg.project.name),
        ("project.seed", str(cfg.project.seed)),
        ("dataset.active", cfg.dataset.active),
        ("dataset.file", spec.raw_file),
        ("dataset.target", spec.target_column),
        (
            "split",
            f"{cfg.split.strategy} {cfg.split.train_fraction}/{cfg.split.validation_fraction}/{cfg.split.test_fraction}",
        ),
        ("imbalance.strategies", ", ".join(cfg.imbalance.strategies)),
        ("models.enabled", ", ".join(cfg.enabled_models()) or "none"),
        ("evaluation.primary_metric", cfg.evaluation.primary_metric),
        ("evaluation.threshold_strategy", cfg.evaluation.threshold_strategy),
        ("registry.tracking_uri", cfg.registry.tracking_uri),
        ("paths.raw", str(cfg.paths.raw())),
        ("paths.processed", str(cfg.paths.processed())),
        ("paths.models", str(cfg.paths.model_dir())),
    ]
    for name, value in rows:
        table.add_row(name, value)

    console.print(table)


def _do_ingest(rc: RunContext) -> None:
    """Stage 1, shared by the `ingest` command and by `run-all`."""
    stage_banner(rc.logger, "1 ingest", rc.config.dataset.spec().name)
    set_global_seed(rc.config.project.seed)

    try:
        manifest = ingest_stage.run(rc.config)
    except IngestError as error:
        console.print(f"[red]Ingest failed.[/red]\n{error}")
        raise typer.Exit(code=1) from error

    console.print(
        f"[green]Ingested[/green] {manifest.rows:,} rows, "
        f"{manifest.fraud_rows} fraud ({manifest.fraud_rate * 100:.3f} percent)."
    )


def _do_validate(rc: RunContext) -> None:
    """Stage 2, shared by the `validate` command and by `run-all`."""
    stage_banner(rc.logger, "2 validate", rc.config.dataset.spec().name)
    set_global_seed(rc.config.project.seed)

    try:
        report = validation_stage.run(rc.config)
    except (IngestError, SplitError) as error:
        console.print(f"[red]Validation could not run.[/red]\n{error}")
        raise typer.Exit(code=1) from error
    except ValidationError as error:
        console.print(f"[red]Validation failed.[/red]\n{error}")
        raise typer.Exit(code=1) from error

    warned = len(report.warnings)
    tail = f", {warned} warning(s)" if warned else ""
    console.print(f"[green]Validation passed[/green] with {len(report.results)} checks{tail}.")


@app.command()
def ingest(ctx: typer.Context) -> None:
    """Stage 1. Read the raw transaction file and write a typed interim table."""
    _do_ingest(_context(ctx))


@app.command()
def validate(ctx: typer.Context) -> None:
    """Stage 2. Check the schema and the data quality rules, then split the data."""
    _do_validate(_context(ctx))


def _do_eda(rc: RunContext) -> None:
    """Exploration and feature selection, shared by the `eda` command and by `run-all`."""
    stage_banner(rc.logger, "eda", rc.config.dataset.spec().name)
    set_global_seed(rc.config.project.seed)

    try:
        stats, selection = eda_stage.run(rc.config)
    except (SplitError, EdaError, SelectionError) as error:
        console.print(f"[red]Exploration failed.[/red]\n{error}")
        raise typer.Exit(code=1) from error

    counts = selection.counts()
    table = Table(title="Feature selection", show_lines=False)
    table.add_column("Set", style="cyan")
    table.add_column("Features", justify="right")
    table.add_column("Meaning", style="white")
    table.add_row("all", str(counts["all"]), "no selection")
    table.add_row("safe", str(counts["safe"]), "tier 1, the correctness drops")
    table.add_row("selected", str(counts["selected"]), "tier 1 and tier 2")
    console.print(table)

    dropped = selection.dropped()
    if dropped:
        console.print(f"[yellow]Dropped from '{selection.active_set}':[/yellow]")
        for decision in dropped:
            console.print(f"  {decision.feature}: {decision.reasons[0]}")

    console.print(
        f"[green]Analysed[/green] {len(stats)} features. "
        f"Active set '{selection.active_set}' keeps {counts[selection.active_set]}."
    )


@app.command()
def eda(ctx: typer.Context) -> None:
    """Explore the training split, then choose features from what it finds."""
    _do_eda(_context(ctx))


@app.command()
def features(ctx: typer.Context) -> None:
    """Stage 3. Build model ready features and write the train, validation and test sets."""
    _context(ctx)
    _not_built_yet("features", "feature/feature-engineering")


@app.command()
def train(ctx: typer.Context) -> None:
    """Stage 4. Fit every enabled model under every imbalance strategy."""
    _context(ctx)
    _not_built_yet("train", "feature/baseline-models")


@app.command()
def evaluate(ctx: typer.Context) -> None:
    """Stage 5. Score the models, tune the threshold and write the results table."""
    _context(ctx)
    _not_built_yet("evaluate", "feature/evaluation-and-results-table")


@app.command("register")
def register_model(ctx: typer.Context) -> None:
    """Stage 6. Promote the best run to champion in the MLflow model registry."""
    _context(ctx)
    _not_built_yet("register", "feature/evaluation-and-results-table")


@app.command()
def serve(ctx: typer.Context) -> None:
    """Stage 7. Start the FastAPI scoring service."""
    _context(ctx)
    _not_built_yet("serve", "feature/serving-api")


@app.command("run-all")
def run_all(ctx: typer.Context) -> None:
    """Run stages 1 to 6 in order, which is what the Makefile target `make all` calls."""
    rc = _context(ctx)
    _do_ingest(rc)
    _do_validate(rc)
    _do_eda(rc)
    _not_built_yet("features", "feature/feature-engineering")


if __name__ == "__main__":  # pragma: no cover
    app()
