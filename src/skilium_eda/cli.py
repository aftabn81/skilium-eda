"""Command-line interface — two commands with rich output."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console(stderr=True)
app = typer.Typer(
    help="Skilium EDA — Agentic Exploratory Data Analysis",
    rich_markup_mode="rich", no_args_is_help=True, add_completion=False,
)

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


def _resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _check_file(path: str) -> Path:
    p = _resolve(path)
    if not p.exists() and not path.startswith("s3://"):
        raise typer.BadParameter(f"File not found: {p}")
    return p


def _banner() -> None:
    console.print()
    console.print(Panel.fit(
        "[bold blue]Skilium EDA[/bold blue]  —  [dim]Agentic Exploratory Data Analysis[/dim]\n"
        "[dim]v0.1.0[/dim]",
        border_style="blue", padding=(0, 2)))
    console.print()


# --------------------------------------------------------------------------- #
# run — Full pipeline
# --------------------------------------------------------------------------- #

@app.command("run")
def cmd_run(
    file: Annotated[str, typer.Argument(help="Path or S3 URL to dataset.")],
    output_dir: Annotated[str, typer.Option("--output-dir", "-o", help="Output directory.")]
    = "./skilium_output",
    fmt: Annotated[Optional[list[str]], typer.Option(
        "--format", "-f", help="Report formats: html, markdown.")]
    = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output.")]
    = False,
) -> None:
    """Run the full EDA pipeline on a dataset."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level)
    if not file.startswith("s3://"):
        _check_file(file)
    _banner()

    formats = fmt or ["html", "markdown"]
    valid = {"html", "markdown"}
    invalid = set(formats) - valid
    if invalid:
        console.print(f"[red]Invalid format(s): {', '.join(invalid)}[/red]")
        raise typer.Exit(code=1)

    from skilium_eda.pipeline import EDAPipeline
    console.print(f"[dim]Source:[/dim]    {file}")
    console.print(f"[dim]Output:[/dim]    {_resolve(output_dir)}")
    console.print(f"[dim]Formats:[/dim]   {', '.join(formats)}")
    console.print()

    pipeline = EDAPipeline(source=file, output_dir=output_dir,
                           config={"report_formats": formats})
    results = pipeline.run()

    if results.get("reports"):
        table = Table(title="Generated Artifacts", show_header=True, header_style="bold")
        table.add_column("Format", style="cyan", no_wrap=True)
        table.add_column("File Path", style="white")
        for f, p in results["reports"].items():
            table.add_row(f.upper(), p)
        console.print()
        console.print(table)
    if results.get("errors"):
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# profile — Profile only
# --------------------------------------------------------------------------- #

@app.command("profile")
def cmd_profile(
    file: Annotated[str, typer.Argument(help="Path or S3 URL to dataset.")],
    output: Annotated[str, typer.Option("--output", "-o", help="Output JSON path.")]
    = "profile.json",
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output.")]
    = False,
) -> None:
    """Generate a data profile (JSON) without running the full pipeline."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level)
    if not file.startswith("s3://"):
        _check_file(file)
    _banner()
    console.print(f"[dim]Source:[/dim]  {file}")
    console.print(f"[dim]Output:[/dim]  {_resolve(output)}")
    console.print()

    with console.status("[cyan]Loading dataset...", spinner="dots"):
        from skilium_eda.core import DataEngine
        df = DataEngine.load(file)
    console.print(f"   [green]✓[/green] Loaded {len(df):,} rows × {len(df.columns)} columns")

    with console.status("[cyan]Profiling...", spinner="dots"):
        profile = DataEngine.profile(df)
    console.print(f"   [green]✓[/green] Profiled {len(df.columns)} columns")

    out_path = _resolve(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2, ensure_ascii=False, default=str)
    console.print()
    console.print(f"[bold green]Profile written to[/bold green] {out_path}")
    _preview(df, profile)


def _preview(df: pd.DataFrame, profile: dict) -> None:
    console.print()
    table = Table(title="Column Preview", show_header=True, header_style="bold")
    table.add_column("Column", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Non-null", justify="right", style="green")
    table.add_column("Missing", justify="right", style="yellow")
    table.add_column("Unique", justify="right", style="blue")
    vars_info = profile.get("columns", {})
    for col in df.columns:
        info = vars_info.get(col, {})
        table.add_row(
            col, info.get("dtype", str(df[col].dtype)),
            str(info.get("count", df[col].count())),
            str(info.get("null_count", int(df[col].isnull().sum()))),
            str(info.get("unique_count", df[col].nunique())),
        )
    console.print(table)
