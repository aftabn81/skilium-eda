"""Main pipeline — orchestrates the full EDA workflow."""

from __future__ import annotations

import logging
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from skilium_eda.core import DataEngine
from skilium_eda.agent import EDAAgent
from skilium_eda.viz import ChartEngine
from skilium_eda.report import ReportGenerator

logger = logging.getLogger(__name__)
console = Console(stderr=True)


class EDAPipeline:
    """Orchestrate the full EDA pipeline from data ingestion to report generation."""

    def __init__(
        self,
        source: str,
        output_dir: str = "./skilium_output",
        config: dict | None = None,
    ) -> None:
        self.source = source
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self._df: pd.DataFrame | None = None
        self._clean_df: pd.DataFrame | None = None
        self._cleaning_log: dict = {}
        self._profile: dict = {}
        self._insights: list[Any] = []
        self._charts: list[str] = []
        self._reports: dict[str, str] = {}
        self._duration = 0.0
        self._setup_logging()

    # ------------------------------------------------------------------ #
    # Public properties for notebook/Colab access
    # ------------------------------------------------------------------ #

    @property
    def cleaned_df(self) -> pd.DataFrame | None:
        """Access the cleaned DataFrame (None if cleaning hasn't run)."""
        return self._clean_df

    @property
    def profile(self) -> dict:
        """Access the profiling results (empty dict if profiling hasn't run)."""
        return self._profile

    @property
    def insights(self) -> list[Any]:
        """Access generated insights (empty list if insights step hasn't run)."""
        return self._insights

    @property
    def reports(self) -> dict[str, str]:
        """Access generated report file paths."""
        return self._reports

    def run(self) -> dict:
        """Execute the full EDA pipeline."""
        start = time.perf_counter()
        completed: list[str] = []
        errors: list[str] = []

        console.print()
        console.rule(Text("Skilium EDA — Agentic Exploratory Data Analysis", style="bold blue"))
        console.print()

        steps = [
            ("load", "Loading data", self._step_load),
            ("clean", "Cleaning data", self._step_clean),
            ("profile", "Profiling data", self._step_profile),
            ("insights", "Generating insights", self._step_insights),
            ("visualize", "Creating visualizations", self._step_visualize),
            ("report", "Building reports", self._step_report),
        ]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console, transient=False,
        ) as progress:
            task = progress.add_task("[cyan]Running pipeline...", total=len(steps))
            for key, label, fn in steps:
                progress.update(task, description=f"[cyan]{label}...")
                try:
                    fn()
                    completed.append(key)
                    logger.info("Step '%s' completed", key)
                except Exception as exc:
                    err_msg = f"Step '{key}' failed: {exc}"
                    logger.error(err_msg)
                    logger.debug(traceback.format_exc())
                    errors.append(err_msg)
                    console.print(f"[yellow]⚠  {label} skipped — {exc}[/yellow]")
                progress.advance(task)

        self._duration = time.perf_counter() - start
        shape = self._clean_df.shape if self._clean_df is not None else (0, 0)
        summary = {
            "source": self.source,
            "output_dir": str(self.output_dir),
            "shape": shape,
            "profile": self._profile,
            "insights_count": len(self._insights),
            "charts": self._charts,
            "reports": self._reports,
            "duration_seconds": round(self._duration, 2),
            "steps_completed": completed,
            "errors": errors,
        }
        self._print_summary(summary)
        return summary

    def run_step(self, step: str) -> Any:
        """Run an individual pipeline step."""
        step_map = {
            "load": self._step_load, "clean": self._step_clean,
            "profile": self._step_profile, "insights": self._step_insights,
            "visualize": self._step_visualize, "report": self._step_report,
        }
        if step not in step_map:
            raise ValueError(f"Unknown step '{step}'. Choose from: {', '.join(step_map)}")
        console.print(f"[dim]Running step:[/dim] [bold]{step}[/bold]")
        return step_map[step]()

    def _step_load(self) -> pd.DataFrame:
        self._df = DataEngine.load(self.source)
        setattr(self._df, "_skilium_name", Path(self.source).stem)
        console.print(f"   [green]✓[/green] Loaded [bold]{len(self._df):,}[/bold] rows × [bold]{len(self._df.columns)}[/bold] cols")
        return self._df

    def _step_clean(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("Load data first.")
        strategy = self.config.get("missing_strategy", "auto")
        self._clean_df, self._cleaning_log = DataEngine.clean(self._df.copy(), strategy=strategy)
        console.print(f"   [green]✓[/green] Cleaned: [bold]{len(self._clean_df):,}[/bold] rows")
        return self._clean_df

    def _step_profile(self) -> dict:
        target = self._clean_df if self._clean_df is not None else self._df
        if target is None:
            raise RuntimeError("Load data first.")
        self._profile = DataEngine.profile(target)
        console.print(f"   [green]✓[/green] Profiled [bold]{len(target.columns)}[/bold] columns")
        return self._profile

    def _step_insights(self) -> list[Any]:
        target = self._clean_df if self._clean_df is not None else self._df
        if target is None:
            raise RuntimeError("Load data first.")
        agent = EDAAgent(target)  # Agent computes its own profile
        self._insights = agent.generate_insights()
        console.print(f"   [green]✓[/green] Generated [bold]{len(self._insights)}[/bold] insights")
        return self._insights

    def _step_visualize(self) -> list[str]:
        target = self._clean_df if self._clean_df is not None else self._df
        if target is None:
            raise RuntimeError("Load data first.")
        chart_dir = self.output_dir / "charts"
        chart_dir.mkdir(parents=True, exist_ok=True)
        engine = ChartEngine(target, str(chart_dir))
        self._charts = engine.generate_all()
        self._chart_descriptions = engine.descriptions  # NEW
        console.print(f"   [green]✓[/green] Generated [bold]{len(self._charts)}[/bold] charts")
        return self._charts

    def _step_report(self) -> dict[str, str]:
        target = self._clean_df if self._clean_df is not None else self._df
        if target is None:
            raise RuntimeError("Load data first.")
        fmt = self.config.get("report_formats", ["html", "markdown"])
        reports: dict[str, str] = {}
        gen = ReportGenerator(
            df=target, profile=self._profile, charts=self._charts,
            insights=self._insights, cleaning_log=self._cleaning_log,
            original_df=self._df,  # NEW: pass original df for before/after
            chart_descriptions=getattr(self, "_chart_descriptions", None),  # NEW
            output_dir=str(self.output_dir),
        )
        if "html" in fmt:
            try:
                reports["html"] = gen.to_html()
            except Exception as exc:
                logger.error("HTML report failed: %s", exc)
        if "markdown" in fmt:
            try:
                reports["markdown"] = gen.to_markdown()
            except Exception as exc:
                logger.error("Markdown report failed: %s", exc)
        self._reports = reports
        console.print(f"   [green]✓[/green] Built [bold]{len(reports)}[/bold] report(s)")
        return reports

    def display_report(self) -> None:
        """Display the HTML report inline in a Jupyter/Colab notebook.

        If not in a notebook environment, prints the report file path.
        """
        html_path = self._reports.get("html")
        if not html_path or not Path(html_path).exists():
            console.print("[yellow]No HTML report found. Run pipeline first.[/yellow]")
            return

        # Try IPython display (Jupyter/Colab)
        try:
            from IPython.display import HTML, display, IFrame
            import os

            # In Colab/Jupyter, display the HTML inline
            if os.environ.get("COLAB_RELEASE_TAG") or self._in_notebook():
                html_content = Path(html_path).read_text(encoding="utf-8")
                # Resize images to relative paths if they exist
                display(HTML(html_content))
                return
        except ImportError:
            pass

        # Fallback: print the path
        console.print(f"[green]HTML report:[/green] {html_path}")
        console.print(f"[dim]Open in browser: file://{Path(html_path).resolve()}[/dim]")

    @staticmethod
    def _in_notebook() -> bool:
        """Detect if running inside a Jupyter/Colab notebook."""
        try:
            from IPython import get_ipython
            if get_ipython() is None:
                return False
            shell = get_ipython().__class__.__name__
            return shell in ("ZMQInteractiveShell", "Google.Colab.interactiveshell")
        except Exception:
            return False

    @staticmethod
    def _setup_logging() -> None:
        logging.basicConfig(
            level=logging.INFO, format="%(message)s", datefmt="[%X]",
            handlers=[RichHandler(console=console, rich_tracebacks=True)],
        )
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

    def _print_summary(self, summary: dict) -> None:
        console.print()
        console.rule(Text("Pipeline Complete", style="bold green"))
        console.print()
        table = Table(title="EDA Results", show_header=True, header_style="bold")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")
        table.add_row("Source", str(summary["source"]))
        table.add_row("Output Directory", str(summary["output_dir"]))
        shape = summary["shape"]
        table.add_row("Dataset Shape", f"{shape[0]:,} rows × {shape[1]} columns")
        table.add_row("Insights", str(summary["insights_count"]))
        table.add_row("Charts", str(len(summary["charts"])))
        table.add_row("Duration", f"{summary['duration_seconds']:.2f}s")
        if summary["errors"]:
            table.add_row("Errors", f"[red]{len(summary['errors'])} step(s) failed[/red]")
        console.print(table)
        if summary["reports"]:
            rtable = Table(title="Generated Reports", show_header=True, header_style="bold")
            rtable.add_column("Format", style="cyan")
            rtable.add_column("Path", style="white")
            for f, p in summary["reports"].items():
                rtable.add_row(f.upper(), p)
            console.print()
            console.print(rtable)
        if not summary["errors"]:
            console.print(Panel.fit(
                "[bold green]All steps completed successfully![/bold green]\nOpen the HTML report to explore results.",
                title="Success", border_style="green"))
        else:
            console.print(Panel.fit(
                f"[bold yellow]Pipeline completed with warnings.[/bold yellow]\nCompleted: {', '.join(summary['steps_completed'])}",
                title="Partial Success", border_style="yellow"))
        console.print()
