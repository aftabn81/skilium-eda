"""Visualization engine — lean chart generation with extracted helpers."""

from __future__ import annotations

import base64
import logging
import os
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")
matplotlib.rcParams["figure.dpi"] = 150
matplotlib.rcParams["savefig.dpi"] = 150
matplotlib.rcParams["figure.figsize"] = (10, 6)
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["axes.titlesize"] = 13
matplotlib.rcParams["axes.labelsize"] = 11

_CORR_CMAP = "RdBu_r"


class ChartEngine:
    """Create and save EDA charts."""

    def __init__(self, df: pd.DataFrame, output_dir: str = "./skilium_output/charts") -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
        self.df = df
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    def _next_file(self, name: str, ext: str = "png") -> str:
        self._counter += 1
        return f"{self._counter:03d}_{name}.{ext}"

    def _save(self, fig: plt.Figure, name: str) -> str:
        fname = self._next_file(name)
        fpath = self.output_dir / fname
        fig.savefig(fpath, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        logger.info("Saved chart: %s", fpath)
        return str(fpath)

    def _style_fig(self, ax: matplotlib.axes.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def _empty_chart(self, name: str, message: str) -> str:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14, color="gray")
        ax.set_axis_off()
        return self._save(fig, name)

    def _chart_description(self, name: str) -> str:
        """Return a human-readable description for a chart type."""
        descriptions: dict[str, str] = {
            "distribution": "Shows the frequency distribution of values. Histogram with KDE overlay for numeric; bar chart for categorical.",
            "correlation": "Heatmap of Pearson correlation coefficients between numeric columns. Red = positive, blue = negative.",
            "missing": "Visualizes missing value patterns across all columns. Helps identify systematic data collection issues.",
            "boxplot": "Box-and-whisker plots showing median, quartiles, and outliers for each numeric column.",
            "pairplot": "Scatterplot matrix showing relationships between pairs of numeric columns with KDE on diagonal.",
        }
        for key, desc in descriptions.items():
            if key in name.lower():
                return desc
        return ""

    @property
    def descriptions(self) -> dict[str, str]:
        """Return descriptions for generated charts."""
        return getattr(self, "_descriptions", {})

    def _numeric_cols(self) -> list[str]:
        return self.df.select_dtypes(include=[np.number]).columns.tolist()

    # ------------------------------------------------------------------ #
    # Distribution plots
    # ------------------------------------------------------------------ #

    def plot_distribution(self, col: str) -> str:
        if col not in self.df.columns:
            raise KeyError(f"Column '{col}' not found")
        series = self.df[col]
        fig, ax = plt.subplots(figsize=(10, 6))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if pd.api.types.is_numeric_dtype(series):
                sns.histplot(series.dropna(), kde=True, stat="density", color="steelblue", edgecolor="white", linewidth=0.5, ax=ax)
                self._style_fig(ax, f"Distribution of '{col}'", col, "Density")
            elif pd.api.types.is_datetime64_any_dtype(series):
                sns.histplot(series.dropna(), kde=False, color="steelblue", ax=ax)
                self._style_fig(ax, f"Distribution of '{col}' (datetime)", col, "Count")
            else:
                vc = series.value_counts().head(20)
                sns.barplot(x=vc.index.astype(str), y=vc.values, palette="viridis", ax=ax)
                self._style_fig(ax, f"Top {len(vc)} values in '{col}'", col, "Count")
                if len(vc) > 8:
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        return self._save(fig, f"distribution_{col}")

    def plot_distributions(self, max_cols: int = 20) -> list[str]:
        paths: list[str] = []
        for col in self.df.columns[:max_cols]:
            try:
                paths.append(self.plot_distribution(col))
            except Exception as exc:
                logger.warning("Skipping '%s': %s", col, exc)
        return paths

    # ------------------------------------------------------------------ #
    # Correlation heatmap
    # ------------------------------------------------------------------ #

    def plot_correlations(self) -> str:
        numeric = self.df[self._numeric_cols()]
        if numeric.empty or numeric.shape[1] < 2:
            return self._empty_chart("correlation_heatmap", "Not enough numeric columns\nfor correlation heatmap")
        corr = numeric.corr(method="pearson")
        n = len(corr.columns)
        fig, ax = plt.subplots(figsize=(max(10, n * 1.2), max(8, n * 1.0)))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=_CORR_CMAP, center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.75, "label": "Pearson r"}, ax=ax)
        self._style_fig(ax, "Correlation Matrix (Pearson)")
        plt.tight_layout()
        return self._save(fig, "correlation_heatmap")

    # ------------------------------------------------------------------ #
    # Missing values
    # ------------------------------------------------------------------ #

    def plot_missing(self) -> str:
        total = self.df.isnull().sum().sum()
        if total == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No missing values!\nDataset is complete.", ha="center", va="center",
                    fontsize=14, color="darkgreen")
            ax.set_axis_off()
            return self._save(fig, "missing_values")
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})
        sns.heatmap(self.df.isnull(), yticklabels=False, cbar=True, cmap="YlOrRd", ax=axes[0])
        self._style_fig(axes[0], "Missing Value Heatmap (yellow = missing)")
        axes[0].set_xlabel("")
        pct = (self.df.isnull().mean() * 100).sort_values(ascending=True)
        pct = pct[pct > 0]
        colors = ["#2ca02c" if v < 5 else "#ff7f0e" if v < 30 else "#d62728" for v in pct.values]
        axes[1].barh(pct.index.astype(str), pct.values, color=colors)
        self._style_fig(axes[1], "Percentage Missing per Column", "% Missing")
        axes[1].axvline(x=5, color="green", linestyle="--", alpha=0.6)
        axes[1].axvline(x=30, color="red", linestyle="--", alpha=0.6)
        plt.tight_layout()
        return self._save(fig, "missing_values")

    # ------------------------------------------------------------------ #
    # Pairplot
    # ------------------------------------------------------------------ #

    def plot_pairplot(self, cols: list[str] | None = None) -> str:
        numeric = self._numeric_cols()
        if not numeric:
            return self._empty_chart("pairplot", "No numeric columns\nfor pairplot")
        cols = (cols or numeric[:5])[:6]
        subset = self.df[cols].dropna()
        if subset.empty:
            return self._empty_chart("pairplot", "No data for pairplot")
        g = sns.pairplot(subset, diag_kind="kde", plot_kws={"alpha": 0.5, "s": 20, "edgecolor": "none"},
                         diag_kws={"fill": True, "alpha": 0.6}, corner=True)
        g.fig.suptitle("Pairwise Relationships", fontweight="bold", y=1.02, fontsize=14)
        g.fig.set_size_inches(max(12, len(cols) * 2.5), max(12, len(cols) * 2.5))
        fname = self._next_file("pairplot")
        fpath = self.output_dir / fname
        g.fig.savefig(fpath, bbox_inches="tight", facecolor="white", dpi=150)
        plt.close(g.fig)
        logger.info("Saved pairplot: %s", fpath)
        return str(fpath)

    # ------------------------------------------------------------------ #
    # Box plots
    # ------------------------------------------------------------------ #

    def plot_boxplots(self, max_cols: int = 12) -> str:
        cols = self._numeric_cols()[:max_cols]
        if not cols:
            return self._empty_chart("boxplots", "No numeric columns\nfor box plots")
        n = len(cols)
        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3))
        axes = np.atleast_1d(axes).flatten()
        for idx, col in enumerate(cols):
            data = self.df[col].dropna()
            if len(data) > 0:
                sns.boxplot(y=data, color="lightblue", ax=axes[idx], width=0.5)
                axes[idx].set_title(col, fontweight="bold", fontsize=10)
                axes[idx].set_ylabel("")
            else:
                axes[idx].set_title(f"{col}\n(all null)", fontweight="bold", fontsize=9, color="red")
                axes[idx].set_axis_off()
        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)
        fig.suptitle("Box Plots of Numeric Features", fontweight="bold", fontsize=14, y=1.01)
        plt.tight_layout()
        return self._save(fig, "boxplots")

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #

    def generate_all(self, config: dict[str, Any] | None = None) -> list[str]:
        """Generate all relevant charts."""
        if config is None:
            config = {}
        max_cols = config.get("max_cols", 20)
        box_max = config.get("boxplot_max_cols", 12)
        pairplot_cols = config.get("pairplot_cols", None)
        paths: list[str] = []
        self._descriptions: dict[str, str] = {}  # NEW
        numeric = self._numeric_cols()

        logger.info("Generating distribution plots ...")
        paths.extend(self.plot_distributions(max_cols=max_cols))

        logger.info("Generating correlation heatmap ...")
        try:
            paths.append(self.plot_correlations())
        except Exception as exc:
            logger.error("Correlation heatmap failed: %s", exc)

        logger.info("Generating missing-value chart ...")
        try:
            paths.append(self.plot_missing())
        except Exception as exc:
            logger.error("Missing chart failed: %s", exc)

        logger.info("Generating box plots ...")
        try:
            paths.append(self.plot_boxplots(max_cols=box_max))
        except Exception as exc:
            logger.error("Box plots failed: %s", exc)

        if 1 < len(numeric) <= 10:
            logger.info("Generating pairplot ...")
            try:
                paths.append(self.plot_pairplot(cols=pairplot_cols))
            except Exception as exc:
                logger.error("Pairplot failed: %s", exc)
        else:
            logger.info("Skipping pairplot (%d numeric cols)", len(numeric))

        # NEW: collect descriptions for all generated charts
        for p in paths:
            self._descriptions[Path(p).stem] = self._chart_description(Path(p).stem)

        logger.info("Chart generation complete — %d charts produced.", len(paths))
        return paths

    # ------------------------------------------------------------------ #
    # Base64 encoding for embedding
    # ------------------------------------------------------------------ #

    @staticmethod
    def encode_image(path: str) -> str:
        """Convert an image file to a base64 data URI."""
        ext = Path(path).suffix.lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "svg": "image/svg+xml"}.get(ext, "image/png")
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"
