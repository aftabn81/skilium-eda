"""Agentic intelligence layer — rule-based EDA decisions and insights."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

logger = logging.getLogger(__name__)


class Insight(BaseModel):
    """A single data-driven insight produced by the agent."""

    type: str = Field(..., pattern=r"^(distribution|correlation|quality|outlier|recommendation)$")
    column: str | None = None
    title: str
    description: str
    severity: str = Field(..., pattern=r"^(info|warning|critical)$")
    recommendation: str


class EDAAgent:
    """Rule-based agent that drives intelligent EDA decisions."""

    def __init__(
        self,
        df: pd.DataFrame,
        profile: dict[str, Any] | None = None,
        target_column: str | None = None,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
        self.df = df
        self.profile = profile if profile is not None else self._compute_profile()
        self.target_column = target_column

    def _compute_profile(self) -> dict[str, Any]:
        from skilium_eda.core import DataEngine
        type_map = DataEngine.infer_types(self.df)

        numeric_cols = []
        categorical_cols = []
        for col in self.df.columns:
            t = type_map.get(col, "")
            if t == "id":
                continue
            elif t == "categorical":
                categorical_cols.append(col)
            elif pd.api.types.is_numeric_dtype(self.df[col].dtype) and t == "numeric":
                numeric_cols.append(col)
            elif pd.api.types.is_numeric_dtype(self.df[col].dtype):
                if t not in ("id", "categorical"):
                    numeric_cols.append(col)

        for col in self.df.select_dtypes(include=["object", "category", "bool"]).columns:
            if col not in categorical_cols:
                categorical_cols.append(col)

        missing = int(self.df.isnull().sum().sum())
        total = self.df.size
        return {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "datetime_cols": self.df.select_dtypes(include=["datetime", "datetime64[ns]"]).columns.tolist(),
            "missing_cells": missing,
            "missing_percent": (missing / total * 100) if total > 0 else 0.0,
            "duplicate_rows": int(self.df.duplicated().sum()),
            "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            "column_types": type_map,
        }

    def _numeric_cols(self) -> list[str]:
        return self.profile.get("numeric_cols", [])

    def _categorical_cols(self) -> list[str]:
        return self.profile.get("categorical_cols", [])

    def _datetime_cols(self) -> list[str]:
        return self.profile.get("datetime_cols", [])

    def _missing_series(self) -> pd.Series:
        return (self.df.isnull().mean() * 100).sort_values(ascending=False)

    # ------------------------------------------------------------------ #
    # Analysis decisions
    # ------------------------------------------------------------------ #

    def decide_analyses(self) -> list[str]:
        """Decide which EDA steps to run based on data characteristics."""
        analyses: list[str] = ["profile"]
        p = self.profile
        if p["missing_percent"] > 0:
            analyses.append("missing")
        if p["duplicate_rows"] > 0 or p["total_rows"] > 10_000:
            analyses.append("duplicate")
        analyses.append("distribution")
        if len(self._numeric_cols()) >= 2:
            analyses.append("correlation")
        if len(self._numeric_cols()) > 0:
            analyses.append("outlier")
        if self._datetime_cols():
            analyses.append("time_series")
        if self.target_column and self.target_column in self.df.columns:
            analyses.append("target_analysis")
        if any(self.df[col].nunique(dropna=True) > 50 for col in self._categorical_cols()):
            analyses.append("cardinality")
        if p["total_rows"] * p["total_columns"] > 1_000_000:
            analyses.append("sampling")
        logger.info("Decided analyses: %s", analyses)
        return analyses

    # ------------------------------------------------------------------ #
    # Insight generation
    # ------------------------------------------------------------------ #

    def generate_insights(self) -> list[Insight]:
        """Generate data-driven insights from statistical analysis."""
        insights: list[Insight] = []
        insights.extend(self._distribution_insights())
        insights.extend(self._correlation_insights())
        insights.extend(self._quality_insights())
        insights.extend(self._outlier_insights())
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        insights.sort(key=lambda i: severity_order.get(i.severity, 3))
        logger.info("Generated %d insights", len(insights))
        return insights

    def _distribution_insights(self) -> list[Insight]:
        insights: list[Insight] = []
        for col in self._numeric_cols():
            series = self.df[col].dropna()
            if len(series) < 8:
                continue
            try:
                skew_val = float(stats.skew(series))
            except Exception:
                skew_val = 0.0
            try:
                kurt_val = float(stats.kurtosis(series))
            except Exception:
                kurt_val = 0.0
            normal, p_value = True, 1.0
            if len(series) >= 20:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        _, p_value = stats.normaltest(series)
                        normal = p_value > 0.05
                    except Exception:
                        pass
            desc_parts, rec_parts = [], []
            severity = "info"
            if abs(skew_val) > 2:
                direction = "right" if skew_val > 0 else "left"
                desc_parts.append(f"Column '{col}' is heavily {direction}-skewed (skewness={skew_val:.2f}).")
                rec_parts.append(f"Consider a log or power transform for '{col}'.")
                severity = "warning"
            elif abs(skew_val) > 1:
                direction = "right" if skew_val > 0 else "left"
                desc_parts.append(f"Column '{col}' is moderately {direction}-skewed (skewness={skew_val:.2f}).")
                rec_parts.append(f"Monitor '{col}' for modelling — transform if residuals show bias.")
            if kurt_val > 3:
                desc_parts.append(f"High kurtosis ({kurt_val:.2f}) — heavy tails / outliers likely.")
                rec_parts.append("Review outliers; robust scalers may help.")
                sev_map = {"info": 0, "warning": 1, "critical": 2}
                severity = max(severity, "warning", key=lambda s: sev_map.get(s, 0))
            if not normal and len(series) >= 20:
                desc_parts.append(f"Normality test rejected (p={p_value:.4f}).")
                rec_parts.append("Use non-parametric tests or apply a Box-Cox / Yeo-Johnson transform.")
            if desc_parts:
                insights.append(Insight(
                    type="distribution", column=col,
                    title=f"Distribution insight for '{col}'",
                    description=" ".join(desc_parts),
                    severity=severity,
                    recommendation=" ".join(rec_parts) if rec_parts else "No action needed.",
                ))
        return insights

    def _correlation_insights(self) -> list[Insight]:
        insights: list[Insight] = []
        numeric_cols = self._numeric_cols()
        if len(numeric_cols) < 2:
            return insights
        corr = self.df[numeric_cols].corr(method="pearson")
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                r = corr.iloc[i, j]
                if abs(r) >= 0.7:
                    sev = "critical" if abs(r) >= 0.9 else "warning"
                    direction = "positive" if r > 0 else "negative"
                    insights.append(Insight(
                        type="correlation", column=f"{corr.columns[i]}, {corr.columns[j]}",
                        title=f"Strong {direction} correlation: {corr.columns[i]} vs {corr.columns[j]}",
                        description=f"Pearson r = {r:.3f}.",
                        severity=sev,
                        recommendation="Check for multicollinearity. Consider PCA / VIF analysis.",
                    ))
                elif abs(r) >= 0.5:
                    direction = "positive" if r > 0 else "negative"
                    insights.append(Insight(
                        type="correlation", column=f"{corr.columns[i]}, {corr.columns[j]}",
                        title=f"Moderate {direction} correlation: {corr.columns[i]} vs {corr.columns[j]}",
                        description=f"Pearson r = {r:.3f}.", severity="info",
                        recommendation="Potentially useful feature interaction — explore further.",
                    ))
        return insights

    def _quality_insights(self) -> list[Insight]:
        insights: list[Insight] = []
        n_rows = self.profile["total_rows"]
        for col, pct in self._missing_series().items():
            if pct == 0:
                continue
            if pct > 50:
                sev, rec = "critical", f"Consider dropping '{col}' — over half missing."
            elif pct > 20:
                sev, rec = "warning", f"Impute '{col}' using mean/median/mode or model-based."
            elif pct > 5:
                sev, rec = "warning", f"Investigate missing pattern in '{col}'."
            else:
                sev, rec = "info", f"Small gap in '{col}' — simple imputation suffices."
            insights.append(Insight(
                type="quality", column=col,
                title=f"'{col}' has {pct:.1f}% missing values",
                description=f"{pct:.1f}% ({int(pct / 100 * n_rows):,} of {n_rows:,}) missing.",
                severity=sev, recommendation=rec,
            ))
        for col in self.df.columns:
            n_unique = self.df[col].nunique(dropna=True)
            if n_unique == 1:
                val = self.df[col].dropna().iloc[0]
                insights.append(Insight(
                    type="quality", column=col,
                    title=f"'{col}' is constant",
                    description=f"All non-null values are '{val}'. Zero information gain.",
                    severity="warning", recommendation=f"Drop '{col}' — zero variance.",
                ))
            elif n_unique == 0:
                insights.append(Insight(
                    type="quality", column=col,
                    title=f"'{col}' is entirely null",
                    description="Column contains no non-null values.",
                    severity="critical", recommendation=f"Drop '{col}' immediately.",
                ))
        dupes = self.profile["duplicate_rows"]
        if dupes > 0:
            pct = dupes / n_rows * 100 if n_rows > 0 else 0
            sev = "critical" if pct > 10 else "warning" if pct > 1 else "info"
            insights.append(Insight(
                type="quality", column=None,
                title=f"{dupes:,} duplicate rows ({pct:.1f}%)",
                description=f"{dupes:,} exact duplicate rows.", severity=sev,
                recommendation="Remove duplicates to avoid data leakage.",
            ))
        for col in self._numeric_cols():
            series = self.df[col].dropna()
            if len(series) == 0:
                continue
            cv = series.std() / series.mean() if series.mean() != 0 else float("inf")
            if cv < 0.01 and series.std() > 0:
                insights.append(Insight(
                    type="quality", column=col,
                    title=f"'{col}' has near-zero variance (CV={cv:.4f})",
                    description=f"Coefficient of variation is {cv:.4f}.", severity="info",
                    recommendation=f"Consider dropping '{col}' unless small variations carry signal.",
                ))
        return insights

    def _outlier_insights(self) -> list[Insight]:
        insights: list[Insight] = []
        for col in self._numeric_cols():
            series = self.df[col].dropna()
            if len(series) < 10:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (series < lo) | (series > hi)
            n_out = int(mask.sum())
            pct = n_out / len(series) * 100
            if n_out == 0:
                continue
            if pct > 10:
                sev, rec = "critical", f"Large fraction of '{col}' are outliers — investigate."
            elif pct > 3:
                sev, rec = "warning", f"Review outliers in '{col}'. Consider Winsorisation."
            else:
                sev, rec = "info", f"Few outliers in '{col}' — likely safe to keep."
            low_c = int((series < lo).sum())
            high_c = int((series > hi).sum())
            direction = ""
            if low_c > 0 and high_c > 0:
                direction = f" ({low_c} low, {high_c} high)"
            elif high_c > 0:
                direction = f" (all {high_c} are high)"
            elif low_c > 0:
                direction = f" (all {low_c} are low)"
            insights.append(Insight(
                type="outlier", column=col,
                title=f"{n_out} outliers in '{col}' ({pct:.1f}%)",
                description=f"IQR method: {n_out:,} outside [{lo:.2f}, {hi:.2f}]{direction}.",
                severity=sev, recommendation=rec,
            ))
        return insights

    # ------------------------------------------------------------------ #
    # Chart selection
    # ------------------------------------------------------------------ #

    def select_charts(self) -> list[str]:
        """Select chart types to generate based on data characteristics."""
        charts = ["plot_distributions", "plot_correlations", "plot_missing", "plot_boxplots"]
        if 2 <= len(self._numeric_cols()) <= 10:
            charts.append("plot_pairplot")
        if self._datetime_cols():
            charts.append("time_series_plots")
        if self.target_column and self.target_column in self.df.columns:
            charts.append("target_analysis_charts")
        logger.info("Selected charts: %s", charts)
        return charts

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #

    def generate_summary(self) -> str:
        """Generate a paragraph-length summary of the dataset."""
        p = self.profile
        n_rows, n_cols = p["total_rows"], p["total_columns"]
        parts = [f"Dataset: {n_rows:,} rows x {n_cols} columns ({len(self._numeric_cols())} numeric, {len(self._categorical_cols())} categorical)."]
        if p["missing_percent"] > 0:
            parts.append(f"{p['missing_percent']:.1f}% cells missing ({p['missing_cells']:,} values).")
        else:
            parts.append("No missing values.")
        if p["duplicate_rows"] > 0:
            parts.append(f"{p['duplicate_rows']:,} duplicate rows.")
        if len(self._numeric_cols()) >= 2:
            corr = self.df[self._numeric_cols()].corr().abs()
            vals = corr.values[np.triu_indices_from(corr.values, k=1)]
            strong = vals[vals >= 0.7]
            if len(strong) > 0:
                parts.append(f"{len(strong)} strong correlation(s), max r={strong.max():.2f}.")
            else:
                parts.append("No strong correlations (|r| >= 0.70).")
        skewed = []
        for col in self._numeric_cols():
            s = self.df[col].dropna()
            if len(s) >= 8 and abs(float(stats.skew(s))) > 1.5:
                skewed.append(col)
        if skewed:
            parts.append(f"{len(skewed)} skewed column(s): {', '.join(skewed)}.")
        parts.append(f"Memory: {p['memory_usage_mb']:.1f} MB.")
        return " ".join(parts)

    # ------------------------------------------------------------------ #
    # Action items
    # ------------------------------------------------------------------ #

    def get_action_items(self) -> list[str]:
        """Return a prioritised list of recommended actions."""
        actions: list[str] = []
        all_insights: list[Insight] = []
        all_insights.extend(self._quality_insights())
        all_insights.extend(self._outlier_insights())
        seen: set[str] = set()
        for ins in all_insights:
            key = f"{ins.column}|{ins.recommendation[:80]}"
            if key not in seen:
                seen.add(key)
                actions.append(f"{ins.recommendation}")
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        actions.sort(key=lambda a: severity_order.get(
            next((i.severity for i in all_insights if i.recommendation == a), "info"), 3
        ))
        return actions
