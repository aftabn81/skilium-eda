"""Tests for agent.py — EDAAgent insights and decisions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skilium_eda.agent import EDAAgent, Insight


@pytest.fixture
def sample_df() -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "num_a": np.random.randn(100),
        "num_b": np.random.randn(100) * 2 + 5,
        "num_c": np.random.exponential(2, 100),
        "cat": np.random.choice(["X", "Y", "Z"], 100),
        "dt": pd.date_range("2024-01-01", periods=100, freq="D"),
    })


@pytest.fixture
def skewed_df() -> pd.DataFrame:
    return pd.DataFrame({
        "normal": np.random.randn(100),
        "skewed": np.random.exponential(2, 100),
        "constant": [42] * 100,
    })


@pytest.fixture
def correlated_df() -> pd.DataFrame:
    x = np.random.randn(100)
    return pd.DataFrame({
        "a": x,
        "b": x * 2 + np.random.randn(100) * 0.1,
        "c": np.random.randn(100),
    })


@pytest.fixture
def missing_df() -> pd.DataFrame:
    df = pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0, 5.0] * 20,
        "b": ["x", np.nan, "y", "z", "x"] * 20,
    })
    return df


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #

class TestConstruction:
    def test_init(self, sample_df: pd.DataFrame) -> None:
        agent = EDAAgent(sample_df)
        assert agent.profile["total_rows"] == 100
        assert agent.profile["total_columns"] == 5

    def test_init_with_profile(self, sample_df: pd.DataFrame) -> None:
        profile = {"total_rows": 100, "total_columns": 5, "numeric_cols": ["num_a"],
                   "categorical_cols": [], "datetime_cols": [], "missing_cells": 0,
                   "missing_percent": 0.0, "duplicate_rows": 0, "memory_usage_mb": 1.0,
                   "column_types": {}}
        agent = EDAAgent(sample_df, profile=profile)
        assert agent.profile["total_rows"] == 100

    def test_init_with_target(self, sample_df: pd.DataFrame) -> None:
        agent = EDAAgent(sample_df, target_column="num_a")
        assert agent.target_column == "num_a"

    def test_init_invalid_type(self) -> None:
        with pytest.raises(TypeError):
            EDAAgent("not a dataframe")


# --------------------------------------------------------------------------- #
# decide_analyses
# --------------------------------------------------------------------------- #

class TestDecideAnalyses:
    def test_always_profile(self, sample_df: pd.DataFrame) -> None:
        agent = EDAAgent(sample_df)
        analyses = agent.decide_analyses()
        assert "profile" in analyses

    def test_includes_distribution(self, sample_df: pd.DataFrame) -> None:
        agent = EDAAgent(sample_df)
        analyses = agent.decide_analyses()
        assert "distribution" in analyses

    def test_correlation_with_numeric(self, sample_df: pd.DataFrame) -> None:
        agent = EDAAgent(sample_df)
        analyses = agent.decide_analyses()
        assert "correlation" in analyses

    def test_time_series_with_datetime(self, sample_df: pd.DataFrame) -> None:
        agent = EDAAgent(sample_df)
        analyses = agent.decide_analyses()
        assert "time_series" in analyses

    def test_missing_when_data_missing(self, missing_df: pd.DataFrame) -> None:
        agent = EDAAgent(missing_df)
        analyses = agent.decide_analyses()
        assert "missing" in analyses

    def test_target_analysis(self, sample_df: pd.DataFrame) -> None:
        agent = EDAAgent(sample_df, target_column="num_a")
        analyses = agent.decide_analyses()
        assert "target_analysis" in analyses

    def test_no_target_analysis_when_invalid(self, sample_df: pd.DataFrame) -> None:
        agent = EDAAgent(sample_df, target_column="nonexistent")
        analyses = agent.decide_analyses()
        assert "target_analysis" not in analyses

    def test_sampling_for_large_dataset(self) -> None:
        df = pd.DataFrame(np.random.randn(2000, 600))
        agent = EDAAgent(df)
        analyses = agent.decide_analyses()
        assert "sampling" in analyses


# --------------------------------------------------------------------------- #
# generate_insights
# --------------------------------------------------------------------------- #

class TestGenerateInsights:
    def test_insights_not_empty(self, sample_df: pd.DataFrame) -> None:
        agent = EDAAgent(sample_df)
        insights = agent.generate_insights()
        assert len(insights) > 0

    def test_insights_are_insight_objects(self, sample_df: pd.DataFrame) -> None:
        agent = EDAAgent(sample_df)
        insights = agent.generate_insights()
        for ins in insights:
            assert isinstance(ins, Insight)

    def test_insight_fields(self, sample_df: pd.DataFrame) -> None:
        agent = EDAAgent(sample_df)
        insights = agent.generate_insights()
        for ins in insights:
            assert ins.type in {"distribution", "correlation", "quality", "outlier", "recommendation"}
            assert ins.severity in {"info", "warning", "critical"}
            assert ins.title
            assert ins.description

    def test_correlation_insights(self, correlated_df: pd.DataFrame) -> None:
        agent = EDAAgent(correlated_df)
        insights = agent.generate_insights()
        corr_insights = [i for i in insights if i.type == "correlation"]
        assert len(corr_insights) > 0

    def test_distribution_insights_skewed(self, skewed_df: pd.DataFrame) -> None:
        agent = EDAAgent(skewed_df)
        insights = agent.generate_insights()
        dist_insights = [i for i in insights if i.type == "distribution"]
        assert len(dist_insights) > 0

    def test_quality_insights_constant(self, skewed_df: pd.DataFrame) -> None:
        agent = EDAAgent(skewed_df)
        insights = agent.generate_insights()
        qual_insights = [i for i in insights if i.type == "quality"]
        constant = [i for i in qual_insights if "constant" in i.title.lower()]
        assert len(constant) > 0

    def test_outlier_insights(self, sample_df: pd.DataFrame) -> None:
        agent = EDAAgent(sample_df)
        insights = agent.generate_insights()
        outlier_insights = [i for i in insights if i.type == "outlier"]
        assert isinstance(outlier_insights, list)

    def test_sorted_by_severity(self, sample_df: pd.DataFrame) -> None:
        agent = EDAAgent(sample_df)
        insights = agent.generate_insights()
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        severities = [severity_order[i.severity] for i in insights]
        assert severities == sorted(severities)


# --------------------------------------------------------------------------- #
# select_charts
# --------------------------------------------------------------------------- #

class TestSelectCharts:
    def test_always_returns_distributions(self, sample_df: pd.DataFrame) -> None:
        agent = EDAAgent(sample_df)
        charts = agent.select_charts()
        assert "plot_distributions" in charts

    def test_pairplot_for_moderate_numeric(self, sample_df: pd.DataFrame) -> None:
        agent = EDAAgent(sample_df)
        charts = agent.select_charts()
        assert "plot_pairplot" in charts

    def test_no_pairplot_for_too_many_numeric(self) -> None:
        df = pd.DataFrame({f"col_{i}": np.random.randn(50) for i in range(15)})
        agent = EDAAgent(df)
        charts = agent.select_charts()
        assert "plot_pairplot" not in charts

    def test_time_series_with_datetime(self, sample_df: pd.DataFrame) -> None:
        agent = EDAAgent(sample_df)
        charts = agent.select_charts()
        assert "time_series_plots" in charts


# --------------------------------------------------------------------------- #
# generate_summary
# --------------------------------------------------------------------------- #

class TestGenerateSummary:
    def test_returns_string(self, sample_df: pd.DataFrame) -> None:
        agent = EDAAgent(sample_df)
        summary = agent.generate_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_contains_row_count(self, sample_df: pd.DataFrame) -> None:
        agent = EDAAgent(sample_df)
        summary = agent.generate_summary()
        assert "100" in summary

    def test_contains_memory(self, sample_df: pd.DataFrame) -> None:
        agent = EDAAgent(sample_df)
        summary = agent.generate_summary()
        assert "MB" in summary


# --------------------------------------------------------------------------- #
# get_action_items
# --------------------------------------------------------------------------- #

class TestGetActionItems:
    def test_returns_list(self, sample_df: pd.DataFrame) -> None:
        agent = EDAAgent(sample_df)
        actions = agent.get_action_items()
        assert isinstance(actions, list)

    def test_actions_are_strings(self, sample_df: pd.DataFrame) -> None:
        agent = EDAAgent(sample_df)
        actions = agent.get_action_items()
        for a in actions:
            assert isinstance(a, str)


# --------------------------------------------------------------------------- #
# Insight model
# --------------------------------------------------------------------------- #

class TestInsight:
    def test_valid_insight(self) -> None:
        ins = Insight(
            type="quality", column="test", title="Test", description="Desc",
            severity="warning", recommendation="Fix it",
        )
        assert ins.type == "quality"
        assert ins.severity == "warning"

    def test_invalid_type(self) -> None:
        with pytest.raises(Exception):
            Insight(type="invalid", title="Test", description="Desc", severity="info", recommendation="Fix")

    def test_invalid_severity(self) -> None:
        with pytest.raises(Exception):
            Insight(type="quality", title="Test", description="Desc", severity="invalid", recommendation="Fix")

    def test_none_column(self) -> None:
        ins = Insight(
            type="quality", column=None, title="Test", description="Desc",
            severity="info", recommendation="Fix it",
        )
        assert ins.column is None
