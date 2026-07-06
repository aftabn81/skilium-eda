"""Tests for pipeline.py — EDAPipeline orchestration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skilium_eda.pipeline import EDAPipeline


@pytest.fixture
def csv_file() -> str:
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        f.close()
        df = pd.DataFrame({
            "a": np.random.randn(50),
            "b": np.random.randn(50) * 2,
            "c": np.random.choice(["x", "y"], 50),
        })
        df.to_csv(f.name, index=False)
        return f.name


@pytest.fixture
def output_dir() -> str:
    d = tempfile.mkdtemp()
    return d


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #

class TestConstruction:
    def test_init(self, csv_file: str, output_dir: str) -> None:
        p = EDAPipeline(csv_file, output_dir=output_dir)
        assert p.source == csv_file
        assert p.output_dir == Path(output_dir)

    def test_default_config(self, csv_file: str, output_dir: str) -> None:
        p = EDAPipeline(csv_file, output_dir=output_dir)
        assert p.config == {}

    def test_custom_config(self, csv_file: str, output_dir: str) -> None:
        p = EDAPipeline(csv_file, output_dir=output_dir, config={"missing_strategy": "mean"})
        assert p.config["missing_strategy"] == "mean"


# --------------------------------------------------------------------------- #
# run_step
# --------------------------------------------------------------------------- #

class TestRunStep:
    def test_step_load(self, csv_file: str, output_dir: str) -> None:
        p = EDAPipeline(csv_file, output_dir=output_dir)
        df = p.run_step("load")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50

    def test_step_clean(self, csv_file: str, output_dir: str) -> None:
        p = EDAPipeline(csv_file, output_dir=output_dir)
        p.run_step("load")
        df = p.run_step("clean")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_step_profile(self, csv_file: str, output_dir: str) -> None:
        p = EDAPipeline(csv_file, output_dir=output_dir)
        p.run_step("load")
        p.run_step("clean")
        profile = p.run_step("profile")
        assert isinstance(profile, dict)
        assert "dataset" in profile
        assert "columns" in profile

    def test_step_insights(self, csv_file: str, output_dir: str) -> None:
        p = EDAPipeline(csv_file, output_dir=output_dir)
        p.run_step("load")
        p.run_step("clean")
        p.run_step("profile")
        insights = p.run_step("insights")
        assert isinstance(insights, list)

    def test_step_visualize(self, csv_file: str, output_dir: str) -> None:
        p = EDAPipeline(csv_file, output_dir=output_dir)
        p.run_step("load")
        p.run_step("clean")
        charts = p.run_step("visualize")
        assert isinstance(charts, list)

    def test_step_report(self, csv_file: str, output_dir: str) -> None:
        p = EDAPipeline(csv_file, output_dir=output_dir,
                        config={"report_formats": ["html", "markdown"]})
        p.run_step("load")
        p.run_step("clean")
        p.run_step("profile")
        p.run_step("insights")
        p.run_step("visualize")
        reports = p.run_step("report")
        assert isinstance(reports, dict)

    def test_unknown_step(self, csv_file: str, output_dir: str) -> None:
        p = EDAPipeline(csv_file, output_dir=output_dir)
        with pytest.raises(ValueError, match="Unknown step"):
            p.run_step("nonexistent")

    def test_step_clean_before_load(self, csv_file: str, output_dir: str) -> None:
        p = EDAPipeline(csv_file, output_dir=output_dir)
        with pytest.raises(RuntimeError, match="Load data first"):
            p.run_step("clean")


# --------------------------------------------------------------------------- #
# run (full pipeline)
# --------------------------------------------------------------------------- #

class TestRun:
    def test_full_run(self, csv_file: str, output_dir: str) -> None:
        p = EDAPipeline(csv_file, output_dir=output_dir,
                        config={"report_formats": ["markdown"]})
        results = p.run()
        assert results["source"] == csv_file
        assert len(results["shape"]) == 2
        assert results["shape"][0] > 0
        assert results["insights_count"] >= 0
        assert isinstance(results["charts"], list)
        assert isinstance(results["reports"], dict)
        assert results["duration_seconds"] >= 0
        assert isinstance(results["steps_completed"], list)
        assert "load" in results["steps_completed"]
        assert isinstance(results["errors"], list)

    def test_run_creates_output_dir(self, csv_file: str, output_dir: str) -> None:
        out = str(Path(output_dir) / "nested" / "output")
        p = EDAPipeline(csv_file, output_dir=out,
                        config={"report_formats": ["markdown"]})
        p.run()
        assert Path(out).exists()

    def test_run_with_html_report(self, csv_file: str, output_dir: str) -> None:
        p = EDAPipeline(csv_file, output_dir=output_dir,
                        config={"report_formats": ["html"]})
        results = p.run()
        assert "html" in results["reports"]
        assert Path(results["reports"]["html"]).exists()

    def test_run_with_markdown_report(self, csv_file: str, output_dir: str) -> None:
        p = EDAPipeline(csv_file, output_dir=output_dir,
                        config={"report_formats": ["markdown"]})
        results = p.run()
        assert "markdown" in results["reports"]
        assert Path(results["reports"]["markdown"]).exists()

    def test_run_with_both_reports(self, csv_file: str, output_dir: str) -> None:
        p = EDAPipeline(csv_file, output_dir=output_dir,
                        config={"report_formats": ["html", "markdown"]})
        results = p.run()
        assert "html" in results["reports"]
        assert "markdown" in results["reports"]
