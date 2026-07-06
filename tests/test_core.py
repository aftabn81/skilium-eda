"""Tests for core.py — DataEngine load, clean, profile."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skilium_eda.core import DataEngine


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "id": range(100),
        "name": [f"item_{i}" for i in range(100)],
        "value": np.random.randn(100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "score": np.random.randint(0, 100, 100),
        "date": pd.date_range("2024-01-01", periods=100, freq="D"),
    })


@pytest.fixture
def messy_df() -> pd.DataFrame:
    df = pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 4.0, 5.0, 5.0],
        "cat": ["a", "b", "a", np.nan, "b", "b"],
        "const": [1, 1, 1, 1, 1, 1],
    })
    return df


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

class TestLoad:
    def test_load_csv(self, sample_df: pd.DataFrame) -> None:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.close()
            sample_df.to_csv(f.name, index=False)
            df = DataEngine.load(f.name)
            assert len(df) == 100
            assert len(df.columns) == 6
            Path(f.name).unlink()

    def test_load_csv_latin1_fallback(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as f:
            f.write(b"col\n\xe9\n")  # latin-1 encoded é
            f.close()
            df = DataEngine.load(f.name)
            assert len(df) == 1
            Path(f.name).unlink()

    def test_load_parquet(self, sample_df: pd.DataFrame) -> None:
        pytest.importorskip("pyarrow")
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            f.close()
            sample_df.to_parquet(f.name, index=False)
            df = DataEngine.load(f.name)
            assert len(df) == 100
            Path(f.name).unlink()

    def test_load_json(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump([{"a": 1, "b": 2}, {"a": 3, "b": 4}], f)
            f.close()
            df = DataEngine.load(f.name)
            assert len(df) == 2
            assert list(df.columns) == ["a", "b"]
            Path(f.name).unlink()

    def test_load_json_nested(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump([{"a": 1, "b": {"c": 2}}], f)
            f.close()
            df = DataEngine.load(f.name)
            assert "b_c" in df.columns
            Path(f.name).unlink()

    def test_load_jsonl(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            f.write('{"a": 1}\n{"a": 2}\n')
            f.close()
            df = DataEngine.load(f.name)
            assert len(df) == 2
            Path(f.name).unlink()

    def test_load_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            DataEngine.load("/nonexistent/file.csv")

    def test_load_unsupported_extension(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.close()
            with pytest.raises(ValueError, match="Unsupported"):
                DataEngine.load(f.name)
            Path(f.name).unlink()

    def test_load_s3_url_no_extension_error(self) -> None:
        with pytest.raises((ValueError, ImportError)):
            DataEngine.load("s3://bucket/key")


# --------------------------------------------------------------------------- #
# Type inference
# --------------------------------------------------------------------------- #

class TestInferTypes:
    def test_infer_numeric(self, sample_df: pd.DataFrame) -> None:
        types = DataEngine.infer_types(sample_df)
        assert types["value"] == "numeric"
        assert types["score"] == "numeric"

    def test_infer_categorical(self, sample_df: pd.DataFrame) -> None:
        types = DataEngine.infer_types(sample_df)
        assert types["category"] == "categorical"

    def test_infer_datetime(self, sample_df: pd.DataFrame) -> None:
        types = DataEngine.infer_types(sample_df)
        assert types["date"] == "datetime"

    def test_infer_id(self) -> None:
        df = pd.DataFrame({
            "id_col": list(range(100)),  # sequential integers
            "group": ["A"] * 100,
        })
        types = DataEngine.infer_types(df)
        assert types["id_col"] == "id"

    def test_infer_text(self) -> None:
        df = pd.DataFrame({
            "short": ["a", "b", "c"],
            "long": ["x" * 100 for _ in range(3)],
        })
        types = DataEngine.infer_types(df)
        assert types["long"] == "text"

    def test_infer_datetime_from_name(self) -> None:
        df = pd.DataFrame({"created_at": ["2024-01-01", "2024-01-02", "2024-01-03"]})
        types = DataEngine.infer_types(df)
        assert types["created_at"] == "datetime"


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #

class TestGetSummary:
    def test_summary_shape(self, sample_df: pd.DataFrame) -> None:
        s = DataEngine.get_summary(sample_df)
        assert s["rows"] == 100
        assert s["columns"] == 6
        assert "memory_mb" in s
        assert "column_types" in s

    def test_summary_missing(self, messy_df: pd.DataFrame) -> None:
        s = DataEngine.get_summary(messy_df)
        assert s["missing_cells"] == 2
        assert s["missing_percent"] > 0

    def test_summary_no_missing(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        s = DataEngine.get_summary(df)
        assert s["missing_cells"] == 0
        assert s["missing_percent"] == 0.0


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #

class TestClean:
    def test_clean_basic(self, messy_df: pd.DataFrame) -> None:
        cleaned, log = DataEngine.clean(messy_df)
        assert cleaned.isnull().sum().sum() == 0
        assert len(cleaned) <= len(messy_df)
        assert isinstance(log, dict)
        assert "actions" in log

    def test_clean_drop_strategy(self, messy_df: pd.DataFrame) -> None:
        cleaned, _ = DataEngine.clean(messy_df, strategy="drop")
        assert cleaned.isnull().sum().sum() == 0

    def test_clean_mean_strategy(self, messy_df: pd.DataFrame) -> None:
        cleaned, _ = DataEngine.clean(messy_df, strategy="mean")
        assert cleaned["num"].isnull().sum() == 0

    def test_clean_median_strategy(self, messy_df: pd.DataFrame) -> None:
        cleaned, _ = DataEngine.clean(messy_df, strategy="median")
        assert cleaned["num"].isnull().sum() == 0

    def test_clean_mode_strategy(self, messy_df: pd.DataFrame) -> None:
        cleaned, _ = DataEngine.clean(messy_df, strategy="mode")
        assert cleaned.isnull().sum().sum() == 0

    def test_clean_ffill_strategy(self, messy_df: pd.DataFrame) -> None:
        cleaned, _ = DataEngine.clean(messy_df, strategy="ffill")
        assert cleaned.isnull().sum().sum() == 0

    def test_clean_invalid_strategy(self, messy_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            DataEngine.clean(messy_df, strategy="invalid")

    def test_clean_type_conversion(self) -> None:
        df = pd.DataFrame({"nums": ["1", "2", "3"], "dates": ["2024-01-01", "2024-01-02", "2024-01-03"]})
        cleaned, _ = DataEngine.clean(df)
        assert pd.api.types.is_numeric_dtype(cleaned["nums"])
        assert pd.api.types.is_datetime64_any_dtype(cleaned["dates"])

    def test_clean_deduplication(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        cleaned, _ = DataEngine.clean(df)
        assert len(cleaned) == 2

    def test_clean_preserves_original(self, messy_df: pd.DataFrame) -> None:
        original_len = len(messy_df)
        original_nulls = messy_df.isnull().sum().sum()
        DataEngine.clean(messy_df)
        assert len(messy_df) == original_len
        assert messy_df.isnull().sum().sum() == original_nulls


# --------------------------------------------------------------------------- #
# Outlier removal
# --------------------------------------------------------------------------- #

class TestRemoveOutliers:
    def test_remove_outliers_iqr(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
        cleaned = DataEngine.remove_outliers(df, method="iqr")
        assert len(cleaned) < len(df)
        assert 100 not in cleaned["x"].values

    def test_remove_outliers_zscore(self) -> None:
        np.random.seed(42)
        df = pd.DataFrame({"x": list(np.random.randn(50)) + [50.0]})
        cleaned = DataEngine.remove_outliers(df, method="zscore")
        assert 50.0 not in cleaned["x"].values

    def test_remove_outliers_invalid_method(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="Unknown outlier method"):
            DataEngine.remove_outliers(df, method="invalid")

    def test_remove_outliers_no_numeric(self) -> None:
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        cleaned = DataEngine.remove_outliers(df)
        assert len(cleaned) == len(df)

    def test_remove_outliers_all_same(self) -> None:
        df = pd.DataFrame({"x": [5, 5, 5, 5, 5]})
        cleaned = DataEngine.remove_outliers(df, method="iqr")
        assert len(cleaned) == len(df)


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #

class TestProfile:
    def test_profile_structure(self, sample_df: pd.DataFrame) -> None:
        p = DataEngine.profile(sample_df)
        assert "dataset" in p
        assert "columns" in p
        assert "correlations" in p
        assert "missing" in p
        assert "distributions" in p
        assert "quality" in p

    def test_profile_dataset_info(self, sample_df: pd.DataFrame) -> None:
        p = DataEngine.profile(sample_df)
        assert p["dataset"]["rows"] == 100
        assert p["dataset"]["columns"] == 6

    def test_profile_column_stats(self, sample_df: pd.DataFrame) -> None:
        p = DataEngine.profile(sample_df)
        assert "value" in p["columns"]
        assert p["columns"]["value"]["type"] == "numeric"
        assert "mean" in p["columns"]["value"]

    def test_profile_datetime_stats(self, sample_df: pd.DataFrame) -> None:
        p = DataEngine.profile(sample_df)
        assert p["columns"]["date"]["type"] == "datetime"
        assert "range_days" in p["columns"]["date"]

    def test_profile_categorical_stats(self, sample_df: pd.DataFrame) -> None:
        p = DataEngine.profile(sample_df)
        assert p["columns"]["category"]["type"] == "categorical"
        assert "top_values" in p["columns"]["category"]

    def test_profile_correlations(self, sample_df: pd.DataFrame) -> None:
        p = DataEngine.profile(sample_df)
        assert "pearson" in p["correlations"]
        assert "spearman" in p["correlations"]

    def test_profile_quality(self, sample_df: pd.DataFrame) -> None:
        p = DataEngine.profile(sample_df)
        assert p["quality"]["total_rows"] == 100
        assert p["quality"]["constant_columns"] == []

    def test_profile_quality_finds_constant(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
        p = DataEngine.profile(df)
        assert "a" in p["quality"]["constant_columns"]

    def test_profile_distributions(self, sample_df: pd.DataFrame) -> None:
        p = DataEngine.profile(sample_df)
        assert "value" in p["distributions"]
        assert p["distributions"]["value"] in ["normal", "skewed", "bimodal", "uniform", "unknown"]

    def test_get_column_stats_invalid_column(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError):
            DataEngine.get_column_stats(sample_df, "nonexistent")


# --------------------------------------------------------------------------- #
# Correlations
# --------------------------------------------------------------------------- #

class TestCorrelations:
    def test_pearson(self, sample_df: pd.DataFrame) -> None:
        corr = DataEngine.get_correlations(sample_df, "pearson")
        assert not corr.empty

    def test_spearman(self, sample_df: pd.DataFrame) -> None:
        corr = DataEngine.get_correlations(sample_df, "spearman")
        assert not corr.empty

    def test_invalid_method(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="method must be"):
            DataEngine.get_correlations(sample_df, "invalid")

    def test_insufficient_columns(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        corr = DataEngine.get_correlations(df)
        assert corr.empty


# --------------------------------------------------------------------------- #
# Quality report
# --------------------------------------------------------------------------- #

class TestQualityReport:
    def test_quality_basic(self, sample_df: pd.DataFrame) -> None:
        q = DataEngine.get_quality_report(sample_df)
        assert q["total_rows"] == 100
        assert q["total_columns"] == 6
        assert "memory_mb" in q

    def test_quality_high_cardinality(self) -> None:
        df = pd.DataFrame({"a": range(200), "b": ["x"] * 200})
        q = DataEngine.get_quality_report(df)
        assert "a" in q["high_cardinality_columns"]
