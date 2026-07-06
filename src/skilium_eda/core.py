"""Core module — data loading, cleaning, and profiling via a single DataEngine class."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: set[str] = {".csv", ".xlsx", ".xls", ".parquet", ".json", ".jsonl"}
CATEGORICAL_CARDINALITY: float = 0.05
ID_CARDINALITY: float = 0.95
IQR_MULTIPLIER: float = 1.5
NORMALTEST_ALPHA: float = 0.05
SKEW_THRESHOLD: float = 2.0
HIGH_CARD_RATIO: float = 0.80
HIGH_CARD_MIN_UNIQUE: int = 100
MISSING_STRINGS: set[str] = {"", "nan", "null", "none", "na", "n/a", "-", ".", "?", "missing", "unknown"}


def _open_s3(source: str):
    """Open an S3 (or other remote) path via fsspec."""
    import fsspec
    return fsspec.open(source)


class DataEngine:
    """Load, clean, and profile data. Supports local files and S3 URLs."""

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    @staticmethod
    def load(source: str) -> pd.DataFrame:
        """Load data from a local path or S3 URL (s3://bucket/path)."""
        if source.startswith("s3://"):
            return DataEngine._load_s3(source)

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported extension '{ext}'. Supported: {SUPPORTED_EXTENSIONS}")

        loaders: dict[str, Any] = {
            ".csv": lambda: DataEngine._load_csv(path),
            ".xlsx": pd.read_excel,
            ".xls": pd.read_excel,
            ".parquet": pd.read_parquet,
            ".json": lambda: DataEngine._load_json(path),
            ".jsonl": lambda: DataEngine._load_jsonl(path),
        }
        df = loaders[ext]()
        logger.info("Loaded %d rows x %d columns from %s", len(df), len(df.columns), source)
        return df

    @staticmethod
    def _load_s3(source: str) -> pd.DataFrame:
        """Load a file from S3 using fsspec."""
        ext = Path(source).suffix.lower()
        with _open_s3(source) as f:
            if ext == ".csv":
                return pd.read_csv(f)
            if ext in (".xlsx", ".xls"):
                return pd.read_excel(f)
            if ext == ".parquet":
                return pd.read_parquet(f)
            if ext == ".json":
                return pd.read_json(f)
            if ext == ".jsonl":
                return pd.read_json(f, lines=True)
            raise ValueError(f"Unsupported S3 file extension: {ext}")

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")

    @staticmethod
    def _load_json(path: Path) -> pd.DataFrame:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            if any(isinstance(v, (dict, list)) for row in data for v in row.values()):
                return pd.json_normalize(data, sep="_")
            return pd.DataFrame(data)
        if isinstance(data, dict):
            return pd.json_normalize(data, sep="_")
        return pd.DataFrame(data)

    @staticmethod
    def _load_jsonl(path: Path) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.json_normalize(records, sep="_") if records else pd.DataFrame()

    # ------------------------------------------------------------------ #
    # Type inference
    # ------------------------------------------------------------------ #

    @staticmethod
    def infer_types(df: pd.DataFrame) -> dict[str, str]:
        """Infer semantic types: numeric, categorical, datetime, text, id."""
        type_map: dict[str, str] = {}
        n_rows = len(df)
        for col in df.columns:
            series = df[col]
            dtype = series.dtype

            if pd.api.types.is_datetime64_any_dtype(dtype):
                type_map[col] = "datetime"
                continue

            non_null = series.dropna()
            n_unique = non_null.nunique()
            ratio = n_unique / n_rows if n_rows > 0 else 0.0

            # ID detection: nearly unique, sequential-looking integers
            if ratio >= ID_CARDINALITY and n_unique > 10:
                if pd.api.types.is_integer_dtype(dtype):
                    clean_int = non_null.astype(int)
                    spread = int(clean_int.max() - clean_int.min() + 1)
                    if spread == n_unique or abs(spread - n_unique) <= 2:
                        type_map[col] = "id"
                        continue
                    # High-cardinality non-sequential integer -> numeric
                    type_map[col] = "numeric"
                    continue
                elif pd.api.types.is_numeric_dtype(dtype):
                    # High-cardinality float columns are numeric, not ID
                    type_map[col] = "numeric"
                    continue
                # High-cardinality non-numeric columns -> text
                type_map[col] = "text"
                continue

            # Categorical-like integer detection: low cardinality integers
            if pd.api.types.is_integer_dtype(dtype) and n_unique <= 10:
                type_map[col] = "categorical"
                continue

            # Boolean -> categorical
            if pd.api.types.is_bool_dtype(dtype):
                type_map[col] = "categorical"
                continue

            if pd.api.types.is_numeric_dtype(dtype):
                type_map[col] = "numeric"
                continue

            # String/object type inference
            if ratio <= CATEGORICAL_CARDINALITY and n_unique <= 50:
                type_map[col] = "categorical"
                continue

            if DataEngine._looks_like_datetime(series):
                type_map[col] = "datetime"
                continue

            if dtype == object and non_null.astype(str).str.len().mean() > 50:
                type_map[col] = "text"
                continue

            type_map[col] = "categorical"
        return type_map

    @staticmethod
    def _detect_column_type(series: pd.Series) -> str:
        """Single-column type detection used as fallback."""
        dtype = series.dtype
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "datetime"
        if pd.api.types.is_bool_dtype(dtype):
            return "categorical"
        if pd.api.types.is_numeric_dtype(dtype):
            return "numeric"
        return "categorical"

    @staticmethod
    def _looks_like_datetime(series: pd.Series) -> bool:
        if series.dtype != object:
            return False
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False
        col_lower = str(series.name).lower()
        hints = ["date", "time", "dt", "timestamp", "created", "updated", "day", "month", "year"]
        has_hint = any(h in col_lower for h in hints)
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            rate = parsed.notna().sum() / len(sample)
            return rate >= (0.7 if has_hint else 0.9)
        except (ValueError, TypeError):
            return False

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_summary(df: pd.DataFrame) -> dict[str, Any]:
        """Return a concise dataset summary."""
        rows, cols = df.shape
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        missing_cells = df.isna().sum().sum()
        missing_pct = (missing_cells / (rows * cols) * 100) if rows * cols > 0 else 0.0
        return {
            "rows": rows,
            "columns": cols,
            "memory_mb": round(memory_mb, 2),
            "column_types": DataEngine.infer_types(df),
            "missing_cells": int(missing_cells),
            "missing_percent": round(missing_pct, 2),
            "duplicate_rows": int(df.duplicated().sum()),
        }

    # ------------------------------------------------------------------ #
    # Cleaning
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_missing_strings(df: pd.DataFrame) -> pd.DataFrame:
        """Convert common string representations of missing values to actual NaN."""
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].replace(
                {s: np.nan for s in MISSING_STRINGS}, regex=False
            )
        return df

    @staticmethod
    def clean(df: pd.DataFrame, strategy: str = "auto") -> tuple[pd.DataFrame, dict]:
        """Full cleaning pipeline. Returns (cleaned_df, cleaning_log)."""
        df = df.copy()

        # Normalize string missing-values BEFORE recording state
        df = DataEngine._normalize_missing_strings(df)

        # Record pre-cleaning state
        cleaning_log = {
            "original_missing": {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()},
            "original_missing_total": int(df.isna().sum().sum()),
            "original_duplicates": int(df.duplicated().sum()),
            "actions": [],
        }

        # Handle missing values FIRST (before type conversion) so that
        # NaN values are detected correctly before object→category conversion
        df, actions = DataEngine._handle_missing(df, strategy)
        cleaning_log["actions"].extend(actions)

        # Then convert types (numeric, datetime, categorical)
        df = DataEngine._convert_types(df)

        n_dups = int(df.duplicated().sum())
        if n_dups > 0:
            df = df.drop_duplicates().reset_index(drop=True)
            cleaning_log["actions"].append(f"Removed {n_dups} duplicate rows")

        logger.info("Cleaned: %d rows x %d columns", len(df), len(df.columns))
        return df, cleaning_log

    @staticmethod
    def _convert_types(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].dtype != object:
                continue
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
                continue
            except (ValueError, TypeError):
                pass
            # Only convert to datetime if column name or values look like dates
            if DataEngine._looks_like_datetime(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col], errors="raise")
                    continue
                except (ValueError, TypeError):
                    pass
            n_unique = df[col].nunique(dropna=True)
            n_rows = len(df)
            ratio = n_unique / n_rows if n_rows > 0 else 0.0
            if n_unique <= 50 and ratio <= CATEGORICAL_CARDINALITY:
                df[col] = df[col].astype("category")
        return df

    @staticmethod
    def _handle_missing(df: pd.DataFrame, strategy: str) -> tuple[pd.DataFrame, list[str]]:
        actions: list[str] = []
        if strategy not in {"auto", "drop", "mean", "median", "mode", "ffill"}:
            raise ValueError(f"Unknown strategy '{strategy}'")
        if strategy == "drop":
            n_before = len(df)
            df = df.dropna().reset_index(drop=True)
            actions.append(f"Dropped {n_before - len(df)} rows with missing values")
            return df, actions
        if strategy == "ffill":
            df = df.ffill().bfill()
            actions.append("Forward-filled missing values")
            return df, actions
        if strategy in ("mean", "median"):
            return DataEngine._fill_numeric(df, strategy)
        if strategy == "mode":
            return DataEngine._fill_mode(df)
        return DataEngine._auto_impute(df)

    @staticmethod
    def _auto_impute(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Auto-impute missing values with smart per-type rules.

        Rules:
        - Numeric (int/float): fill with median
        - Datetime: forward-fill then backward-fill
        - Categorical with >30% missing: fill with "Unknown"
        - Categorical with <=30% missing: fill with mode
        - All-null columns: flagged, left as-is (no invented values)
        """
        actions: list[str] = []
        n_rows = len(df)
        HIGH_MISSING_THRESHOLD = 0.30  # 30%

        for col in df.columns:
            n_missing = int(df[col].isna().sum())
            if n_missing == 0:
                continue

            # Flag all-null columns — do not silently invent values
            if df[col].isna().all():
                actions.append(f"'{col}': column is entirely null — no values to infer from")
                continue

            missing_ratio = n_missing / n_rows if n_rows > 0 else 0.0

            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].ffill().bfill()
                actions.append(f"'{col}': forward-filled {n_missing} datetime values")
            elif pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                actions.append(f"'{col}': filled {n_missing} missing with median ({median_val:.2f})")
            elif missing_ratio > HIGH_MISSING_THRESHOLD:
                # High-missingness categorical: use "Unknown" instead of mode
                df[col] = df[col].fillna("Unknown")
                actions.append(f"'{col}': filled {n_missing} missing ({missing_ratio:.0%}) with 'Unknown'")
            else:
                # Low-missingness categorical: use mode
                mode_val = df[col].mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
                df[col] = df[col].fillna(fill_val)
                actions.append(f"'{col}': filled {n_missing} missing with mode ({fill_val})")
        return df, actions

    @staticmethod
    def _fill_numeric(df: pd.DataFrame, method: str) -> tuple[pd.DataFrame, list[str]]:
        actions: list[str] = []
        for col in df.select_dtypes(include=[np.number]).columns:
            n_missing = int(df[col].isna().sum())
            if n_missing == 0:
                continue
            fill = df[col].mean() if method == "mean" else df[col].median()
            df[col] = df[col].fillna(fill)
            actions.append(f"'{col}': filled {n_missing} missing with {method} ({fill:.2f})")
        return df, actions

    @staticmethod
    def _fill_mode(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Fill missing values with mode, using 'Unknown' for high-missingness columns."""
        actions: list[str] = []
        n_rows = len(df)
        HIGH_MISSING_THRESHOLD = 0.30

        for col in df.columns:
            n_missing = int(df[col].isna().sum())
            if n_missing == 0 or df[col].isna().all():
                continue
            missing_ratio = n_missing / n_rows if n_rows > 0 else 0.0

            if missing_ratio > HIGH_MISSING_THRESHOLD:
                df[col] = df[col].fillna("Unknown")
                actions.append(f"'{col}': filled {n_missing} missing ({missing_ratio:.0%}) with 'Unknown'")
            else:
                modes = df[col].mode()
                fill_val = modes.iloc[0] if not modes.empty else "Unknown"
                df[col] = df[col].fillna(fill_val)
                actions.append(f"'{col}': filled {n_missing} missing with '{fill_val}'")
        return df, actions

    @staticmethod
    def remove_outliers(df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
        """Remove outlier rows using IQR (default), zscore, or isolation forest."""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return df.copy()
        mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        if method == "iqr":
            for col in numeric_df.columns:
                q1, q3 = numeric_df[col].quantile(0.25), numeric_df[col].quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
                mask[col] = (numeric_df[col] < lo) | (numeric_df[col] > hi)
        elif method == "zscore":
            for col in numeric_df.columns:
                mean, std = numeric_df[col].mean(), numeric_df[col].std()
                if std == 0:
                    continue
                mask[col] = np.abs((numeric_df[col] - mean) / std) > 3.0
        elif method == "isolation":
            from sklearn.ensemble import IsolationForest
            fill_vals = numeric_df.median()
            filled = numeric_df.fillna(fill_vals)
            preds = IsolationForest(contamination=0.05, random_state=42, n_estimators=100).fit_predict(filled)
            for col in numeric_df.columns:
                mask[col] = preds == -1
        else:
            raise ValueError(f"Unknown outlier method '{method}'")
        outlier_rows = mask.any(axis=1)
        return df[~outlier_rows].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Profiling
    # ------------------------------------------------------------------ #

    @staticmethod
    def profile(df: pd.DataFrame, type_map: dict[str, str] | None = None) -> dict[str, Any]:
        """Generate a comprehensive profile of the dataset."""
        if type_map is None:
            type_map = DataEngine.infer_types(df)

        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if type_map.get(c) not in ("id", "categorical")]

        return {
            "dataset": DataEngine._dataset_info(df),
            "columns": {col: DataEngine.get_column_stats(df, col, type_map.get(col)) for col in df.columns},
            "correlations": {
                "pearson": DataEngine.get_correlations(df, "pearson", numeric_cols).to_dict(),
                "spearman": DataEngine.get_correlations(df, "spearman", numeric_cols).to_dict(),
            },
            "missing": DataEngine._missing_summary(df),
            "distributions": DataEngine.get_distributions(df, type_map),
            "quality": DataEngine.get_quality_report(df),
        }

    @staticmethod
    def _dataset_info(df: pd.DataFrame) -> dict[str, Any]:
        rows, cols = df.shape
        return {
            "rows": rows,
            "columns": cols,
            "missing_cells": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_size": int(df.memory_usage(deep=True).sum()),
        }

    @staticmethod
    def get_column_stats(df: pd.DataFrame, col: str, hinted_type: str | None = None) -> dict[str, Any]:
        """Return detailed statistics for a single column."""
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found")
        series = df[col]
        base = {
            "column": col,
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "null_percent": round(series.isna().mean() * 100, 2),
            "count": int(series.count()),
        }

        effective_type = hinted_type or DataEngine._detect_column_type(series)

        if effective_type == "id":
            return {
                **base,
                "type": "id",
                "unique_count": int(series.nunique(dropna=True)),
                "note": "Identifier column -- excluded from statistical analysis",
            }

        if pd.api.types.is_numeric_dtype(series.dtype) and effective_type == "numeric":
            clean = series.dropna()
            if len(clean) == 0:
                return {**base, "error": "All values are null"}
            return {
                **base, "type": "numeric",
                "mean": float(clean.mean()), "std": float(clean.std()),
                "min": float(clean.min()), "max": float(clean.max()),
                "q25": float(clean.quantile(0.25)), "q50": float(clean.quantile(0.50)),
                "q75": float(clean.quantile(0.75)),
                "skewness": float(sp_stats.skew(clean)),
                "kurtosis": float(sp_stats.kurtosis(clean)),
            }

        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            clean = series.dropna()
            if len(clean) == 0:
                return {**base, "error": "All values are null"}
            monthly = clean.dt.to_period("M").value_counts().sort_index()
            return {
                **base, "type": "datetime",
                "min": str(clean.min()), "max": str(clean.max()),
                "range_days": int((clean.max() - clean.min()).days),
                "seasonal_counts": {str(k): int(v) for k, v in monthly.head(12).to_dict().items()},
            }

        vc = series.value_counts(dropna=True)
        return {
            **base, "type": "categorical",
            "unique_count": int(series.nunique(dropna=True)),
            "most_frequent": vc.index[0] if not vc.empty else None,
            "most_frequent_count": int(vc.iloc[0]) if not vc.empty else 0,
            "top_values": vc.head(10).to_dict(),
        }

    @staticmethod
    def get_correlations(df: pd.DataFrame, method: str = "pearson", cols: list[str] | None = None) -> pd.DataFrame:
        """Compute correlation matrix for numeric columns."""
        if method not in {"pearson", "spearman"}:
            raise ValueError("method must be 'pearson' or 'spearman'")
        numeric = df[cols] if cols else df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            return pd.DataFrame()
        return numeric.corr(method=method)

    @staticmethod
    def get_quality_report(df: pd.DataFrame) -> dict[str, Any]:
        """Assess overall data quality."""
        rows, cols = df.shape
        missing_cells = int(df.isna().sum().sum())
        missing_pct = round((missing_cells / (rows * cols) * 100) if rows * cols > 0 else 0.0, 2)
        constant = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
        high_card = []
        for c in df.columns:
            n_unique = df[c].nunique(dropna=True)
            ratio = n_unique / rows if rows > 0 else 0.0
            if n_unique >= HIGH_CARD_MIN_UNIQUE and ratio > HIGH_CARD_RATIO:
                high_card.append(c)
        return {
            "total_rows": rows, "total_columns": cols,
            "missing_cells": missing_cells, "missing_percent": missing_pct,
            "duplicate_rows": int(df.duplicated().sum()),
            "constant_columns": constant,
            "high_cardinality_columns": high_card,
            "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        }

    @staticmethod
    def get_distributions(df: pd.DataFrame, type_map: dict[str, str] | None = None) -> dict[str, str]:
        """Guess distribution type for each numeric column."""
        result: dict[str, str] = {}
        numeric = df.select_dtypes(include=[np.number])
        for col in numeric.columns:
            if type_map and type_map.get(col) in ("id", "categorical"):
                continue
            clean = df[col].dropna()
            if len(clean) < 8:
                continue
            try:
                _, p = sp_stats.normaltest(clean)
                if p >= NORMALTEST_ALPHA:
                    result[col] = "normal"
                    continue
            except Exception:
                pass
            skew = sp_stats.skew(clean)
            if abs(skew) > SKEW_THRESHOLD:
                result[col] = "skewed"
                continue
            try:
                counts, _ = np.histogram(clean, bins="auto")
                if DataEngine._count_peaks(counts) >= 2:
                    result[col] = "bimodal"
                    continue
            except Exception:
                pass
            try:
                counts, _ = np.histogram(clean, bins="auto")
                probs = counts / counts.sum()
                entropy = -np.sum(probs * np.log2(probs + 1e-12))
                n_bins = len(counts)
                max_ent = np.log2(n_bins) if n_bins > 1 else 1.0
                if max_ent > 0 and (entropy / max_ent) > 0.9:
                    result[col] = "uniform"
                    continue
            except Exception:
                pass
            result[col] = "unknown"
        return result

    @staticmethod
    def _count_peaks(counts: np.ndarray) -> int:
        if len(counts) < 3:
            return 0
        peaks = 0
        for i in range(len(counts)):
            left = counts[i - 1] if i > 0 else -1
            right = counts[i + 1] if i < len(counts) - 1 else -1
            if counts[i] > left and counts[i] > right:
                peaks += 1
        return peaks

    @staticmethod
    def _missing_summary(df: pd.DataFrame) -> dict[str, Any]:
        missing = df.isna().sum()
        return {
            "total_missing": int(missing.sum()),
            "missing_per_column": missing.to_dict(),
            "columns_with_missing": missing[missing > 0].to_dict(),
        }
