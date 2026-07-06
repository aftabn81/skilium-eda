"""Report generation — HTML and Markdown output with embedded charts."""

from __future__ import annotations

import base64
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Skilium EDA — {{ dataset_name }}</title>
<style>
:root { --bg: #f8f9fa; --fg: #212529; --card-bg: #ffffff; --accent: #0d6efd; --muted: #6c757d; --accent2: #0d9488; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
       background: var(--bg); color: var(--fg); line-height: 1.6; }
header { background: linear-gradient(135deg, #0d6efd 0%, #0a58ca 100%); color: white;
         padding: 2rem 1rem; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.12); }
header h1 { font-weight: 700; letter-spacing: -0.5px; }
.meta { max-width: 1200px; margin: 1rem auto; padding: 0 1rem; display: flex; gap: 1.5rem;
        flex-wrap: wrap; justify-content: center; font-size: 0.9rem; color: var(--muted); }
.meta span { background: var(--card-bg); padding: 0.4rem 0.8rem; border-radius: 20px;
             box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
section { max-width: 1200px; margin: 1.5rem auto; padding: 0 1rem; }
h2 { font-size: 1.3rem; margin-bottom: 0.75rem; border-bottom: 2px solid var(--accent);
     padding-bottom: 0.3rem; display: inline-block; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 1rem; }
.card { background: var(--card-bg); border-radius: 10px; padding: 1.2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07); }
.card h3 { font-size: 0.85rem; color: var(--muted); margin-bottom: 0.3rem; text-transform: uppercase; }
.card .value { font-size: 1.6rem; font-weight: 700; color: var(--accent); }
table { width: 100%; border-collapse: collapse; background: var(--card-bg); border-radius: 8px;
        overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.05); margin-top: 0.5rem; font-size: 0.85rem; }
th, td { padding: 0.5rem 0.7rem; text-align: left; }
th { background: #e9ecef; font-weight: 600; }
tr:nth-child(even) { background: #f8f9fa; }
.severity-critical { color: #dc3545; font-weight: 600; }
.severity-warning { color: #fd7e14; font-weight: 600; }
.severity-info { color: #0d6efd; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 520px), 1fr)); gap: 1.25rem; }
.chart-card { background: var(--card-bg); border-radius: 10px; overflow: hidden;
              box-shadow: 0 2px 10px rgba(0,0,0,0.07); }
.chart-card img { width: 100%; height: auto; display: block; }
.caption { padding: 0.6rem 1rem; font-size: 0.8rem; color: var(--muted); text-align: center;
           border-top: 1px solid #f1f3f5; font-weight: 500; }
.chart-desc { padding: 0.5rem 1rem 0.8rem; font-size: 0.8rem; color: var(--muted);
              background: #fafbfc; border-top: 1px dashed #e9ecef; }
.insight { background: var(--card-bg); border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem;
           box-shadow: 0 1px 4px rgba(0,0,0,0.05); border-left: 4px solid var(--accent); }
.insight.critical { border-left-color: #dc3545; }
.insight.warning { border-left-color: #fd7e14; }
.insight.info { border-left-color: #0d6efd; }
.cleaning-section { background: var(--card-bg); border-radius: 10px; padding: 1rem; margin: 1rem auto; max-width: 1200px; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }
.cleaning-section h2 { margin-bottom: 0.5rem; }
.cleaning-table { margin-top: 0.5rem; }
.cleaning-table th { background: #e7f3ff; }
.before-after { display: flex; gap: 2rem; flex-wrap: wrap; justify-content: center; margin: 0.5rem 0; }
.before-after .stat { text-align: center; }
.before-after .stat .num { font-size: 1.4rem; font-weight: 700; color: var(--accent); }
.before-after .stat .lbl { font-size: 0.8rem; color: var(--muted); }
/* Dataset Snapshot styles */
.snapshot { background: var(--card-bg); border-radius: 12px; padding: 1.25rem; margin: 1rem auto; max-width: 1200px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); }
.snapshot h2 { margin-bottom: 0.75rem; }
.commentary { background: #f0f9ff; border-left: 4px solid var(--accent2); padding: 0.75rem 1rem; margin: 0.75rem 0; border-radius: 0 6px 6px 0; font-size: 0.9rem; color: #0c4a6e; }
.type-group { display: inline-block; background: #f1f5f9; padding: 0.25rem 0.6rem; border-radius: 14px; margin: 0.15rem; font-size: 0.8rem; }
.type-group .tag { font-weight: 600; color: var(--accent); }
.data-table { font-size: 0.78rem; max-height: 320px; overflow: auto; }
.data-table th { position: sticky; top: 0; background: #e9ecef; z-index: 1; }
footer { text-align: center; padding: 2rem 1rem; font-size: 0.8rem; color: var(--muted); margin-top: 2rem; }
</style>
</head>
<body>
<header><h1>Skilium EDA Report</h1><p>{{ dataset_name }} &middot; {{ timestamp }}</p></header>

<!-- DATASET SNAPSHOT -->
<div class="snapshot">
<h2>Dataset Snapshot</h2>

<!-- Shape & overview -->
<div class="before-after">
  <div class="stat"><div class="num">{{ row_count|number_format }}</div><div class="lbl">Rows</div></div>
  <div class="stat"><div class="num">{{ column_count }}</div><div class="lbl">Columns</div></div>
  <div class="stat"><div class="num">{{ missing_before|number_format }}</div><div class="lbl">Missing (before cleaning)</div></div>
  <div class="stat"><div class="num">{{ missing_after|number_format }}</div><div class="lbl">Missing (after cleaning)</div></div>
  <div class="stat"><div class="num">{{ dups_before|number_format }}</div><div class="lbl">Duplicates (before)</div></div>
  <div class="stat"><div class="num">{{ dups_after|number_format }}</div><div class="lbl">Duplicates (after)</div></div>
</div>

<!-- Commentary -->
{% if commentary %}
<div class="commentary"><strong>Summary:</strong> {{ commentary }}</div>
{% endif %}

<!-- Column type summary -->
<h3 style="font-size:1rem;margin:0.75rem 0 0.5rem;">Column Types</h3>
{% for group in column_groups %}
<div style="margin-bottom:0.5rem;">
  <span class="type-group"><span class="tag">{{ group.type }}</span> ({{ group.count }})</span>
  {% if group.columns %}
    <span style="font-size:0.82rem;color:var(--muted);">{{ group.columns|join(", ") }}</span>
  {% endif %}
</div>
{% endfor %}

<!-- Head -->
<h3 style="font-size:1rem;margin:0.75rem 0 0.5rem;">First 5 Rows</h3>
<div class="data-table">{{ head_html }}</div>

<!-- Tail -->
<h3 style="font-size:1rem;margin:0.75rem 0 0.5rem;">Last 5 Rows</h3>
<div class="data-table">{{ tail_html }}</div>
</div>

{% if cleaning_log %}
<section class="cleaning-section">
<h2>Data Cleaning Summary</h2>
<div class="before-after">
  <div class="stat"><div class="num">{{ cleaning_log.original_missing_total }}</div><div class="lbl">Missing values (before)</div></div>
  <div class="stat"><div class="num">{{ cleaning_log.original_duplicates }}</div><div class="lbl">Duplicate rows (before)</div></div>
  <div class="stat"><div class="num">{{ cleaning_log.actions|length }}</div><div class="lbl">Cleaning actions</div></div>
</div>
{% if cleaning_log.actions %}
<table class="cleaning-table"><tr><th>Action</th></tr>
{% for action in cleaning_log.actions %}<tr><td>{{ action }}</td></tr>{% endfor %}
</table>
{% endif %}
</section>
{% endif %}

<section><h2>Key Insights</h2>
{% if insights %}
  {% for ins in insights %}
  <div class="insight {{ ins.severity }}">
    <strong>[{{ ins.type.upper() }}]</strong> {{ ins.title }}<br>
    {{ ins.description }}<br>
    {% if ins.recommendation %}<em>&rarr; {{ ins.recommendation }}</em>{% endif %}
  </div>
  {% endfor %}
{% else %}
  <p>No insights generated.</p>
{% endif %}
</section>

<section><h2>Column Quality</h2>
<table><tr><th>Column</th><th>Type</th><th>Missing %</th><th>Unique</th><th>Summary</th></tr>
{% for row in column_quality %}<tr>
  <td>{{ row.name }}</td><td>{{ row.dtype }}</td>
  <td>{% if row.missing_pct > 0 %}<span class="severity-{{ row.missing_class }}">{{ row.missing_pct }}%</span>{% else %}0%{% endif %}</td>
  <td>{{ row.unique }}</td><td>{{ row.summary }}</td>
</tr>{% endfor %}
</table>
</section>

<section><h2>Charts</h2>
<div class="gallery">
{% for chart in charts %}
<div class="chart-card">
  <img src="{{ chart.data_uri }}" alt="{{ chart.title }}" loading="lazy" />
  <div class="caption">{{ chart.title }}</div>
  {% if chart.description %}<div class="chart-desc">{{ chart.description }}</div>{% endif %}
</div>
{% endfor %}
</div>
</section>

<section><h2>Recommendations</h2>
{% if recommendations %}
  <ol>{% for rec in recommendations %}<li><strong>{{ rec.title }}</strong> — {{ rec.description }}</li>{% endfor %}</ol>
{% else %}<p>No recommendations.</p>{% endif %}
</section>

<footer>Generated by skilium-eda v{{ version }} &middot; {{ timestamp }}</footer>
</body>
</html>'''


def _build_readable_type_map(df: pd.DataFrame) -> dict[str, str]:
    """Map pandas dtypes to human-readable type labels."""
    from skilium_eda.core import DataEngine
    type_map = DataEngine.infer_types(df)
    readable: dict[str, str] = {}
    for col in df.columns:
        t = type_map.get(col, "unknown")
        dtype = str(df[col].dtype)
        if t == "id":
            readable[col] = "id"
        elif t == "categorical":
            readable[col] = "categorical"
        elif t == "datetime":
            readable[col] = "datetime"
        elif t == "text":
            readable[col] = "text"
        elif pd.api.types.is_integer_dtype(df[col].dtype):
            readable[col] = "int"
        elif pd.api.types.is_float_dtype(df[col].dtype):
            readable[col] = "float"
        elif pd.api.types.is_numeric_dtype(df[col].dtype):
            readable[col] = "numeric"
        elif pd.api.types.is_bool_dtype(df[col].dtype):
            readable[col] = "categorical"
        else:
            readable[col] = "text"
    return readable


def _build_column_groups(df: pd.DataFrame) -> list[dict]:
    """Group columns by readable type for the report."""
    readable = _build_readable_type_map(df)
    groups: dict[str, list[str]] = {}
    for col, t in readable.items():
        groups.setdefault(t, []).append(col)
    order = ["int", "float", "categorical", "text", "datetime", "id"]
    result = []
    for t in order:
        if t in groups:
            result.append({"type": t, "count": len(groups[t]), "columns": groups[t]})
    # Any types not in order
    for t, cols in groups.items():
        if t not in order:
            result.append({"type": t, "count": len(cols), "columns": cols})
    return result


def _build_commentary(df: pd.DataFrame, type_map: dict[str, str], cleaning_log: dict | None = None) -> str:
    """Generate a plain-English summary of the dataset."""
    rows, cols = df.shape
    parts: list[str] = []

    parts.append(f"This dataset contains {rows:,} rows and {cols} columns.")

    # Type summary
    groups = _build_column_groups(df)
    type_descs = [f"{g['count']} {g['type']}" for g in groups]
    parts.append(f"Columns are: {', '.join(type_descs)}.")

    # Missing values
    if cleaning_log and cleaning_log.get("original_missing_total", 0) > 0:
        orig_missing = cleaning_log["original_missing_total"]
        parts.append(f"{orig_missing:,} missing values were found and handled during cleaning.")

    # Notable observations
    numeric_cols = [c for c, t in type_map.items() if t == "numeric"]
    if numeric_cols:
        parts.append(f"{len(numeric_cols)} numeric column(s) were analyzed for distributions, correlations, and outliers.")

    id_cols = [c for c, t in type_map.items() if t == "id"]
    if id_cols:
        parts.append(f"{len(id_cols)} identifier column(s) were detected and excluded from statistical analysis.")

    return " ".join(parts)


class ReportGenerator:
    """Generate HTML and Markdown reports with embedded charts."""

    def __init__(
        self,
        df: pd.DataFrame,
        profile: dict,
        charts: list[str],
        insights: list[Any],
        cleaning_log: dict | None = None,
        original_df: pd.DataFrame | None = None,  # NEW: for before/after comparison
        chart_descriptions: dict[str, str] | None = None,  # NEW: descriptions for each chart
        output_dir: str = "./skilium_output",
    ) -> None:
        self.df = df
        self.profile = profile
        self.charts = charts or []
        self.insights = insights or []
        self.cleaning_log = cleaning_log or {}
        self.original_df = original_df  # NEW
        self.chart_descriptions = chart_descriptions or {}  # NEW
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_html(self, output_path: str | None = None) -> str:
        """Render a self-contained HTML report with embedded base64 charts."""
        if output_path is None:
            output_path = str(self.output_dir / "report.html")
        try:
            import jinja2
        except ImportError:
            raise ImportError("jinja2 required: pip install jinja2") from None

        env = jinja2.Environment(loader=jinja2.BaseLoader(), autoescape=False)
        env.filters["number_format"] = _fmt_number
        template = env.from_string(HTML_TEMPLATE)

        chart_objs = []
        for path in self.charts:
            if os.path.isfile(path):
                chart_objs.append({
                    "data_uri": _encode_image(path),
                    "title": Path(path).stem.replace("_", " ").title(),
                    "description": self.chart_descriptions.get(Path(path).stem, ""),
                })

        row_count = len(self.df)
        col_count = len(self.df.columns)
        missing_cells = int(self.df.isnull().sum().sum())
        missing_pct = round((missing_cells / (row_count * col_count)) * 100, 2) if row_count else 0.0

        # Build column groups for readable type summary
        column_groups = _build_column_groups(self.df)

        # Build commentary
        from skilium_eda.core import DataEngine
        type_map = DataEngine.infer_types(self.df)
        commentary = _build_commentary(self.df, type_map, self.cleaning_log)

        # Compute before/after stats
        orig_df = self.original_df if self.original_df is not None else self.df
        missing_before = int(orig_df.isna().sum().sum())
        missing_after = int(self.df.isna().sum().sum())
        dups_before = int(orig_df.duplicated().sum())
        dups_after = int(self.df.duplicated().sum())

        # Head and tail as HTML tables
        head_html = orig_df.head(5).to_html(classes="data-table", border=0, index=True)
        tail_html = orig_df.tail(5).to_html(classes="data-table", border=0, index=True)

        html = template.render(
            dataset_name=getattr(self.df, "_skilium_name", "Dataset"),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            row_count=row_count, column_count=col_count,
            missing_percent=missing_pct, memory_mb=round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            duplicate_rows=int(self.df.duplicated().sum()),
            version="0.1.0", insights=[_insight_to_dict(i) for i in self.insights],
            charts=chart_objs, column_quality=_build_column_quality(self.df),
            recommendations=_build_recommendations(self.insights),
            cleaning_log=self.cleaning_log,
            column_groups=column_groups,
            commentary=commentary,
            missing_before=missing_before,
            missing_after=missing_after,
            dups_before=dups_before,
            dups_after=dups_after,
            head_html=head_html,
            tail_html=tail_html,
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html, encoding="utf-8")
        logger.info("HTML report written to %s", output_path)
        return output_path

    def to_markdown(self, output_path: str | None = None) -> str:
        """Generate a Markdown summary."""
        from skilium_eda.core import DataEngine  # local import to avoid circular deps
        if output_path is None:
            output_path = str(self.output_dir / "report.md")
        row_count = len(self.df)
        col_count = len(self.df.columns)
        missing_cells = int(self.df.isnull().sum().sum())
        missing_pct = round((missing_cells / (row_count * col_count)) * 100, 2) if row_count else 0.0
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        name = getattr(self.df, "_skilium_name", "Dataset")

        lines = [
            f"# EDA Report: {name}", "",
            f"**Generated:** {ts} | **skilium-eda v0.1.0**", "", "---", "",
            "## Dataset Overview", "",
            f"| Metric | Value |", f"|--------|-------|",
            f"| Rows | {row_count:,} |", f"| Columns | {col_count} |",
            f"| Missing cells | {missing_cells:,} ({missing_pct}%) |",
            f"| Duplicate rows | {int(self.df.duplicated().sum()):,} |",
            f"| Memory | {round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2)} MB |",
        ]

        # Dataset Snapshot
        lines.extend(["", "## Dataset Snapshot", ""])

        # Before/after stats
        orig_df = self.original_df if self.original_df is not None else self.df
        missing_before = int(orig_df.isna().sum().sum())
        missing_after = int(self.df.isna().sum().sum())
        dups_before = int(orig_df.duplicated().sum())
        dups_after = int(self.df.duplicated().sum())

        lines.extend([
            f"| Metric | Before | After |", f"|--------|--------|-------|",
            f"| Missing values | {missing_before:,} | {missing_after:,} |",
            f"| Duplicate rows | {dups_before:,} | {dups_after:,} |",
            "",
        ])

        # Commentary
        type_map = DataEngine.infer_types(self.df)
        commentary = _build_commentary(self.df, type_map, self.cleaning_log)
        lines.extend([f"**Summary:** {commentary}", ""])

        # Column type groups
        column_groups = _build_column_groups(self.df)
        lines.extend(["### Column Types", ""])
        for group in column_groups:
            lines.append(f"- **{group['type']}** ({group['count']}): {', '.join(group['columns'])}")
        lines.extend(["", "### First 5 Rows", ""])
        lines.append(orig_df.head(5).to_markdown())
        lines.extend(["", "### Last 5 Rows", ""])
        lines.append(orig_df.tail(5).to_markdown())
        lines.extend(["", "---", ""])

        # Add cleaning summary
        if self.cleaning_log:
            lines.extend(["", "## Data Cleaning Summary", ""])
            orig_missing = self.cleaning_log.get("original_missing_total", 0)
            orig_dups = self.cleaning_log.get("original_duplicates", 0)
            lines.extend([
                f"| Metric | Value |", f"|--------|-------|",
                f"| Missing values (before cleaning) | {orig_missing} |",
                f"| Duplicate rows (before cleaning) | {orig_dups} |",
                f"| Cleaning actions taken | {len(self.cleaning_log.get('actions', []))} |",
                "", "**Actions taken:**", "",
            ])
            for action in self.cleaning_log.get("actions", []):
                lines.append(f"- {action}")
            lines.extend(["", "---", ""])

        lines.extend(["## Column Types", "", "| Column | Type |", "|--------|------|"])
        for col_name, dtype in self.df.dtypes.items():
            lines.append(f"| `{col_name}` | `{dtype}` |")
        lines.extend(["", "---", "", "## Key Insights", ""])
        if self.insights:
            for ins in self.insights:
                d = _insight_to_dict(ins)
                emoji = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(d["severity"], "ℹ️")
                lines.append(f"### {emoji} {d['title']}")
                lines.append("")
                lines.append(d["description"])
                if d.get("recommendation"):
                    lines.append(f"**Recommendation:** {d['recommendation']}")
                lines.append("")
        else:
            lines.append("*No insights generated.*")
        lines.extend(["", "---", "", "## Charts", ""])
        if self.charts:
            for chart_path in self.charts:
                if os.path.isfile(chart_path):
                    caption = Path(chart_path).stem.replace("_", " ").title()
                    try:
                        rel = str(Path(chart_path).relative_to(self.output_dir))
                    except ValueError:
                        rel = chart_path
                    lines.extend([f"### {caption}", "", f"![{caption}]({rel})", ""])
        else:
            lines.append("*No charts generated.*")
        lines.extend(["", "---", "", "## Recommendations", ""])
        recs = _build_recommendations(self.insights)
        if recs:
            for idx, rec in enumerate(recs, 1):
                lines.append(f"{idx}. **{rec['title']}** — {rec['description']}")
        else:
            lines.append("*No recommendations.*")
        lines.extend(["", "---", "", "*Generated by skilium-eda v0.1.0*", ""])

        md = "\n".join(lines)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(md, encoding="utf-8")
        logger.info("Markdown report written to %s", output_path)
        return output_path


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _fmt_number(value: int | float) -> str:
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return str(value)


def _encode_image(path: str) -> str:
    ext = Path(path).suffix.lstrip(".").lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "gif": "image/gif", "svg": "image/svg+xml"}.get(ext, "image/png")
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _insight_to_dict(ins: Any) -> dict[str, Any]:
    if isinstance(ins, dict):
        return ins
    return {
        "type": getattr(ins, "type", "info"),
        "column": getattr(ins, "column", None),
        "title": getattr(ins, "title", ""),
        "description": getattr(ins, "description", ""),
        "severity": getattr(ins, "severity", "info"),
        "recommendation": getattr(ins, "recommendation", ""),
    }


def _build_column_quality(df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    total = len(df)
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_count = int(df[col].isnull().sum())
        missing_pct = round((missing_count / total) * 100, 1) if total else 0.0
        if df[col].dtype.kind in "iufc":
            tclass = "num"
        elif df[col].dtype.kind == "b":
            tclass = "bool"
        elif "datetime" in str(df[col].dtype):
            tclass = "date"
        else:
            tclass = "cat"
        if missing_pct >= 20:
            mclass = "high"
        elif missing_pct >= 5:
            mclass = "medium"
        else:
            mclass = "low"
        unique = df[col].nunique(dropna=False)
        if df[col].dtype.kind in "iufc":
            m = df[col].mean()
            summary = f"mean = {m:.2f}" if pd.notna(m) else "—"
        else:
            try:
                mode = df[col].mode().iloc[0] if not df[col].mode().empty else "—"
            except Exception:
                mode = "—"
            summary = f"top = {mode}" if mode != "—" else "—"
        rows.append({"name": col, "dtype": dtype, "type_class": tclass,
                     "missing_pct": missing_pct, "missing_class": mclass,
                     "unique": f"{unique:,}", "summary": summary})
    return rows


def _build_recommendations(insights: list[Any]) -> list[dict]:
    recs: list[dict] = []
    seen: set[str] = set()
    for ins in insights:
        d = _insight_to_dict(ins)
        rec_text = d.get("recommendation", "")
        if rec_text and rec_text not in seen:
            seen.add(rec_text)
            recs.append({"title": d.get("title", "Recommendation"), "description": rec_text})
    return recs
