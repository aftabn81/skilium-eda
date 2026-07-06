# skilium-eda

Lightweight agentic EDA library for data science. From raw data to insights, charts, and reports in one command.

## Install

```bash
pip install skilium-eda
```

With S3 support:

```bash
pip install skilium-eda[s3]
```

## Quick Start

### Python API

```python
from skilium_eda import DataEngine, EDAPipeline

# Load and explore
df = DataEngine.load("data.csv")
summary = DataEngine.get_summary(df)
profile = DataEngine.profile(df)

# Full pipeline
pipeline = EDAPipeline("data.csv", output_dir="./reports")
results = pipeline.run()
```

### S3 Support

```python
# Load from S3
df = DataEngine.load("s3://my-bucket/datasets/sales.csv")

# Full pipeline from S3
pipeline = EDAPipeline("s3://my-bucket/datasets/sales.csv", output_dir="./reports")
results = pipeline.run()
```

S3 authentication uses standard AWS credential chains (env vars, `~/.aws/credentials`, IAM roles).

### CLI

```bash
# Full pipeline
skilium-eda run data.csv --output-dir ./reports

# Profile only
skilium-eda profile data.csv --output profile.json

# S3
skilium-eda run s3://bucket/data.csv --output-dir ./reports
```

## Features

- **Load**: CSV, Excel, Parquet, JSON/JSONL — local files and S3 URLs
- **Clean**: Missing values, duplicates, type inference, outlier removal
- **Profile**: Statistics, correlations, distributions, quality metrics
- **Agent**: Rule-based insights, chart selection, action items
- **Visualize**: Distributions, correlations, missing values, boxplots, pairplots
- **Report**: Self-contained HTML with embedded charts, Markdown export

## Project Structure

```
src/skilium_eda/
  __init__.py    # Clean exports
  core.py        # DataEngine: load, clean, profile (with S3)
  agent.py       # EDAAgent: insights, chart selection
  viz.py         # ChartEngine: matplotlib/seaborn charts
  report.py      # HTML and Markdown report generation
  pipeline.py    # EDAPipeline: orchestrate full workflow
  cli.py         # skilium-eda CLI commands
```

6 modules, ~2,800 lines, production-ready.
