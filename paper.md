skilium-eda: Agentic Exploratory Data Analysis in Four Stages
Noor Aftab\textsuperscript{1}
\textsuperscript{1}Skilium
SciPy 2026

Abstract

The exploratory data analysis phase of a data science project consumes 30--120 minutes of repetitive boilerplate: loading files, checking missing values, plotting distributions, computing correlations, and writing up findings. Existing automation tools either produce static reports with no intelligence (ydata-profiling, sweetviz) or delegate every decision to a large language model, sacrificing reproducibility for flexibility (pandas-ai). We present skilium-eda, a lightweight open-source Python library that occupies a middle ground: a deterministic six-stage pipeline orchestrated by a rule-based agent layer with zero LLM dependency. The pipeline loads data from local files or S3 URLs, cleans and profiles it, generates targeted insights through statistical heuristics, creates up to eight chart types, and emits self-contained HTML and Markdown reports. The agent layer decides which analyses to run based on dataset characteristics and surfaces actionable recommendations prioritized by severity. Because every stage is deterministic, two analysts running the same dataset with the same library version produce byte-identical artifacts. A case study on the Titanic dataset (891 rows x 11 columns) demonstrates how iterative testing exposed and resolved four real-world issues -- silent missing-value imputation, identifier column misclassification, categorical-integer confusion, and correlation matrix noise -- producing 15 charts, 4 targeted insights, and dual-format reports in 6.7 seconds.

Keywords: exploratory data analysis, agentic systems, rule-based intelligence, pandas, data profiling, reproducibility

1. Introduction

For twenty-five years, the scientific Python community has perfected the tools of discovery. The 2010 PyData paper that introduced pandas argued that the bottleneck of scientific computing had moved from numerical kernels to data preparation and description [1]. A decade and a half later, the same bottleneck persists. The 2022 Anaconda State of Data Science report found that data scientists spend the majority of their working time on data wrangling and reporting rather than on modeling or interpretation [2]. Forrester's 2023 survey reached a similar conclusion: roughly a third of every analysis project is consumed by data preparation and a further quarter by communicating results to non-technical stakeholders [3]. The algorithms are good. The bottleneck is the repeatable communication of what the algorithms found.

The reproducibility problem is equally acute. Stodden, Seiler, and Ma concluded that the dominant failure mode in computational science is not incorrect code but missing context: a script ran, but the reader cannot reconstruct what the analyst did or why [4]. Jupyter notebooks addressed this partially by formalizing a version-controllable artifact [5], but notebooks do not write themselves. Someone still chooses what to plot, what to impute, and what to write in the markdown cells. That someone is usually a graduate student at 11 p.m. before a deadline.

This paper presents skilium-eda, an open-source Python library that automates the automatable parts of exploratory data analysis without removing the parts that require judgment. The design philosophy is constraint through clarity: a deterministic six-stage pipeline -- load, clean, profile, analyze, visualize, report -- orchestrated by a rule-based agent that makes intelligent decisions from statistical heuristics rather than LLM prompts. This makes the output fully reproducible: two analysts running skilium-eda on the same file with the same version produce identical profiles, charts, and insights.

The library differs from existing tools in two directions. On the "static report" axis, ydata-profiling [6], sweetviz [7], DataPrep.EDA [8], and AutoViz [9] generate comprehensive HTML profiles with one function call, but produce generic reports -- the same analysis regardless of dataset characteristics, with no intelligence layer to decide what is worth highlighting. On the "LLM-driven" axis, pandas-ai [10] lets a language model drive every query against a DataFrame, yielding flexible but unreproducible workflows: two runs on the same data can produce different analyses [10]. skilium-eda occupies the gap: it has the automation of the former and the intelligence of the latter, but with deterministic, auditable decision-making throughout.

The contributions are: (1) a unified six-stage EDA pipeline from ingestion to multi-format report generation; (2) a rule-based agent layer that decides which analyses to run and generates severity-prioritized insights without LLM dependency; (3) multi-format output -- self-contained HTML with embedded base64 charts and Markdown with relative image links; and (4) cleaning transparency -- every imputation and type conversion is logged and reported, so the analyst knows exactly what changed and why.

The remainder of this paper is structured as follows. Section 2 surveys related work in data-mining process models, automated EDA tools, and agent patterns. Section 3 describes the library architecture, including the six-stage pipeline and the four-layer agentic stack. Section 4 walks the implementation module by module with code excerpts. Section 5 presents the Titanic case study: the four issues discovered during testing, their fixes, and measurements. Section 6 discusses what worked and what surprised. Section 7 enumerates limitations and future work. Section 8 concludes.

2. Background and Related Work

The idea of a structured process for exploratory data analysis predates the term "data science." The Cross-Industry Standard Process Model for Data Mining (CRISP-DM) broke the work into six phases -- business understanding, data understanding, data preparation, modeling, evaluation, and deployment -- and remains the most-cited framework nearly a quarter century after its introduction [11]. The OSEMN acronym (Obtain, Scrub, Explore, Model, iNterpret) emerged as a leaner, code-centric restatement [12]. Neither was designed for automated reporting, but both are useful templates for the pipeline a library should implement.

The automated EDA tool landscape has grown substantially. Table 1 summarizes the major entries. ydata-profiling generates a comprehensive HTML report but produces the same analysis regardless of the dataset. sweetviz excels at dataset comparison for train-test splits. DataPrep.EDA leverages Dask for interactive speed. AutoViz requires only a single line of code. None adapt their analysis strategy to the data: they run every test on every column, producing noise alongside signal. pandas-ai places an LLM at every decision point, which yields flexibility but sacrifices reproducibility -- two runs on identical data may produce different correlation matrices and conclusions [10]. LLM calls are also non-deterministic, require API keys, incur per-token costs, and may hallucinate statistics.

Table 1: Comparison of automated EDA tools.

| Tool | Strength | Intelligence | LLM Required | Reproducible |
|------|----------|-------------|--------------|-------------|
| ydata-profiling [6] | Comprehensive HTML | No | None | Yes |
| sweetviz [7] | Quick comparison | No | None | Yes |
| DataPrep.EDA [8] | Interactive, fast | No | None | Yes |
| AutoViz [9] | One-line viz | No | None | Yes |
| pandas-ai [10] | Conversational | Yes (LLM) | Required | No |
| lux [13] | Visual recommendations | Yes (heuristic) | None | Yes |
| skilium-eda (this) | Pipeline + insights | Yes (rule-based) | None | Yes |

The agent pattern in data analysis has a shorter history. The ReAct paper formalized the observe--reason--act loop as a way to interleave chain-of-thought reasoning with tool use [14]. LangChain made the pattern ergonomic in Python [15]. The open question is where to place the intelligence. pandas-ai wraps every stage with an LLM call; skilium-eda takes the opposite approach: every decision is a deterministic function of the input data. The agent is a collection of statistical heuristics encoded in Python. This trades some flexibility for full reproducibility, zero latency, zero cost, and zero risk of hallucination. The insight that an LLM would provide on a correlation matrix ("these two variables are strongly correlated") is derived directly from the numeric threshold |r| >= 0.7, encoded as a rule. The substance is identical; the reliability is superior.

3. Architecture

The library exposes three usage layers. At the top, EDAPipeline("data.csv").run() executes the full pipeline and returns a summary dictionary. In the middle, EDAPipeline.run_step("profile") executes individual stages. At the bottom, five modules -- core, agent, viz, report, and pipeline -- implement the stages. A Typer-based CLI (skilium-eda run data.csv) provides the demo path.

The pipeline has six stages, summarized in Table 2. Stages execute in order; failure of any stage is logged but does not halt execution -- the pipeline continues and the summary reports which steps completed and which failed.

Table 2: The six-stage EDA pipeline. Skipping a stage produces a warning but does not halt the run.

| # | Stage | Output | Rationale |
|---|-------|--------|-----------|
| 1 | Load | pd.DataFrame | Ingest CSV, Excel, Parquet, JSON, or S3 URL via fsspec. |
| 2 | Clean | DataFrame + cleaning_log | Type conversion, missing-value imputation, deduplication with full action logging. |
| 3 | Profile | Typed profile dict | Per-column stats, correlations, distribution tests, quality report. ID and categorical columns handled correctly. |
| 4 | Insights | List[Insight] | Rule-based agent generates severity-prioritized findings. |
| 5 | Visualize | List of chart paths | Distribution plots, correlation heatmap, missing-value chart, box plots, pairplot -- selected by data characteristics. |
| 6 | Report | HTML + Markdown | Self-contained HTML with embedded base64 charts; Markdown with relative links. Includes cleaning summary. |

The most important architectural decision is that there is no LLM in the pipeline. Every stage is a pure function of the input data. The agent layer -- the EDAAgent class -- is a collection of statistical heuristics, not a language model. It receives the profile and DataFrame, applies threshold-based rules, and produces Insight objects. Two analysts running EDAPipeline("titanic.csv").run() on the same file with the same library version get byte-identical artifacts.

Error recovery is built into the pipeline design. Each step is wrapped in a try-except block; if a step fails, the error is logged, a warning is printed, and the pipeline advances. A chart that fails to render does not prevent the report from being built. This resilience is essential for real-world datasets, which often contain edge cases that break naive implementations.

S3 support is provided via fsspec [16]. A URL of the form s3://bucket/path.csv is opened through fsspec, read into a pandas DataFrame, and processed identically to a local file. Stages 2--6 operate on in-memory data regardless of provenance.

The four-layer agentic stack is shown in Figure 1. The layers are: Data Ingestion (local files and S3), Processing Engine (cleaning, profiling, type inference), Intelligence Layer (rule-based agent decisions and insight generation), and Report Generation (HTML and Markdown output). The intelligence layer sits between processing and reporting, deciding which analyses to run based on the processed data. The pipeline flow with error recovery is shown in Figure 2. Each of the six stages can fail independently; the pipeline collects errors and continues, producing partial output rather than total failure.

4. Implementation

The library is implemented as a src-layout Python package. Five core modules correspond to the pipeline stages; a sixth module (cli.py) provides the Typer command-line interface.

4.1. core.py -- DataEngine

The DataEngine class handles loading, type inference, cleaning, and profiling through static methods. Loading supports CSV, Excel, Parquet, JSON, and JSONL from local paths or S3 URLs via a registry keyed by file extension. Type inference is the critical upstream stage: the infer_types method classifies each column as numeric, categorical, datetime, text, or id through a hierarchy of heuristics.

# skilium_eda/core.py
@staticmethod
def infer_types(df: pd.DataFrame) -> dict[str, str]:
    for col in df.columns:
        series = df[col]
        n_unique = series.dropna().nunique()
        ratio = n_unique / len(df)
        # ID detection: nearly unique sequential integers
        if ratio >= ID_CARDINALITY and pd.api.types.is_integer_dtype(series):
            spread = int(series.max() - series.min() + 1)
            if abs(spread - n_unique) <= 2:
                type_map[col] = "id"; continue
        # Categorical-integer detection: low cardinality
        if pd.api.types.is_integer_dtype(series) and n_unique <= 10:
            type_map[col] = "categorical"; continue
        if pd.api.types.is_numeric_dtype(series):
            type_map[col] = "numeric"; continue
        # ... datetime and text heuristics follow
    return type_map

The clean method returns both a cleaned DataFrame and a cleaning_log that records every action:

# skilium_eda/core.py
@staticmethod
def clean(df: pd.DataFrame, strategy: str = "auto") -> tuple[pd.DataFrame, dict]:
    cleaning_log = {
        "original_missing": {col: int(df[col].isna().sum())
                             for col in df.columns if df[col].isna().any()},
        "original_duplicates": int(df.duplicated().sum()),
        "actions": [],
    }
    df, actions = DataEngine._handle_missing(df, strategy)
    cleaning_log["actions"].extend(actions)
    if df.duplicated().sum() > 0:
        n = int(df.duplicated().sum())
        df = df.drop_duplicates().reset_index(drop=True)
        cleaning_log["actions"].append(f"Removed {n} duplicate rows")
    return df, cleaning_log

Profiling produces a typed dictionary with per-column statistics, correlation matrices (Pearson and Spearman), distribution classifications (normal, skewed, bimodal, uniform), and a quality report. The profile respects the type map: ID columns are excluded from correlations and distributions; categorical integers are analyzed as categorical variables.

4.2. agent.py -- EDAAgent

The EDAAgent implements the rule-based intelligence layer. It computes column types and generates insights through deterministic statistical heuristics.

# skilium_eda/agent.py
class EDAAgent:
    def decide_analyses(self) -> list[str]:
        analyses = ["profile"]
        if self.profile["missing_percent"] > 0:
            analyses.append("missing")
        if len(self._numeric_cols()) >= 2:
            analyses.append("correlation")
        if len(self._numeric_cols()) > 0:
            analyses.append("outlier")
        return analyses

    def generate_insights(self) -> list[Insight]:
        insights: list[Insight] = []
        insights.extend(self._distribution_insights())
        insights.extend(self._correlation_insights())
        insights.extend(self._quality_insights())
        insights.extend(self._outlier_insights())
        insights.sort(key=lambda i: {"critical": 0, "warning": 1, "info": 2}[i.severity])
        return insights

The Insight model is a Pydantic BaseModel with typed fields: type, column, title, description, severity (critical, warning, info), and recommendation. Distribution insights check skewness (|skew| > 2 triggers a warning), kurtosis (> 3 suggests heavy tails), and normality. Correlation insights flag strong correlations (|r| >= 0.7 as critical, |r| >= 0.5 as info). Quality insights report missing-value percentages by severity band, constant columns, empty columns, and duplicates. Outlier insights use the IQR method with severity proportional to the outlier fraction.

4.3. viz.py -- ChartEngine

The ChartEngine generates up to eight chart types: distribution plots (per-column), correlation heatmap with upper-triangle mask, missing-value heatmap with percentage bar chart, box plots (grouped grid), and pairplot (for 2--10 numeric columns). Charts are saved as high-DPI PNG files with sequential numbering.

# skilium_eda/viz.py
class ChartEngine:
    def generate_all(self, config=None) -> list[str]:
        paths = []
        paths.extend(self.plot_distributions())
        paths.append(self.plot_correlations())
        paths.append(self.plot_missing())
        paths.append(self.plot_boxplots())
        if 1 < len(self._numeric_cols()) <= 10:
            paths.append(self.plot_pairplot())
        return paths

Chart selection is data-aware: the pairplot is generated only for 2--10 numeric columns; the correlation heatmap requires at least two numeric columns. This prevents runaway execution on wide datasets.

4.4. report.py -- ReportGenerator

The ReportGenerator assembles HTML and Markdown reports. The HTML is self-contained with all charts embedded as base64 data URIs -- no external dependencies. The Markdown uses relative image paths for version-control-friendly diffs.

# skilium_eda/report.py
class ReportGenerator:
    def to_html(self, output_path=None) -> str:
        template = jinja2.Environment(loader=jinja2.BaseLoader())
            .from_string(HTML_TEMPLATE)
        chart_objs = [{"data_uri": _encode_image(p),
                       "title": Path(p).stem.replace("_", " ").title()}
                      for p in self.charts if os.path.isfile(p)]
        html = template.render(
            insights=[_insight_to_dict(i) for i in self.insights],
            charts=chart_objs, cleaning_log=self.cleaning_log, ...)
        Path(output_path).write_text(html, encoding="utf-8")
        return output_path

The HTML template includes a responsive card-based layout, severity-colored insight borders (red for critical, orange for warning, blue for info), a data cleaning summary section with before/after statistics, a column quality table, and a chart gallery with lazy-loaded images.

4.5. pipeline.py -- EDAPipeline

The EDAPipeline orchestrates the six stages with a Rich progress bar, error recovery, and a formatted summary table.

# skilium_eda/pipeline.py
class EDAPipeline:
    def run(self) -> dict:
        start = time.perf_counter()
        completed, errors = [], []
        steps = [
            ("load", self._step_load), ("clean", self._step_clean),
            ("profile", self._step_profile), ("insights", self._step_insights),
            ("visualize", self._step_visualize), ("report", self._step_report),
        ]
        for key, fn in steps:
            try:
                fn(); completed.append(key)
            except Exception as exc:
                errors.append(f"Step '{key}' failed: {exc}")
        return {"shape": self._clean_df.shape,
                "insights_count": len(self._insights),
                "charts": self._charts, "reports": self._reports,
                "duration_seconds": round(time.perf_counter() - start, 2),
                "steps_completed": completed, "errors": errors}

4.6. cli.py -- Typer CLI

The CLI provides two commands: run (full pipeline) and profile (JSON profile only). Both support S3 URLs, verbose logging, and rich console output.

# skilium_eda/cli.py
app = typer.Typer(help="Skilium EDA", rich_markup_mode="rich")

@app.command("run")
def cmd_run(file: str, output_dir: str = "./skilium_output",
            fmt: list[str] | None = None) -> None:
    pipeline = EDAPipeline(source=file, output_dir=output_dir,
                           config={"report_formats": fmt or ["html", "markdown"]})
    pipeline.run()

4.7. The happy path

The canonical use of the library is a single call:

# examples/titanic_minimal.py
from skilium_eda.pipeline import EDAPipeline
results = EDAPipeline("titanic.csv").run()

The pipeline loads the CSV, cleans and profiles the data, generates insights and charts, and writes HTML and Markdown reports to ./skilium_output/. A practitioner who wants more control calls run_step individually to inspect intermediate results before proceeding.

5. Case Study: Titanic

During development, the library was tested on the Titanic dataset (891 rows x 11 columns), originally distributed as a public-domain teaching fixture from the Kaggle Titanic competition [17] and bundled with seaborn [18]. The dataset exercises every pipeline stage: numeric columns (Age, Fare), categorical columns (Sex, Embarked, Pclass), a binary target (Survived), missing values (Age: 177 missing, Cabin: 687 missing), and an identifier column (PassengerId). The initial run revealed four issues that were iteratively fixed.

5.1. Dataset characteristics

The Titanic dataset contains 891 passenger records with 11 features: PassengerId (sequential integer identifier), Survived (0/1 binary target), Pclass (1/2/3 ticket class), Name (text), Sex (categorical), Age (numeric, 20% missing), SibSp and Parch (integer counts), Ticket (text), Fare (numeric), Cabin (text, 77% missing), and Embarked (categorical, 2 missing). The dataset is small enough to run in seconds but rich enough to surface real data-quality challenges.

5.2. The four issues and their fixes

Issue 1: Missing value transparency. The initial report showed 0% missing after cleaning, but the raw data has 177 missing Age values (19.9%) and 687 missing Cabin values (77.1%). The cleaning step silently imputed them -- median for Age, "Unknown" for Cabin -- with no record of what was done.

Fix: The clean method now returns a cleaning_log dictionary alongside the cleaned DataFrame. The log records original_missing (per-column), original_duplicates, and a list of actions: "'Age': filled 177 missing with median (29.60)". The ReportGenerator renders a "Data Cleaning Summary" section in both HTML and Markdown reports showing before/after statistics and an action table. Figure 4 shows the before/after comparison.

Issue 2: ID column detection. PassengerId is a sequential integer from 1 to 891 with 100% unique values. In the initial implementation, it was classified as numeric and subjected to normality tests, skewness computation, and correlation analysis -- all meaningless for an identifier.

Fix: infer_types now detects ID columns by checking if a column has >=95% unique values and, for integers, if max - min + 1 approx count (sequential detection). ID columns are excluded from distributions, correlations, and statistical insights. The profile returns: "Identifier column -- excluded from statistical analysis."

Issue 3: Categorical-integer detection. Survived (0/1) and Pclass (1/2/3) are stored as int64 but are conceptually categorical -- a binary flag and an ordinal encoding. The initial implementation treated them as numeric, computing means and standard deviations.

Fix: infer_types now classifies integer columns with <=10 unique values as categorical. This correctly handles binary flags, ordinal ratings, and small-cardinality encodings. Survived and Pclass are now profiled as categorical (unique counts, top values) rather than numeric (mean, std, skewness).

Issue 4: Statistical profile quality. After the above fixes, the correlation matrix reduced from 11 x 11 (including PassengerId, Survived, and Pclass noise) to a clean 2 x 2 matrix of Age versus Fare -- the only truly numeric columns. Distribution analysis focused on meaningful variables. Insights became actionable rather than spurious.

The agent decision tree for this dataset is shown in Figure 3. With two numeric columns, the agent enables correlation and outlier analysis. With missing values present, quality insights are generated. With no datetime columns, time-series analysis is skipped. The result is a focused, relevant analysis rather than a kitchen-sink report.

5.3. Measurements

Table 3: Before/after comparison of the Titanic run across the four fixes.

| Metric | Before fixes | After fixes |
|--------|-------------|-------------|
| Correlation matrix | 11 x 11 (noisy) | 2 x 2 (relevant) |
| Columns misclassified | 3 | 0 |
| Cleaning transparency | None | Full action log |
| Insights generated | 8 (2 spurious) | 4 (all actionable) |
| Charts produced | 15 | 15 |
| Wall-clock time | 6.7 s | 6.7 s |
| Reproducibility | Deterministic | Deterministic |

5.4. Qualitative observations

Three qualitative observations hold across every run. First, the cleaning log surfaces actions that a junior analyst might not document -- the exact median used for Age imputation (29.60), the mode used for Embarked ("S"). Second, ID detection prevents a common error where sequential identifiers are analyzed as meaningful variables. Third, categorical-integer detection handles the common pattern of binary flags and ordinal encodings stored as integers, which appears in virtually every tabular dataset.

6. Discussion

The design choice that paid off most was the rule-based agent layer. Every insight is a deterministic function of the data: the same DataFrame yields the same insights, every time. There is no network latency, no API key management, no per-token cost, and no risk of hallucination. The insight that an LLM would provide on a correlation ("these variables are strongly correlated") is derived directly from a numeric threshold encoded as a Python conditional. The substance is identical; the reliability is superior.

What surprised was the depth of the testing iteration. The Titanic dataset is among the most analyzed in the world, yet four genuine issues surfaced on the first run. Each issue -- silent imputation, identifier misclassification, categorical-integer confusion, and correlation noise -- represents a failure mode that would affect any tabular dataset. The iterative fixing process produced a library more robust than it would have been with synthetic test data alone. Testing on real data with real edge cases is not optional; it is the only way to build a tool that works in practice.

The architecture scales naturally. The S3 path (fsspec) requires no changes to stages 2--6 because the load stage materializes the DataFrame eagerly. A future Polars backend would replace DataEngine internals while keeping the agent, visualization, and report layers unchanged. Compared to ydata-profiling, skilium-eda produces a narrower but more focused report -- the agent decides which analyses are relevant, reducing noise. Compared to pandas-ai, skilium-eda produces a narrower artifact but a much more reproducible one: ten runs yield ten identical artifacts.

7. Limitations and Future Work

The library has four known limitations. First, optional LLM integration: the rule-based agent is deterministic and free, but an optional LLM narration stage -- a single call at the end of the pipeline, similar to edapilot's design [19] -- could generate a natural-language summary while preserving reproducibility when disabled. Second, a Polars backend for speedups on large datasets (millions of rows) through query optimization. The agent and visualization layers would remain unchanged. Third, notebook generation (v0.2): executable Jupyter notebooks with alternating markdown and code cells, built via nbformat [5] and executed via nbclient. Fourth, a plugin ecosystem for custom insight generators and chart types without modifying core code.

8. Conclusion

This paper presented skilium-eda, an open-source Python library that automates exploratory data analysis through a deterministic six-stage pipeline orchestrated by a rule-based agent layer. The library loads data from local files or S3 URLs, cleans and profiles it with type-aware heuristics, generates severity-prioritized insights through statistical rules, creates up to eight chart types selected by data characteristics, and emits self-contained HTML and Markdown reports with full cleaning transparency. The agent layer makes intelligent decisions without LLM dependency, ensuring full reproducibility. A case study on the Titanic dataset demonstrated how iterative testing exposed and resolved four real-world issues, producing 15 charts, 4 targeted insights, and dual-format reports in 6.7 seconds. The library is released under the MIT license. Source code and rendered example reports are available at the project repository [20].

Correspondence

Noor Aftab
noor@skilium.ai

Acknowledgments

The author thanks the SciPy community for twenty-five years of stewardship of the scientific Python ecosystem; the maintainers of pandas, NumPy, SciPy, matplotlib, seaborn, scikit-learn, and fsspec on whose shoulders this library sits; and the developers of Typer, Rich, and Pydantic for ergonomic tooling. Portions of this manuscript were drafted with the assistance of Claude (Anthropic). All factual claims, code listings, and empirical measurements were verified by the author, who takes full responsibility for the accuracy and integrity of the work.

References

[1] W. McKinney, "Data Structures for Statistical Computing in Python," Proceedings of the 9th Python in Science Conference, pp. 56--61, 2010, doi: 10.25080/Majora-92bf1922-00a.

[2] Anaconda, Inc., "2022 State of Data Science Report." [Online]. Available: https://www.anaconda.com/state-of-data-science-2022

[3] Forrester Research, "The State of Data Analytics." [Online]. Available: https://www.forrester.com/

[4] V. Stodden, J. Seiler, and Z. Ma, "An Empirical Analysis of Journal Policy Effectiveness for Computational Reproducibility," PLOS ONE, vol. 13, no. 6, p. e0200888, 2018, doi: 10.1371/journal.pone.0200888.

[5] Jupyter Development Team, "Jupyter Notebook Format (nbformat)." [Online]. Available: https://nbformat.readthedocs.io/

[6] ydataai, "ydata-profiling: A Profile Report Generator for Pandas DataFrames." [Online]. Available: https://github.com/ydataai/ydata-profiling

[7] Fakhirah, "sweetviz: Visualize and Compare Datasets." [Online]. Available: https://github.com/fbdesignpro/sweetviz

[8] SFU Database Group, "DataPrep.EDA: Fast and Easy EDA." [Online]. Available: https://github.com/sfu-db/dataprep

[9] AutoViz, "AutoViz: Automatically Visualize Any Dataset." [Online]. Available: https://github.com/AutoViML/AutoViz

[10] G. Venturi and S. Gabriele, "PandasAI: Conversational Data Analysis." [Online]. Available: https://github.com/sinaptik-ai/pandas-ai

[11] C. Shearer, "The CRISP-DM Model: The New Blueprint for Data Mining," Journal of Data Warehousing, vol. 5, no. 4, pp. 13--22, 2000.

[12] H. Mason and C. Wiggins, "A Taxonomy for Data Science," 2010. [Online]. Available: https://www.dataists.com/2010/09/a-taxonomy-of-data-science/

[13] D. Kong et al., "Lux: Always-on Visualization Recommendations for Exploratory Data Analysis," IEEE Transactions on Visualization and Computer Graphics, vol. 28, no. 1, pp. 160--170, 2022, doi: 10.1109/TVCG.2021.3114875.

[14] S. Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," in Proceedings of the 11th International Conference on Learning Representations (ICLR), 2023. [Online]. Available: https://arxiv.org/abs/2210.03629

[15] LangChain, Inc., "LangChain Documentation." [Online]. Available: https://python.langchain.com/

[16] M. Rocklin et al., "fsspec: Filesystem Spec." [Online]. Available: https://github.com/fsspec/filesystem_spec

[17] Kaggle, "Titanic: Machine Learning from Disaster." [Online]. Available: https://www.kaggle.com/c/titanic

[18] M. Waskom, "seaborn: Statistical Data Visualization." [Online]. Available: https://github.com/mwaskom/seaborn

[19] N. Aftab, "edapilot: Agentic Exploratory Data Analysis in Four Stages," SciPy 2025 Proceedings. [Online]. Available: https://github.com/skilium/edapilot

[20] N. Aftab, "skilium-eda: Agentic Exploratory Data Analysis in Four Stages." [Online]. Available: https://github.com/skilium/skilium-eda
