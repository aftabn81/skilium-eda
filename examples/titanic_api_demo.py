from pathlib import Path

from skilium_eda.core import DataEngine
from skilium_eda.agent import EDAAgent
from skilium_eda.viz import ChartEngine
from skilium_eda.report import ReportGenerator

source = "data/titanic.csv"
output_dir = Path("titanic_api_output")
charts_dir = output_dir / "charts"
output_dir.mkdir(exist_ok=True)
charts_dir.mkdir(exist_ok=True)

# Load, clean, profile
df = DataEngine.load(source)
clean_df, cleaning_report = DataEngine.clean(df)
type_map = DataEngine.infer_types(clean_df)
profile = DataEngine.profile(clean_df, type_map=type_map)

# Agentic insights
agent = EDAAgent(clean_df)
insights = agent.generate_insights()

# Charts
chart_engine = ChartEngine(clean_df, output_dir=str(charts_dir))
charts = chart_engine.generate_all()

# Report
reporter = ReportGenerator(clean_df, profile, charts, insights)
html_path = reporter.to_html(str(output_dir / "report.html"))
md_path = reporter.to_markdown(str(output_dir / "report.md"))

print("✅ Titanic API test complete")
print(f"Rows: {clean_df.shape[0]}")
print(f"Columns: {clean_df.shape[1]}")
print(f"Charts: {len(charts)}")
print(f"Output: {output_dir / 'report.html'}")
