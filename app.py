from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from datasentry.profiler import profile_dataframe, build_quality_findings
from datasentry.corruption import inject_corruptions


from pathlib import Path
import streamlit as st

report_path = Path("reports/model_evaluation_report.txt")

if report_path.exists():
    st.download_button(
        label="Download Model Evaluation Report",
        data=report_path.read_text(encoding="utf-8"),
        file_name="DataSentry_Model_Evaluation_Report.txt",
        mime="text/plain",
    )
else:
    st.info("Run the ML training pipeline to generate the evaluation report.")

st.set_page_config(
    page_title="DataSentry ML | Data Quality Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Design system ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --bg: #0b1020;
  --panel: #121a2b;
  --panel-2: #172238;
  --border: rgba(153, 174, 210, .16);
  --muted: #94a3b8;
  --text: #e8eef9;
  --teal: #48e0c2;
  --blue: #7ca7ff;
  --amber: #f5bf62;
  --red: #ff7d8b;
}
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
}
.stApp {
  background:
    radial-gradient(ellipse at 8% 0%, rgba(43, 95, 156, .20), transparent 32%),
    radial-gradient(ellipse at 95% 8%, rgba(31, 164, 151, .10), transparent 28%),
    var(--bg);
  color: var(--text);
}
[data-testid="stHeader"] { background: rgba(11,16,32,.85); }
[data-testid="stSidebar"] {
  background: #0e1627;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text); }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px; }
h1, h2, h3, [data-testid="stMetricValue"] {
  font-family: 'Space Grotesk', sans-serif !important;
}
h1 { letter-spacing: -1.2px; font-size: 2.25rem !important; }
h2 { letter-spacing: -.5px; font-size: 1.25rem !important; }
p, label, .stCaption { color: var(--muted); }
[data-testid="stMetric"] {
  background: linear-gradient(145deg, rgba(23,34,56,.96), rgba(18,26,43,.96));
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 17px 18px;
  min-height: 115px;
}
[data-testid="stMetricLabel"] { color: #9aabc4 !important; font-size: .82rem; }
[data-testid="stMetricValue"] { color: #f3f7ff !important; font-size: 1.8rem; }
[data-testid="stMetricDelta"] { font-size: .75rem; }
.ds-eyebrow {
  color: var(--teal); text-transform: uppercase; letter-spacing: 2px;
  font-size: .70rem; font-weight: 700; margin-bottom: .35rem;
}
.ds-subtitle { color: #9aabc4; margin-top: -.6rem; font-size: .95rem; }
.ds-panel {
  background: linear-gradient(145deg, rgba(20,30,49,.96), rgba(15,23,40,.96));
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 18px 20px;
  margin-bottom: 12px;
}
.ds-chip {
  display: inline-block; border: 1px solid rgba(72,224,194,.28);
  background: rgba(72,224,194,.08); color: #76ead3;
  border-radius: 999px; padding: 5px 10px; font-size: .72rem; font-weight: 600;
}
.ds-muted { color: #94a3b8; font-size: .82rem; }
div[data-testid="stTabs"] button { font-weight: 600; }
div[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--teal) !important;
  border-bottom-color: var(--teal) !important;
}
.stButton button, .stDownloadButton button {
  border-radius: 10px;
  border: 1px solid rgba(72,224,194,.38);
  background: rgba(72,224,194,.10);
  color: #9af4e4;
  font-weight: 600;
}
.stButton button:hover, .stDownloadButton button:hover {
  border-color: var(--teal); color: white; background: rgba(72,224,194,.18);
}
[data-testid="stFileUploader"] {
  border: 1px dashed rgba(124,167,255,.4);
  border-radius: 14px;
  background: rgba(124,167,255,.04);
  padding: 8px;
}
hr { border-color: var(--border); }
</style>
""", unsafe_allow_html=True)

PLOT_TEMPLATE = "plotly_dark"
CHART_LAYOUT = dict(
    template=PLOT_TEMPLATE,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#b9c7dc"),
    margin=dict(l=8, r=8, t=42, b=8),
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(gridcolor="rgba(153,174,210,.10)", zeroline=False),
)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🛡️ DataSentry")
    st.markdown('<span class="ds-chip">ML DATA INTELLIGENCE</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Workspace")
    uploaded = st.file_uploader("Upload dataset", type=["csv"])
    st.caption("CSV files · CPU-only processing")
    st.markdown("---")
    st.markdown("### Data source")
    use_demo = st.toggle("Use built-in demo dataset", value=uploaded is None)
    st.markdown("---")
    st.markdown("### Monitoring settings")
    outlier_sensitivity = st.select_slider(
        "Outlier sensitivity",
        options=["Conservative", "Balanced", "Sensitive"],
        value="Balanced",
    )
    st.caption("Sensitivity setting is a UI preference in this starter; baseline IQR checks are unchanged.")
    st.markdown("---")
    st.markdown('<div class="ds-muted">DataSentry ML v0.2 · Local workspace</div>', unsafe_allow_html=True)

# ---------- Data loading ----------
def make_demo_data() -> pd.DataFrame:
    # Small in-memory demo. Use the downloadable 20k CSV for full-size testing.
    return pd.DataFrame({
        "order_id": [f"ORD{i:04d}" for i in range(1, 17)],
        "customer_age": [21, 35, 28, 42, 999, 31, None, 26, 35, 29, 33, 41, 24, 38, 45, 30],
        "region": ["North", "South", "West", "North", "East", "West", "South", "north", "East", "West", "South", "North", "Central", "SOUTH ", "East", "West"],
        "sales_channel": ["Website", "Mobile App", "Store", "Website", "Marketplace", "Store", "Mobile App", "website", "Website", "Marketplace", "Store", "Mobile App", "Website", "STORE ", "Marketplace", "Website"],
        "order_amount": [250.0, 540.0, 125.0, 890.0, 120.0, None, 310.0, 280.0, 540.0, 9999.0, 410.0, 365.0, 720.0, 230.0, -100.0, 415.0],
        "quantity": [1, 2, 1, 3, 1, 2, 1, 1, 2, 4, 1, 2, 3, 1, 2, 1],
        "discount_pct": [0, 10, 5, 15, 0, 20, 10, 5, 15, 25, 0, 10, 15, 5, 110, 10],
        "customer_email": ["a@example.com", "b@example.com", "c@example.com", None, "e@example.com", "f@example.com", "g@example.com", "h@example.com", "not-an-email", "j@example.com", "k@example.com", "l@example.com", "m@example.com", "n@example.com", "o@", "p@example.com"],
    })

df = None
source_label = ""
if uploaded is not None and not use_demo:
    try:
        df = pd.read_csv(uploaded, nrows=100_000, low_memory=False)
        source_label = uploaded.name
    except Exception as exc:
        st.error(f"Unable to read this CSV: {exc}")
        st.stop()
elif uploaded is not None and use_demo:
    st.sidebar.info("Demo mode is on. Turn it off to analyze the uploaded file.")
    df = make_demo_data()
    source_label = "Built-in demo dataset"
else:
    df = make_demo_data()
    source_label = "Built-in demo dataset"

# ---------- Header ----------
st.markdown('<div class="ds-eyebrow">DATA OBSERVABILITY PLATFORM</div>', unsafe_allow_html=True)
st.title("Dataset Health Overview")
st.markdown(
    '<div class="ds-subtitle">Understand data reliability, investigate quality issues, and prepare cleaner data for analytics and ML.</div>',
    unsafe_allow_html=True,
)
st.write("")
left, right = st.columns([3, 1])
with left:
    st.markdown(f'<span class="ds-chip">● LOCAL ANALYSIS</span> &nbsp; <span class="ds-muted">Source: {source_label}</span>', unsafe_allow_html=True)
with right:
    st.markdown('<div style="text-align:right;color:#94a3b8;font-size:.8rem;padding-top:6px">NO GPU REQUIRED</div>', unsafe_allow_html=True)

try:
    profile = profile_dataframe(df)
    findings = build_quality_findings(df)
except Exception as exc:
    st.error(f"Could not analyze this dataset: {exc}")
    st.stop()

# Health score is a transparent heuristic, not a trained model prediction.
issue_penalty = min(
    100.0,
    profile["missing_pct"] * 1.8
    + (profile["duplicate_rows"] / max(profile["rows"], 1) * 100) * 1.5
    + min(len(findings) * 0.6, 20),
)
health_score = max(0, round(100 - issue_penalty))
risk_label = "Healthy" if health_score >= 90 else "Needs attention" if health_score >= 75 else "High risk"

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Dataset health", f"{health_score}/100", risk_label)
k2.metric("Total records", f"{profile['rows']:,}")
k3.metric("Total columns", f"{profile['columns']:,}")
k4.metric("Missing cells", f"{profile['missing_cells']:,}", f"{profile['missing_pct']:.2f}% of all cells")
k5.metric("Duplicate rows", f"{profile['duplicate_rows']:,}")

st.write("")
tab_overview, tab_findings, tab_data, tab_lab = st.tabs(
    ["◈ Overview", "⚑ Quality findings", "▦ Data explorer", "⌁ Corruption lab"]
)

with tab_overview:
    col_a, col_b = st.columns([1.45, 1])
    with col_a:
        st.markdown("### Column completeness")
        column_profile = profile["columns_profile"].copy()
        column_profile["complete_pct"] = 100 - column_profile["missing_pct"]
        chart = px.bar(
            column_profile.sort_values("missing_pct", ascending=True),
            x="complete_pct", y="column", orientation="h",
            color="complete_pct", range_x=[0, 100],
            color_continuous_scale=["#ff7d8b", "#f5bf62", "#48e0c2"],
            labels={"complete_pct": "Complete (%)", "column": ""},
        )
        chart.update_layout(**CHART_LAYOUT, height=350, coloraxis_showscale=False, title=None)
        st.plotly_chart(chart, use_container_width=True)
    with col_b:
        st.markdown("### Quality signal mix")
        if findings.empty:
            st.success("No issues were found by the current baseline checks.")
        else:
            mix = findings["issue"].value_counts().reset_index()
            mix.columns = ["Issue type", "Count"]
            pie = px.pie(
                mix, names="Issue type", values="Count", hole=0.68,
                color_discrete_sequence=["#48e0c2", "#7ca7ff", "#f5bf62", "#ff7d8b", "#b6a0ff"],
            )
            pie.update_traces(textposition="inside", textinfo="percent")
            pie.update_layout(**CHART_LAYOUT, height=350, showlegend=True, legend=dict(orientation="h", y=-0.18))
            st.plotly_chart(pie, use_container_width=True)

    st.markdown("### Priority queue")
    if findings.empty:
        st.success("No baseline findings to prioritize.")
    else:
        priority = findings.copy()
        priority["_rank"] = priority["severity"].map({"High": 0, "Medium": 1, "Low": 2}).fillna(3)
        priority = priority.sort_values(["_rank", "column"]).drop(columns="_rank").head(8)
        st.dataframe(priority, use_container_width=True, hide_index=True)
    st.caption("Dataset health is a heuristic summary score, not an ML prediction or a guarantee of data correctness.")

with tab_findings:
    st.markdown("### Investigate detected issues")
    if findings.empty:
        st.success("No issues were detected by the current baseline checks.")
    else:
        f1, f2 = st.columns([1, 1])
        with f1:
            selected_severity = st.multiselect("Filter severity", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        with f2:
            issue_types = sorted(findings["issue"].unique().tolist())
            selected_issues = st.multiselect("Filter issue type", issue_types, default=issue_types)
        filtered = findings[
            findings["severity"].isin(selected_severity)
            & findings["issue"].isin(selected_issues)
        ]
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button(
            "↓ Export findings CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="datasentry_findings.csv",
            mime="text/csv",
        )

with tab_data:
    st.markdown("### Inspect records")
    search = st.text_input("Search visible rows", placeholder="Type a value to filter the preview…")
    view = df.copy()
    if search:
        mask = view.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        view = view[mask]
    st.caption(f"Showing {min(len(view), 500):,} of {len(view):,} matching records.")
    st.dataframe(view.head(500), use_container_width=True, hide_index=True)
    st.download_button(
        "↓ Export analyzed dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="datasentry_dataset.csv",
        mime="text/csv",
    )

with tab_lab:
    st.markdown("### Controlled corruption lab")
    st.markdown("Generate a reproducible sample with injected missing values. Use this later to evaluate detection models.")
    c1, c2 = st.columns(2)
    with c1:
        fraction = st.slider("Fraction of cells to attempt to corrupt", 0.01, 0.10, 0.03, 0.01)
    with c2:
        seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=42)
    if st.button("Generate corrupted sample", type="primary"):
        corrupted, labels = inject_corruptions(df, seed=int(seed), fraction=float(fraction))
        st.session_state["ds_corrupted"] = corrupted
        st.session_state["ds_labels"] = labels
    if "ds_corrupted" in st.session_state:
        st.success("Corrupted sample generated.")
        st.dataframe(st.session_state["ds_corrupted"].head(100), use_container_width=True)
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "↓ Corrupted sample CSV",
                data=st.session_state["ds_corrupted"].to_csv(index=False).encode("utf-8"),
                file_name="datasentry_corrupted.csv",
                mime="text/csv",
            )
        with dl2:
            st.download_button(
                "↓ Synthetic labels CSV",
                data=st.session_state["ds_labels"].to_csv(index=False).encode("utf-8"),
                file_name="datasentry_synthetic_labels.csv",
                mime="text/csv",
            )
        st.warning("These labels mark injected missingness only. They do not label all pre-existing errors.")

st.markdown("---")
st.markdown(
    '<div style="display:flex;justify-content:space-between;color:#71819b;font-size:.75rem">'
    '<span>DATASENTRY ML · LOCAL DATA QUALITY WORKSPACE</span>'
    '<span>Starter UI · ML models are the next milestone</span></div>',
    unsafe_allow_html=True,
)
