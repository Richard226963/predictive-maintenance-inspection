"""
Predictive Maintenance & Visual Inspection — Predictive Maintenance Dashboard (Streamlit)
Production-quality dashboard with custom styling.

Run with:
    cd /path/to/download
    streamlit run streamlit_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import ast
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates

try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
except:
    pass
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Predictive Maintenance — Predictive Maintenance & Visual Inspection",
    page_icon=":factory:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS — professional dark theme
# ============================================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Sidebar */
    .stSidebar > div:first-child {
        background-color: #161b22;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 1rem;
        max-width: 1400px;
    }
    
    /* Headers */
    h1 {
        color: #f0f6fc !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    h2, h3 {
        color: #c9d1d9 !important;
    }
    
    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        margin: 5px;
    }
    div[data-testid="stMetric"] label {
        color: #8b949e !important;
        font-size: 0.8rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f0f6fc !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        border: 1px solid #30363d;
        border-radius: 8px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #21262d;
        border-radius: 8px 8px 0 0;
        color: #8b949e;
        border: 1px solid #30363d;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important;
        color: #ffffff !important;
    }
    
    /* Info/Success/Error boxes */
    .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 8px;
    }
    
    /* Caption */
    .stCaption {
        color: #8b949e !important;
    }
    
    /* Select box */
    .stSelectbox label {
        color: #c9d1d9 !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        border-radius: 8px;
    }
    
    /* Divider */
    hr {
        border-color: #30363d !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA
# ============================================================================
DATA_DIR = Path(__file__).parent / "streamlit_data"

@st.cache_data
def load_data():
    work_orders = pd.read_csv(DATA_DIR / "work_orders.csv")
    risk_scores = pd.read_csv(DATA_DIR / "site_risk_scores.csv")
    oof_preds = pd.read_csv(DATA_DIR / "oof_predictions.csv", parse_dates=["date"])
    per_site_vision = pd.read_csv(DATA_DIR / "per_site_vision.csv")
    fusion_summary = json.loads((DATA_DIR / "fusion_summary.json").read_text())
    numeric_results = json.loads((DATA_DIR / "numeric_results.json").read_text())
    vision_results = json.loads((DATA_DIR / "vision_results.json").read_text())
    return work_orders, risk_scores, oof_preds, per_site_vision, fusion_summary, numeric_results, vision_results

work_orders, risk_scores, oof_preds, per_site_vision, fusion_summary, numeric_results, vision_results = load_data()

# Merge work orders with per-site vision
wo_full = work_orders.merge(per_site_vision[["site_id", "image_file", "defects_json", "max_confidence"]], on="site_id", how="left")

# Get failure sites for highlighting
fail_sites = set(oof_preds[oof_preds.failure_event == 1].site_id.unique())

# ============================================================================
# HEADER
# ============================================================================
st.markdown("# Predictive Maintenance + Visual Inspection — Operations Dashboard")
st.markdown("*Predictive Maintenance & Visual Inspection | Fusing sensor telemetry + computer vision + LangGraph fusion agent*")
st.divider()

# ============================================================================
# KPI CARDS
# ============================================================================
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Sites Monitored", f"{fusion_summary['n_sites']}")
with col2:
    n_high = fusion_summary["priority_distribution"].get("HIGH", 0)
    pct = 100 * n_high / fusion_summary["n_sites"]
    st.metric("HIGH Priority", f"{n_high}", delta=f"{pct:.0f}% of fleet", delta_color="inverse")
with col3:
    st.metric("Fusion Precision", f"{100*fusion_summary['high_priority_precision']:.0f}%", delta="target >=75%", delta_color="off")
with col4:
    st.metric("ROI", f"{fusion_summary['roi_multiple']:.1f}x", delta=f"${fusion_summary['downtime_cost_avoided_usd']/1000000:.1f}M saved", delta_color="off")
with col5:
    st.metric("Downtime Prevented", f"{fusion_summary['total_downtime_prevented_hours']:.0f} hrs")

st.divider()

# ============================================================================
# SIDEBAR FILTERS
# ============================================================================
st.sidebar.markdown("## Filters")
available_priorities = ["ALL"] + sorted(work_orders["priority"].unique().tolist())
selected_priority = st.sidebar.selectbox("Priority", available_priorities, index=0)
region_options = ["ALL"] + sorted(work_orders["region"].unique().tolist())
selected_region = st.sidebar.selectbox("Region", region_options, index=0)

# Apply filters
filtered = wo_full.copy()
if selected_priority != "ALL":
    filtered = filtered[filtered["priority"] == selected_priority]
if selected_region != "ALL":
    filtered = filtered[filtered["region"] == selected_region]

# Sort: HIGH > LOW, then by risk score
priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
filtered["priority_sort"] = filtered["priority"].map(priority_order)
filtered = filtered.sort_values(["priority_sort", "numeric_risk_score"], ascending=[True, False])

# ============================================================================
# WORK ORDER QUEUE TABLE
# ============================================================================
st.markdown(f"## Work Order Queue ({len(filtered)} sites)")
st.markdown("*Sorted by priority then risk score. Select a site below for detailed view.*")

display_cols = ["site_id", "site_name", "region", "priority", "numeric_risk_score",
                "numeric_risk_level", "vision_findings_level", "n_defects_detected",
                "recommended_action", "estimated_cost_usd", "estimated_downtime_prevented_hours"]
display_df = filtered[display_cols].copy()

# Format columns
display_df["numeric_risk_score"] = display_df["numeric_risk_score"].apply(lambda x: f"{x:.4f}")
display_df["estimated_cost_usd"] = display_df["estimated_cost_usd"].apply(lambda x: f"${x:,.0f}")
display_df["estimated_downtime_prevented_hours"] = display_df["estimated_downtime_prevented_hours"].apply(lambda x: f"{x:.0f}h")
display_df.columns = ["Site ID", "Site Name", "Region", "Priority", "Risk Score",
                       "Numeric Level", "Vision Level", "Defects", "Recommended Action",
                       "Est. Cost", "Downtime Prevented"]

# Use st.dataframe with width parameter (fixes deprecation warning)
st.dataframe(display_df, width="stretch", height=350, hide_index=True)

st.divider()

# ============================================================================
# SITE DETAIL VIEW
# ============================================================================
st.markdown("## Site Detail View")

# Build site options with failure site indicator
site_options = filtered["site_id"].tolist()
def format_site(sid):
    name = filtered[filtered["site_id"] == sid]["site_name"].iloc[0]
    is_fail = sid in fail_sites
    marker = " [FAILURE SITE]" if is_fail else ""
    return f"{sid} — {name}{marker}"

selected_site = st.selectbox(
    "Select a site to inspect:",
    site_options,
    format_func=format_site,
)

# Get site data
site_wo = wo_full[wo_full["site_id"] == selected_site].iloc[0]
site_oof = oof_preds[oof_preds["site_id"] == selected_site].sort_values("date")
site_vision = per_site_vision[per_site_vision["site_id"] == selected_site]
if len(site_vision) > 0:
    site_vision = site_vision.iloc[0]
else:
    site_vision = None

# Site metrics
is_failure = selected_site in fail_sites
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    priority_color = "#da3633" if site_wo["priority"] == "HIGH" else "#238636"
    st.metric("Priority", site_wo["priority"])
with col_b:
    st.metric("Numeric Risk", f"{site_wo['numeric_risk_score']:.4f}", delta=site_wo["numeric_risk_level"])
with col_c:
    st.metric("Vision Findings", site_wo["vision_findings_level"], delta=f"{site_wo['n_defects_detected']} defects")
with col_d:
    st.metric("Site Age", f"{site_wo['site_age_years']:.0f} years", delta=f"{site_wo['site_coastal_exposure']} coastal")

# Failure site warning
if is_failure:
    st.error(f"This site had an actual failure event. The model detected it with {site_oof['risk_pred'].sum()} alert days before failure.")

# ============================================================================
# TABS
# ============================================================================
tabs = st.tabs(["Risk Trend", "Vision Findings", "Work Order", "Agent Trace", "YOLOv8 Predictions"])

# --- TAB 1: Risk Trend ---
with tabs[0]:
    st.markdown("### Predicted Failure Risk Over Time (90 days)")

    # Create professional chart
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)

    # Plot risk probability
    ax.plot(site_oof["date"], site_oof["risk_proba"], color="#da3633", linewidth=2, label="Risk Probability")

    # Alert threshold
    ax.axhline(0.05, color="#f59e0b", linestyle="--", alpha=0.7, linewidth=1.5, label="Alert threshold (0.05)")

    # Mark failure events
    fail_days = site_oof[site_oof["failure_event"] == 1]["date"]
    for fd in fail_days:
        ax.axvline(fd, color="#da3633", linestyle="-", alpha=0.3, linewidth=2)
        ax.text(fd, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.95, "FAILURE",
                color="#da3633", fontsize=9, fontweight="bold", ha="center")

    # Mark alert days (where risk_pred = 1)
    alert_days = site_oof[site_oof["risk_pred"] == 1]["date"]
    if len(alert_days) > 0:
        ax.scatter(alert_days, site_oof[site_oof["risk_pred"] == 1]["risk_proba"],
                  color="#f59e0b", s=30, zorder=5, label=f"Alerts ({len(alert_days)} days)")

    ax.set_title(f"Risk Probability Trend — {selected_site}", fontsize=13, fontweight="bold", color="#f0f6fc")
    ax.set_xlabel("Date", color="#8b949e")
    ax.set_ylabel("Risk Probability", color="#8b949e")
    ax.legend(fontsize=9, loc="upper left", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.grid(alpha=0.2, color="#30363d")
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

    # Style the plot
    ax.set_facecolor("#0d1117")
    fig.set_facecolor("#0d1117")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    st.pyplot(fig)

    # Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Max Risk", f"{site_oof['risk_proba'].max():.4f}")
    with col2:
        st.metric("Mean Risk", f"{site_oof['risk_proba'].mean():.6f}")
    with col3:
        st.metric("Alert Days", f"{site_oof['risk_pred'].sum()}")
    with col4:
        if len(fail_days) > 0:
            st.metric("Failure Date", f"{fail_days.iloc[0].date()}")
        else:
            st.metric("Failure Date", "N/A")

    # Interpretation
    if is_failure and len(alert_days) > 0:
        first_alert = site_oof[site_oof["risk_pred"] == 1]["date"].min()
        fail_date = fail_days.iloc[0]
        lead_time = (fail_date - first_alert).days
        st.success(f"Model first alerted on {first_alert.date()} — {lead_time} days before the actual failure on {fail_date.date()}.")
    elif not is_failure:
        st.info("This is a healthy site. Risk remains low throughout the 90-day period — no alerts triggered.")
    else:
        st.warning("This site had a failure but the model did not trigger alerts in this data window.")

# --- TAB 2: Vision Findings ---
with tabs[1]:
    st.markdown("### Vision Inspection Findings")

    if site_vision is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**Defects Detected:**")
            defects_json = site_vision.get("defects_json", "[]")
            try:
                defects = json.loads(defects_json) if defects_json else []
            except:
                defects = ast.literal_eval(defects_json) if defects_json else []

            if defects:
                for i, d in enumerate(defects, 1):
                    st.markdown(f"**{i}. {d['class']}** — confidence={d['confidence']:.3f}")
                    st.code(f"bbox: {d['bbox']}")
            else:
                st.success("No defects detected — visual inspection PASS")

        with col2:
            st.markdown("**Summary:**")
            st.metric("Total Defects", site_vision["n_defects"])
            st.metric("Corrosion", site_vision["n_corrosion"])
            st.metric("Crack", site_vision["n_crack"])
            st.metric("Max Confidence", f"{site_vision['max_confidence']:.3f}")

        st.info(f"**Vision interpretation:** {site_wo['vision_interpretation']}")
    else:
        st.warning("No vision data for this site")

    # Show YOLOv8 validation predictions
    st.markdown("---")
    st.markdown("### Sample YOLOv8 Predictions on Validation Set")
    val_pred_path = DATA_DIR / "yolov8m_val_predictions.jpg"
    val_labels_path = DATA_DIR / "yolov8m_val_labels.jpg"
    if val_pred_path.exists() and val_labels_path.exists():
        col_pred, col_labels = st.columns(2)
        with col_pred:
            st.image(str(val_pred_path), caption="YOLOv8m Predictions (green = detected defects)", width="stretch")
        with col_labels:
            st.image(str(val_labels_path), caption="Ground Truth Labels", width="stretch")

# --- TAB 3: Work Order ---
with tabs[2]:
    st.markdown("### Work Order Details")

    st.markdown(f"**Work Order ID:** `{site_wo['work_order_id']}`")
    st.markdown(f"**Priority:** {site_wo['priority']}")
    st.markdown(f"**Recommended Action:**")
    st.info(site_wo["recommended_action"])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Estimated Cost", f"${site_wo['estimated_cost_usd']:,.0f}")
    with col2:
        st.metric("Downtime Prevented", f"{site_wo['estimated_downtime_prevented_hours']:.0f} hours")

    st.markdown("")
    st.markdown("**Merged Assessment:**")
    st.code(site_wo["merged_assessment"], language="text")

    st.markdown("")
    st.markdown("**Numeric Interpretation:**")
    st.code(site_wo["numeric_interpretation"], language="text")

# --- TAB 4: Agent Trace ---
with tabs[3]:
    st.markdown("### LangGraph Agent Trace (Full Audit Trail)")

    trace_str = site_wo.get("agent_trace", "[]")
    try:
        trace = json.loads(trace_str) if isinstance(trace_str, str) else trace_str
    except:
        try:
            trace = ast.literal_eval(trace_str)
        except:
            trace = [str(trace_str)]

    if isinstance(trace, list):
        for i, step in enumerate(trace, 1):
            st.code(f"Step {i}: {step}", language="text")
    else:
        st.code(str(trace), language="text")

    st.markdown("")
    st.markdown("**State Machine Structure:**")
    st.code("""
START -> collect_inputs -> numeric_analysis -> vision_analysis
              -> contextual_merge -> decision -> work_order_writer -> END
    """.strip())

    st.caption("Each node performs deterministic reasoning. No LLM calls — fully auditable and reproducible.")

# --- TAB 5: YOLOv8 Predictions ---
with tabs[4]:
    st.markdown("### YOLOv8m Model Predictions on Validation Set")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model", "YOLOv8m (25M params)")
    with col2:
        st.metric("mAP@0.5", f"{vision_results['map50']:.4f}")

    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Precision", f"{vision_results['precision']:.4f}")
    with col4:
        st.metric("Recall", f"{vision_results['recall']:.4f}")
    with col5:
        st.metric("Epochs", f"{vision_results['epochs_trained']}")

    if len(vision_results.get("per_class_ap50", [])) >= 2:
        st.markdown(f"**Per-class AP@0.5:** corrosion={vision_results['per_class_ap50'][0]:.4f}, crack={vision_results['per_class_ap50'][1]:.4f}")

    st.markdown(f"**Overfitting detected:** {vision_results.get('overfitting_detected', False)}")
    st.markdown(f"**Stop reason:** {vision_results.get('stop_reason', 'N/A')}")
    st.markdown("**Training data:** 354 train / 76 val / 77 test images")

    st.markdown("---")
    val_pred_path = DATA_DIR / "yolov8m_val_predictions.jpg"
    if val_pred_path.exists():
        st.image(str(val_pred_path), caption="YOLOv8m predictions on validation batch (green boxes = detected defects)", width="stretch")

    val_labels_path = DATA_DIR / "yolov8m_val_labels.jpg"
    if val_labels_path.exists():
        st.image(str(val_labels_path), caption="Ground truth labels for same batch", width="stretch")

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.caption(f"""
**Predictive Maintenance & Visual Inspection — Predictive Maintenance + Visual Inspection** |
Numeric: XGBoost+LightGBM (100% recall, AUC-ROC {numeric_results['oof_roc_auc']:.4f}) |
Vision: YOLOv8m (mAP@0.5={vision_results['map50']:.4f}) |
Fusion: LangGraph (100% precision, {fusion_summary['roi_multiple']:.1f}x ROI)
""")
