# app.py
# -----------------------------------------------------------------------------
# UAC Care System Load Analytics — Streamlit Dashboard
#
# IMPORTANT: Run data_prep.py FIRST. This app reads "uac_cleaned.csv",
# which that script generates.
#
# Run with:  streamlit run app.py
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="UAC Care System Load Analytics", layout="wide")
st.title("UAC Care System Load Analytics Dashboard")
st.caption("Capacity awareness & care load monitoring for the Unaccompanied Alien Children (UAC) Program")

DATA_PATH = "uac_cleaned.csv"

# ----------------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------------
@st.cache_data
def load_data(path):
    data = pd.read_csv(path, index_col=0, parse_dates=True)
    data.index.name = "Date"
    return data

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Could not find '{DATA_PATH}' in this folder.\n\n"
        f"Run **data_prep.py** first, then restart this app."
    )
    st.stop()

required_cols = [
    "CBP_Custody", "HHS_Care", "Transfers_to_HHS", "Discharges", "Total_System_Load",
    "Net_Daily_Intake", "Backlog_Accumulation", "Backlog_Flag",
    "HHS_Care_roll7_mean", "HHS_Care_roll14_mean",
    "Care_Load_Volatility_Index", "Care_Load_Volatility_Index_14",
    "Sustained_High_Load", "Prolonged_Strain_Window",
]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(f"Processed CSV is missing required columns: {missing_cols}. Re-run data_prep.py.")
    st.stop()

# ----------------------------------------------------------------------------
# SIDEBAR — USER CAPABILITIES
# ----------------------------------------------------------------------------
st.sidebar.header("Filters")

# 1) Date range selector
min_date, max_date = df.index.min().date(), df.index.max().date()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# 2) Time granularity filter
granularity = st.sidebar.selectbox("Time granularity", ["Daily", "Weekly", "Monthly"])

# 3) Metric toggles
st.sidebar.subheader("Metric toggles")
show_cbp = st.sidebar.checkbox("Show CBP custody load", value=True)
show_hhs = st.sidebar.checkbox("Show HHS care load", value=True)
show_rolling = st.sidebar.checkbox("Show 7-day / 14-day rolling averages", value=True)
show_flags = st.sidebar.checkbox("Highlight sustained backlog/strain periods", value=True)

# ----------------------------------------------------------------------------
# APPLY DATE RANGE FILTER
# ----------------------------------------------------------------------------
mask = (df.index.date >= start_date) & (df.index.date <= end_date)
filtered = df.loc[mask].copy()

if filtered.empty:
    st.warning("No data in the selected date range. Adjust the date filter in the sidebar.")
    st.stop()


# ----------------------------------------------------------------------------
# APPLY TIME GRANULARITY (resample stocks as mean, flows as sum)
# ----------------------------------------------------------------------------
def resample_for_view(data, granularity):
    if granularity == "Daily":
        return data
    rule = "W" if granularity == "Weekly" else "ME"
    agg = {
        "CBP_Custody": "mean",
        "HHS_Care": "mean",
        "Transfers_to_HHS": "sum",
        "Discharges": "sum",
        "Total_System_Load": "mean",
        "Net_Daily_Intake": "sum",
        "Backlog_Accumulation": "last",
        "HHS_Care_roll7_mean": "mean",
        "HHS_Care_roll14_mean": "mean",
        "Care_Load_Volatility_Index": "mean",
        "Care_Load_Volatility_Index_14": "mean",
        "Sustained_High_Load": "max",
        "Backlog_Flag": "max",
        "Prolonged_Strain_Window": "max",
    }
    agg = {k: v for k, v in agg.items() if k in data.columns}
    return data.resample(rule).agg(agg)


view = resample_for_view(filtered, granularity)

# ----------------------------------------------------------------------------
# KPI SUMMARY CARDS  (Core Module: KPI Summary Cards)
# ----------------------------------------------------------------------------
st.subheader("KPI Summary Cards")

latest = filtered.iloc[-1]
last7 = filtered.tail(7)

total_children_under_care = latest["Total_System_Load"]
net_intake_pressure = last7["Net_Daily_Intake"].mean()
care_load_volatility_index = latest["Care_Load_Volatility_Index"]

if len(filtered) >= 8:
    backlog_accumulation_rate = (
        filtered["Backlog_Accumulation"].iloc[-1] - filtered["Backlog_Accumulation"].iloc[-8]
    ) / 7
else:
    backlog_accumulation_rate = np.nan

transferred_7 = last7["Transfers_to_HHS"].sum()
discharged_7 = last7["Discharges"].sum()
discharge_offset_ratio = discharged_7 / transferred_7 if transferred_7 > 0 else np.nan

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Children Under Care", f"{total_children_under_care:,.0f}")
k2.metric(
    "Net Intake Pressure (7d avg)",
    f"{net_intake_pressure:+.1f}",
    help="Transfers into HHS minus Discharges, averaged over the last 7 days. "
         "Positive = inflow exceeding outflow (building pressure)."
)
k3.metric(
    "Care Load Volatility Index",
    f"{care_load_volatility_index:.3f}" if pd.notna(care_load_volatility_index) else "N/A",
    help="7-day coefficient of variation (std/mean) of HHS care load. Higher = less stable."
)
k4.metric(
    "Backlog Accumulation Rate",
    f"{backlog_accumulation_rate:+.1f}/day" if pd.notna(backlog_accumulation_rate) else "N/A",
    help="Average daily change in accumulated backlog over the last 7 days."
)
k5.metric(
    "Discharge Offset Ratio",
    f"{discharge_offset_ratio:.2f}" if pd.notna(discharge_offset_ratio) else "N/A",
    help="Discharges \u00f7 Transfers-to-HHS over the last 7 days. 1.0 = fully offsetting inflow."
)

st.divider()

# ----------------------------------------------------------------------------
# TABS — remaining 3 Core Modules
# ----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "System Load Overview Pane",
    "CBP vs HHS Load Comparison",
    "Net Intake & Backlog Trends",
])

# --- Module: System Load Overview Pane --------------------------------------
with tab1:
    st.markdown(f"**Total System Load over time** ({granularity} view)")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=view.index, y=view["Total_System_Load"], name="Total System Load",
                               line=dict(color="#1f77b4", width=2)))
    if show_rolling:
        fig1.add_trace(go.Scatter(x=view.index, y=view["HHS_Care_roll7_mean"], name="HHS Care 7-day avg",
                                   line=dict(color="#ff7f0e", dash="dot")))
        fig1.add_trace(go.Scatter(x=view.index, y=view["HHS_Care_roll14_mean"], name="HHS Care 14-day avg",
                                   line=dict(color="#2ca02c", dash="dash")))
    if show_flags:
        high_periods = view[view["Sustained_High_Load"].astype(bool)]
        if not high_periods.empty:
            fig1.add_trace(go.Scatter(x=high_periods.index, y=high_periods["Total_System_Load"],
                                       mode="markers", name="Sustained high-load period",
                                       marker=dict(color="red", size=6)))
    fig1.update_layout(xaxis_title="Date", yaxis_title="Children", hovermode="x unified")
    st.plotly_chart(fig1, width="stretch")

    st.markdown("**Calendar comparison \u2014 early vs. late period**")
    midpoint = filtered.index.min() + (filtered.index.max() - filtered.index.min()) / 2
    early = filtered[filtered.index < midpoint]
    late = filtered[filtered.index >= midpoint]
    comp_col1, comp_col2 = st.columns(2)
    comp_col1.metric("Avg HHS Care \u2014 Early period",
                      f"{early['HHS_Care'].mean():,.0f}" if not early.empty else "N/A")
    comp_col2.metric("Avg HHS Care \u2014 Late period",
                      f"{late['HHS_Care'].mean():,.0f}" if not late.empty else "N/A")

# --- Module: CBP vs HHS Load Comparison -------------------------------------
with tab2:
    st.markdown(f"**CBP custody load vs. HHS care load** ({granularity} view)")
    fig2 = go.Figure()
    if show_cbp:
        fig2.add_trace(go.Scatter(x=view.index, y=view["CBP_Custody"], name="Children in CBP custody",
                                   line=dict(color="#d62728")))
    if show_hhs:
        fig2.add_trace(go.Scatter(x=view.index, y=view["HHS_Care"], name="Children in HHS care",
                                   line=dict(color="#1f77b4")))
    fig2.update_layout(xaxis_title="Date", yaxis_title="Children", hovermode="x unified")
    st.plotly_chart(fig2, width="stretch")

    c1, c2 = st.columns(2)
    c1.metric("Average CBP custody load (selected range)", f"{view['CBP_Custody'].mean():,.0f}")
    c2.metric("Average HHS care load (selected range)", f"{view['HHS_Care'].mean():,.0f}")

# --- Module: Net Intake & Backlog Trends ------------------------------------
with tab3:
    st.markdown(f"**Net daily intake & accumulated backlog** ({granularity} view)")
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=view.index, y=view["Net_Daily_Intake"], name="Net Daily Intake",
                           marker_color=np.where(view["Net_Daily_Intake"] >= 0, "#d62728", "#2ca02c")))
    fig3.update_layout(xaxis_title="Date", yaxis_title="Net Daily Intake (Transfers \u2212 Discharges)",
                        hovermode="x unified")
    st.plotly_chart(fig3, width="stretch")

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=view.index, y=view["Backlog_Accumulation"], name="Backlog Accumulation",
                               fill="tozeroy", line=dict(color="#9467bd")))
    if show_flags:
        strain_periods = view[view["Prolonged_Strain_Window"].astype(bool)]
        if not strain_periods.empty:
            fig4.add_trace(go.Scatter(x=strain_periods.index, y=strain_periods["Backlog_Accumulation"],
                                       mode="markers", name="Prolonged strain window",
                                       marker=dict(color="red", size=6)))
    fig4.update_layout(xaxis_title="Date", yaxis_title="Accumulated Backlog", hovermode="x unified")
    st.plotly_chart(fig4, width="stretch")

st.divider()

# ----------------------------------------------------------------------------
# DATASET PREVIEW
# ----------------------------------------------------------------------------
st.subheader("Dataset Preview (filtered range, most recent 20 rows)")
st.dataframe(filtered.tail(20), width="stretch")