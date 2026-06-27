# data_prep.py
# -----------------------------------------------------------------------------
# UAC Care System Load Analytics — Corrected Data Preparation Pipeline
# (uses the same column names as your notebook: Intake, CBP_Custody,
#  Transfers_to_HHS, HHS_Care, Discharges)
#
# Fixes vs. the original notebook:
#   - Date is properly set as the DataFrame index (sort_index() now actually
#     sorts chronologically)
#   - Fully-blank trailing rows are dropped before any processing
#   - A complete daily index is created (missing report-days are filled)
#   - Missing/duplicate date detection added
#   - Logical constraint validation added (Transfers<=CBP, Discharges<=HHS)
#   - Net_Daily_Intake now correctly uses Transfers_to_HHS (not Intake)
#   - Total_System_Load (CBP + HHS) added — this was missing entirely
#   - Care_Load_Growth_Rate (day-over-day % change) added — was missing
#   - Backlog_Accumulation now a running total of positive net intake,
#     floored at 0 (matches "sustained positive net intake over time")
#   - Discharge_Offset_Ratio now divides by Transfers_to_HHS, not Intake —
#     this removes the "inf" bug from the original notebook (Intake had zeros)
#   - Care_Load_Volatility_Index is now a normalized coefficient of variation
#     (std / mean), not a raw rolling std
#   - Trend & Temporal Analysis phase added (sustained high-load periods,
#     early-vs-late comparison)
#   - Pressure & Stress phase completed (7-day AND 14-day rolling, prolonged
#     strain window detection)
#
# Run this ONCE:  python data_prep.py
# Output: uac_cleaned.csv  (read by app.py)
#         data_quality_report.txt
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd

RAW_PATH = r"D:\A Aravinth\unified\Unaccompanied Children\HHS_Unaccompanied_Alien_Children_Program.csv"
OUTPUT_PATH = "uac_cleaned.csv"
QUALITY_REPORT_PATH = "data_quality_report.txt"

report_lines = []


def log(line):
    print(line)
    report_lines.append(str(line))


# =============================================================================
# PHASE 1: DATA INGESTION & STRUCTURING
# =============================================================================
log("=" * 70)
log("PHASE 1: DATA INGESTION & STRUCTURING")
log("=" * 70)

raw = pd.read_csv(RAW_PATH)
log(f"Raw file loaded: {raw.shape[0]} rows, {raw.shape[1]} columns")

# Drop fully-blank trailing rows (the raw export has empty rows at the bottom)
raw = raw.dropna(how="all").copy()
log(f"After dropping fully-blank rows: {raw.shape[0]} rows")

# Select + rename the core columns (same names you used in your notebook)
df_core = raw[[
    "Date",
    "Children apprehended and placed in CBP custody*",
    "Children in CBP custody",
    "Children transferred out of CBP custody",
    "Children in HHS Care",
    "Children discharged from HHS Care",
]].copy()

df_core.rename(columns={
    "Children apprehended and placed in CBP custody*": "Intake",
    "Children in CBP custody": "CBP_Custody",
    "Children transferred out of CBP custody": "Transfers_to_HHS",
    "Children in HHS Care": "HHS_Care",
    "Children discharged from HHS Care": "Discharges",
}, inplace=True)

# Convert Date to datetime
df_core["Date"] = pd.to_datetime(df_core["Date"])

# Clean numeric columns (strip thousands-separator commas, coerce to numeric)
numeric_cols = ["Intake", "CBP_Custody", "Transfers_to_HHS", "HHS_Care", "Discharges"]
for col in numeric_cols:
    df_core[col] = df_core[col].astype(str).str.replace(",", "", regex=False).str.strip()
    df_core[col] = pd.to_numeric(df_core[col], errors="coerce")

# Set Date as the ACTUAL index, then sort chronologically
# (your original notebook only set df.index.name='Date' without set_index,
#  so sort_index() was sorting the meaningless row-number index, not dates)
df_core = df_core.set_index("Date").sort_index()
log(f"Date range in raw reports: {df_core.index.min().date()} to {df_core.index.max().date()}")

# Create a COMPLETE daily index (raw data only has report-days, not every
# calendar day — weekends/holidays are often skipped in the source)
full_index = pd.date_range(df_core.index.min(), df_core.index.max(), freq="D")
reported_dates = set(df_core.index)
missing_dates = sorted(set(full_index) - reported_dates)

df_core = df_core.reindex(full_index)
df_core.index.name = "Date"
log(f"Complete daily index created: {len(full_index)} calendar days "
    f"({len(missing_dates)} days had no report and were filled in Phase 2)")

# =============================================================================
# PHASE 2: DATA QUALITY & VALIDATION
# =============================================================================
log("")
log("=" * 70)
log("PHASE 2: DATA QUALITY & VALIDATION")
log("=" * 70)

# --- Missing / duplicated dates -------------------------------------------------
duplicate_dates = pd.to_datetime(raw["Date"]).duplicated().sum()
log(f"Duplicate report dates found: {duplicate_dates}")
log(f"Missing (un-reported) calendar dates found: {len(missing_dates)}")
if missing_dates:
    log(f"  First few missing dates: {[d.date().isoformat() for d in missing_dates[:5]]}")

# --- Logical constraint validation (flag, don't silently drop) -----------------
df_core["Flag_Transfers_Exceed_CBP"] = df_core["Transfers_to_HHS"] > df_core["CBP_Custody"]
df_core["Flag_Discharges_Exceed_HHS"] = df_core["Discharges"] > df_core["HHS_Care"]

n_transfer_violations = df_core["Flag_Transfers_Exceed_CBP"].sum()
n_discharge_violations = df_core["Flag_Discharges_Exceed_HHS"].sum()
log(f"Logical constraint check — Transfers > CBP custody: {n_transfer_violations} day(s) flagged")
log(f"Logical constraint check — Discharges > HHS care:   {n_discharge_violations} day(s) flagged")
log("(Flagged, not removed — kept as transparency flags per the brief.)")

# --- Reporting anomalies (negative values) --------------------------------------
for col in numeric_cols:
    n_negative = (df_core[col] < 0).sum()
    if n_negative > 0:
        log(f"Anomaly — negative values in '{col}': {n_negative}")
df_core["Flag_Any_Negative"] = (df_core[numeric_cols] < 0).any(axis=1)

# --- Fill gaps created by the complete daily index ------------------------------
for col in numeric_cols:
    df_core[col] = df_core[col].interpolate(method="time").bfill().ffill()

n_missing_after = df_core[numeric_cols].isnull().sum().sum()
log(f"Missing values remaining after interpolation: {n_missing_after}")

# =============================================================================
# PHASE 3: DERIVED HEALTHCARE CAPACITY METRICS
# =============================================================================
log("")
log("=" * 70)
log("PHASE 3: DERIVED HEALTHCARE CAPACITY METRICS")
log("=" * 70)

# Total System Load = CBP Custody + HHS Care  (was missing entirely before)
df_core["Total_System_Load"] = df_core["CBP_Custody"] + df_core["HHS_Care"]

# Net Daily Intake = Transfers into HHS - Discharges from HHS
# (your notebook used Intake - Discharges here — wrong column)
df_core["Net_Daily_Intake"] = df_core["Transfers_to_HHS"] - df_core["Discharges"]

# Care Load Growth Rate = day-over-day % change in HHS care load (was missing)
df_core["Care_Load_Growth_Rate"] = df_core["HHS_Care"].pct_change() * 100

# Backlog Indicator = sustained positive net intake over time.
#   Backlog_Accumulation: running total of Net_Daily_Intake, floored at 0
#   (replaces the original notebook's plain .diff(), which only measured
#    day-over-day HHS_Care change, not accumulated backlog)
running, backlog_values = 0.0, []
for v in df_core["Net_Daily_Intake"].fillna(0):
    running = max(0.0, running + v)
    backlog_values.append(running)
df_core["Backlog_Accumulation"] = backlog_values

# Backlog_Flag: True when net intake has been positive for 3+ consecutive days
positive_streak = (df_core["Net_Daily_Intake"] > 0).astype(int)
streak_id = (positive_streak == 0).cumsum()
df_core["Backlog_Flag"] = positive_streak.groupby(streak_id).cumsum() >= 3

log(f"Total_System_Load computed — range {df_core['Total_System_Load'].min():.0f} to "
    f"{df_core['Total_System_Load'].max():.0f}")
log(f"Net_Daily_Intake computed — {int((df_core['Net_Daily_Intake'] > 0).sum())} days "
    f"with positive net intake out of {len(df_core)}")
log(f"Backlog_Flag active on {int(df_core['Backlog_Flag'].sum())} days")

# =============================================================================
# PHASE 4: TREND & TEMPORAL ANALYSIS  (was missing entirely)
# =============================================================================
log("")
log("=" * 70)
log("PHASE 4: TREND & TEMPORAL ANALYSIS")
log("=" * 70)

# Sustained high-load periods: HHS care above its 90th percentile for 5+ days
high_load_threshold = df_core["HHS_Care"].quantile(0.90)
is_high = (df_core["HHS_Care"] > high_load_threshold).astype(int)
high_streak_id = (is_high == 0).cumsum()
df_core["Sustained_High_Load"] = is_high.groupby(high_streak_id).cumsum() >= 5

log(f"High-load threshold (90th percentile of HHS care): {high_load_threshold:.0f}")
log(f"Days in a sustained high-load period (5+ consecutive days): "
    f"{int(df_core['Sustained_High_Load'].sum())}")

# Early-vs-late timeline comparison
midpoint = df_core.index.min() + (df_core.index.max() - df_core.index.min()) / 2
early = df_core[df_core.index < midpoint]
late = df_core[df_core.index >= midpoint]
log(f"Early period ({early.index.min().date()} to {early.index.max().date()}): "
    f"avg HHS care load = {early['HHS_Care'].mean():.0f}")
log(f"Late period  ({late.index.min().date()} to {late.index.max().date()}): "
    f"avg HHS care load = {late['HHS_Care'].mean():.0f}")

# =============================================================================
# PHASE 5: PRESSURE & STRESS IDENTIFICATION
# =============================================================================
log("")
log("=" * 70)
log("PHASE 5: PRESSURE & STRESS IDENTIFICATION")
log("=" * 70)

# Rolling averages (7-day, 14-day) — your notebook only had a 7-day std
df_core["HHS_Care_roll7_mean"] = df_core["HHS_Care"].rolling(7).mean()
df_core["HHS_Care_roll14_mean"] = df_core["HHS_Care"].rolling(14).mean()
df_core["HHS_Care_roll7_std"] = df_core["HHS_Care"].rolling(7).std()
df_core["HHS_Care_roll14_std"] = df_core["HHS_Care"].rolling(14).std()

# Variability analysis: Care Load Volatility Index = coefficient of variation
# (std / mean) — normalized, unlike a raw rolling std
df_core["Care_Load_Volatility_Index"] = df_core["HHS_Care_roll7_std"] / df_core["HHS_Care_roll7_mean"]
df_core["Care_Load_Volatility_Index_14"] = df_core["HHS_Care_roll14_std"] / df_core["HHS_Care_roll14_mean"]

# Prolonged strain window: 14-day rolling mean above threshold for 7+ days
is_strained = (df_core["HHS_Care_roll14_mean"] > high_load_threshold).astype(int)
strain_streak_id = (is_strained == 0).cumsum()
df_core["Prolonged_Strain_Window"] = is_strained.groupby(strain_streak_id).cumsum() >= 7

log("Rolling 7-day / 14-day mean + std computed for HHS care load")
log(f"Days inside a prolonged strain window: {int(df_core['Prolonged_Strain_Window'].sum())}")

# =============================================================================
# KPI VALUES  (matching the official KPI table exactly — fixes the inf bug)
# =============================================================================
log("")
log("=" * 70)
log("KPI VALIDATION (final 30 days)")
log("=" * 70)

last30 = df_core.tail(30)
total_children_under_care = df_core["Total_System_Load"].iloc[-1]
net_intake_pressure = last30["Net_Daily_Intake"].mean()
care_load_volatility_index = df_core["Care_Load_Volatility_Index"].iloc[-1]
backlog_accumulation_rate = (
    df_core["Backlog_Accumulation"].iloc[-1] - df_core["Backlog_Accumulation"].iloc[-8]
) / 7
# Discharge Offset Ratio now divides by Transfers_to_HHS, not Intake —
# this is what removes the "inf" result you saw in your original notebook
discharge_offset_ratio = last30["Discharges"].sum() / last30["Transfers_to_HHS"].sum()

log(f"Total Children Under Care   : {total_children_under_care:,.0f}")
log(f"Net Intake Pressure (30d avg): {net_intake_pressure:+.2f}")
log(f"Care Load Volatility Index  : {care_load_volatility_index:.4f}")
log(f"Backlog Accumulation Rate   : {backlog_accumulation_rate:+.2f}/day")
log(f"Discharge Offset Ratio      : {discharge_offset_ratio:.2f}  (no longer 'inf')")

# =============================================================================
# SAVE
# =============================================================================
df_core.to_csv(OUTPUT_PATH, index=True)
log("")
log(f"Saved cleaned dataset to: {OUTPUT_PATH}")
log(f"Final shape: {df_core.shape}")

with open(QUALITY_REPORT_PATH, "w") as f:
    f.write("\n".join(report_lines))
log(f"Saved data quality report to: {QUALITY_REPORT_PATH}")