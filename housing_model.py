"""
Housing Structural Risk Model (Index-Based, Outcome Neutral)

Key design constraints implemented:
- No absolute currency values are computed, displayed, or exported.
- All “financial” outputs are indices (Base Year 2025 = 100) plus DSCR-derived safety margin.
- Default settings are outcome-neutral (Flat debt profile), so risk only materializes when
  the user selects the "Cliff" debt shape or otherwise introduces stress.

Run:
    streamlit run uw_housing_model.py
"""

from __future__ import annotations

from io import StringIO
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Altair is included in most Streamlit installs, but we keep a fallback path
try:
    import altair as alt

    HAS_ALTAIR = True
except Exception:
    HAS_ALTAIR = False


# -----------------------------
# Hardcoded reconciled inputs
# -----------------------------
BASE_YEAR = 2025
END_YEAR = 2045
YEARS = list(range(BASE_YEAR, END_YEAR + 1))
TARGET_YEAR_TORNADO = 2035  # per requirements

RECONCILED_DATA = {
    "_metadata": {
        "reconciliation_date": "2025-12-15",
        "note": "Financial values here are for BASELINE INDEXING only.",
    },
    "housing_portfolio": {
        "totals": {
            "total_operating_capacity": {"value": 9114},
            "overflow_beds": {"value": 879},
        },
        "occupancy": {
            "current_rate": {"value": 1.004, "year": "2025-26"},
            "current_headcount": {"value": 9149},
        },
        "planned_changes": {
            "haggett_hall_replacement": {
                "status": "under_construction",
                "projected_opening": "Fall 2027",
                "planned_beds": 800,
            }
        },
    },
    "demographics": {
        "source_note": "WA OFM 18-year-old Population Projections",
        "wa_18yo_population": [
            {"year": 2025, "population": 101845},
            {"year": 2026, "population": 103205},
            {"year": 2027, "population": 104050},
            {"year": 2028, "population": 104473},
            {"year": 2029, "population": 104553},
            {"year": 2030, "population": 104571},
            {"year": 2031, "population": 104324},
            {"year": 2032, "population": 103789},
            {"year": 2033, "population": 102995},
            {"year": 2034, "population": 101993},
            {"year": 2035, "population": 101521},
            {"year": 2036, "population": 101513},
            {"year": 2037, "population": 101908},
            {"year": 2038, "population": 102390},
            {"year": 2039, "population": 97490},
            {"year": 2040, "population": 99750},
            {"year": 2041, "population": 97327},
            {"year": 2042, "population": 96030},
            {"year": 2043, "population": 97471},
            {"year": 2044, "population": 98124},
            {"year": 2045, "population": 98770},
        ],
    },
    "financial_ratios": {
        "base_dscr": {"value": 1.57, "note": "2022 Actual, serves as Index anchor"},
        "required_dscr": {"value": 1.25},
        "debt_service_share": {
            "value": 0.35,
            "note": "Est. debt service as % of base-year indexed revenue",
        },
        "expense_share": {
            "value": 0.50,
            "note": "Est. operating expense as % of base-year indexed revenue",
        },
        "margin_share": {"value": 0.15, "note": "Net margin share (context only)"},
    },
}

WA_OFM_18YO = [(x["year"], x["population"]) for x in RECONCILED_DATA["demographics"]["wa_18yo_population"]]

DEBT_SHAPES = [
    "Flat (Baseline)",
    "Front-Loaded",
    'The "Cliff" (Risk)',
    "Custom",
]

SCENARIOS: Dict[str, Dict[str, object]] = {
    # Outcome-neutral default: flat debt, matched inflation and escalation, no behavioral headwind.
    "Baseline": {
        "debt_shape": "Flat (Baseline)",
        "rate_escalation_pct": 2.5,
        "expense_inflation_pct": 2.5,
        "national_trend_pct_by_2035": 0,
        "wa_demand_share": 0.70,
        "behavior_headwind_pct_by_2035": 0,
        "haggett_net_beds": 0,
        "expense_share_pct": int(RECONCILED_DATA["financial_ratios"]["expense_share"]["value"] * 100),
        "debt_share_pct": int(RECONCILED_DATA["financial_ratios"]["debt_service_share"]["value"] * 100),
        "custom_peak_multiplier": 1.20,
        "custom_peak_year": 2030,
    },
    "Structural Squeeze": {
        "debt_shape": 'The "Cliff" (Risk)',
        "rate_escalation_pct": 3.0,
        "expense_inflation_pct": 4.0,
        "national_trend_pct_by_2035": 0,
        "wa_demand_share": 0.70,
        "behavior_headwind_pct_by_2035": 0,
        "haggett_net_beds": 0,
        "expense_share_pct": int(RECONCILED_DATA["financial_ratios"]["expense_share"]["value"] * 100),
        "debt_share_pct": int(RECONCILED_DATA["financial_ratios"]["debt_service_share"]["value"] * 100),
        "custom_peak_multiplier": 1.20,
        "custom_peak_year": 2030,
    },
    "Demographic Trough": {
        "debt_shape": "Flat (Baseline)",
        "rate_escalation_pct": 2.5,
        "expense_inflation_pct": 2.5,
        "national_trend_pct_by_2035": -10,
        "wa_demand_share": 0.70,
        "behavior_headwind_pct_by_2035": 0,
        "haggett_net_beds": 0,
        "expense_share_pct": int(RECONCILED_DATA["financial_ratios"]["expense_share"]["value"] * 100),
        "debt_share_pct": int(RECONCILED_DATA["financial_ratios"]["debt_service_share"]["value"] * 100),
        "custom_peak_multiplier": 1.20,
        "custom_peak_year": 2030,
    },
    # Leaves whatever the user has set.
    "Custom (Keep Current Settings)": {},
}


# -----------------------------
# Numerical safety helpers
# -----------------------------
def safe_div(n, d, default=np.nan):
    """
    Division that never throws divide-by-zero, and never returns inf/-inf.
    Works for scalars, numpy arrays, and pandas Series.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(n, d)

    if np.isscalar(out):
        return out if np.isfinite(out) else default

    out = np.where(np.isfinite(out), out, default)
    return out


def clamp(x, lo=None, hi=None):
    lo = -np.inf if lo is None else lo
    hi = np.inf if hi is None else hi
    return np.clip(x, lo, hi)


def linear_progress(years, start_year, end_year):
    """
    0 at start_year, 1 at end_year, flat (clamped) outside the window.
    Used to implement "by 2035" sliders without extrapolating beyond the chosen horizon.
    """
    years = np.asarray(years, dtype=float)
    denom = max(1.0, float(end_year - start_year))
    prog = (years - float(start_year)) / denom
    return clamp(prog, 0.0, 1.0)


# -----------------------------
# Demand and debt profile builders
# -----------------------------
def build_ofm_df() -> pd.DataFrame:
    df = pd.DataFrame(WA_OFM_18YO, columns=["year", "wa_18yo_population"]).copy()
    df = df[(df["year"] >= BASE_YEAR) & (df["year"] <= END_YEAR)].sort_values("year")
    base_pop = float(df.loc[df["year"] == BASE_YEAR, "wa_18yo_population"].iloc[0])
    df["wa_18yo_index"] = safe_div(df["wa_18yo_population"], base_pop, default=0.0) * 100.0
    return df.reset_index(drop=True)


def build_national_index(years, pct_by_2035: float) -> np.ndarray:
    """
    National/Global HS Grad Trend slider is interpreted as a level change by 2035,
    relative to 2025, for non-resident demand (macro environment proxy).

    Example:
    -10% means non-resident pipeline index moves from 100 in 2025 to 90 in 2035,
    then holds at 90 through the remainder of the horizon (no extrapolation).
    """
    prog = linear_progress(years, BASE_YEAR, 2035)
    idx = 100.0 * (1.0 + pct_by_2035 * prog)
    return clamp(idx, 0.0, None)


def build_behavior_index(years, pct_by_2035: float) -> np.ndarray:
    """
    Optional, outcome-neutral control: broad behavioral headwind/tailwind on housing demand.
    Interpreted as a level change by 2035 relative to 2025, then held constant.
    """
    prog = linear_progress(years, BASE_YEAR, 2035)
    idx = 100.0 * (1.0 + pct_by_2035 * prog)
    return clamp(idx, 0.0, None)


def build_capacity_headcount(years, haggett_net_beds: int) -> np.ndarray:
    """
    Capacity is used only as a physical cap on occupancy headcount, not as a revenue driver.
    This avoids the common mistake of turning capacity expansion into automatic revenue.

    Baseline cap uses operating capacity + overflow beds (both non-dollar, operational facts).
    Haggett replacement is modeled as a net bed change effective in 2027 (Fall 2027 opening).
    """
    base_operating = int(RECONCILED_DATA["housing_portfolio"]["totals"]["total_operating_capacity"]["value"])
    overflow = int(RECONCILED_DATA["housing_portfolio"]["totals"]["overflow_beds"]["value"])
    cap_base = float(base_operating + overflow)

    cap = np.full(len(years), cap_base, dtype=float)
    cap = np.where(np.asarray(years) >= 2027, cap + float(haggett_net_beds), cap)
    return clamp(cap, 0.0, None)


def build_debt_index(
    years: np.ndarray,
    shape: str,
    custom_peak_multiplier: float,
    custom_peak_year: int,
) -> np.ndarray:
    """
    Debt profile expressed purely as an index (2025 = 100). No dollars.

    Shapes:
    - Flat (Baseline): 100 forever.
    - Front-Loaded: 100 in 2025 linearly declining to 80 by 2035, then flat at 80.
    - The "Cliff" (Risk): 100 until 2030, 120 from 2030-2037, then 80 after.
    - Custom: user sets Peak Debt Multiplier and Peak Year.
      Interpreted as an 8-year peak window starting at Peak Year (mirrors 2030-2037),
      then dropping to 80 afterwards.
    """
    years = np.asarray(years, dtype=int)
    debt = np.full(len(years), 100.0, dtype=float)

    if shape == "Flat (Baseline)":
        return debt

    if shape == "Front-Loaded":
        # Linear decline from 100 (2025) to 80 (2035), then flat 80.
        start, end = BASE_YEAR, 2035
        slope = (80.0 - 100.0) / max(1, (end - start))
        for i, y in enumerate(years):
            if y <= end:
                debt[i] = 100.0 + slope * (y - start)
            else:
                debt[i] = 80.0
        return clamp(debt, 1.0, None)

    if shape == 'The "Cliff" (Risk)':
        for i, y in enumerate(years):
            if 2030 <= y <= 2037:
                debt[i] = 120.0
            elif y >= 2038:
                debt[i] = 80.0
            else:
                debt[i] = 100.0
        return clamp(debt, 1.0, None)

    if shape == "Custom":
        peak_mult = float(custom_peak_multiplier)
        peak_mult = clamp(peak_mult, 0.5, 2.0)
        peak_start = int(custom_peak_year)
        peak_end = peak_start + 7  # fixed 8-year peak window
        for i, y in enumerate(years):
            if peak_start <= y <= peak_end:
                debt[i] = 100.0 * peak_mult
            elif y > peak_end:
                debt[i] = 80.0
            else:
                debt[i] = 100.0
        return clamp(debt, 1.0, None)

    # Safety fallback
    return debt


# -----------------------------
# Core model (indices only)
# -----------------------------
def run_model(
    wa_demand_share: float,
    national_trend_pct_by_2035: float,
    behavior_headwind_pct_by_2035: float,
    haggett_net_beds: int,
    rate_escalation: float,
    expense_inflation: float,
    expense_share: float,
    debt_share: float,
    debt_shape: str,
    custom_peak_multiplier: float,
    custom_peak_year: int,
) -> pd.DataFrame:
    """
    Builds a year-by-year dataframe of indices and DSCR-derived safety metrics.

    Everything is index-based:
    - Occupancy_Index: occupied headcount vs base headcount (2025=100).
    - Revenue_Index: Occupancy_Index * (1+Rate_Escalation)^t
    - Expense_Index: 100*(1+Expense_Inflation)^t
    - Debt_Index: shape-based index (no dollars)
    - NOI_Index: derived from Revenue_Index and base-year expense share
    - DSCR_Est: Base_DSCR * (NOI_Index / Debt_Index)
    - Safety_Margin_%: remaining cushion as % of base-year cushion
    """
    df = build_ofm_df()
    years = df["year"].to_numpy(dtype=int)
    t = (years - BASE_YEAR).astype(int)

    # Non-resident macro proxy (by 2035), expressed as an index like the WA OFM series.
    national_index = build_national_index(years, national_trend_pct_by_2035)

    # Optional behavioral headwind, outcome-neutral by default (0%).
    behavior_index = build_behavior_index(years, behavior_headwind_pct_by_2035)

    # Combine WA and non-WA demand environments into a single demographic index.
    # Using weighted average avoids pretending we know exact WA/non-WA enrollment counts.
    wa_share = float(clamp(wa_demand_share, 0.0, 1.0))
    demographic_index = wa_share * df["wa_18yo_index"].to_numpy(dtype=float) + (1.0 - wa_share) * national_index

    # Demand index includes behavior overlay (both are base-100 indices).
    demand_index = safe_div(demographic_index * behavior_index, 100.0, default=0.0)
    demand_index = clamp(demand_index, 0.0, None)

    # Convert demand index into a headcount scale to apply a physical cap.
    base_headcount = float(RECONCILED_DATA["housing_portfolio"]["occupancy"]["current_headcount"]["value"])
    demand_headcount = base_headcount * safe_div(demand_index, 100.0, default=0.0)

    capacity_headcount = build_capacity_headcount(years, haggett_net_beds)
    occupied_headcount = np.minimum(demand_headcount, capacity_headcount)
    occupied_headcount = clamp(occupied_headcount, 0.0, None)

    # Occupancy index is the financial “volume driver” (no dollars).
    occupancy_index = safe_div(occupied_headcount, base_headcount, default=0.0) * 100.0

    # Financial indices
    # Revenue_Index uses occupancy plus rate escalation compounding.
    revenue_index = occupancy_index * np.power(1.0 + float(rate_escalation), t)

    # Expense_Index is modeled as an annual inflation index (fixed-cost flavor).
    expense_index = 100.0 * np.power(1.0 + float(expense_inflation), t)

    # Debt index is shape-based. This is where the “debt cliff” shows up structurally.
    debt_index = build_debt_index(
        years=years,
        shape=debt_shape,
        custom_peak_multiplier=custom_peak_multiplier,
        custom_peak_year=custom_peak_year,
    )

    # Net Operating Index (raw, not rebased) per spec:
    # Net_Operating_Index = Revenue_Index - (Expense_Ratio_Base * Expense_Index)
    exp_share = float(clamp(expense_share, 0.0, 1.0))
    net_operating_index = revenue_index - (exp_share * expense_index)

    # NOI_Index rebases Net Operating to 2025 = 100 so DSCR approximation is well-behaved.
    # Base-year net operating (using indices) is: 100 - exp_share*100 = 100*(1-exp_share)
    net_operating_base = 100.0 * (1.0 - exp_share)
    noi_index = safe_div(net_operating_index, net_operating_base, default=np.nan) * 100.0

    # Relative coverage approximation (high-level, strategic):
    # New_DSCR ≈ Base_DSCR * (NOI_Index / Debt_Index)
    base_dscr = float(RECONCILED_DATA["financial_ratios"]["base_dscr"]["value"])
    dscr_est = base_dscr * safe_div(noi_index, debt_index, default=np.nan)

    # Safety Margin expressed as % of base-year cushion above covenant.
    required_dscr = float(RECONCILED_DATA["financial_ratios"]["required_dscr"]["value"])
    base_cushion = base_dscr - required_dscr

    safety_margin_pct = safe_div((dscr_est - required_dscr), base_cushion, default=np.nan) * 100.0

    # Extra diagnostic metric requested in prompt (unanchored structural coverage)
    # Relative_Coverage_Ratio = Net_Operating_Index / (Debt_Ratio_Base * Debt_Index)
    # Note: This uses a base-year debt share purely as a structural denominator, not dollars.
    debt_sh = float(clamp(debt_share, 0.0, 1.0))
    relative_coverage_ratio = safe_div(net_operating_index, (debt_sh * debt_index), default=np.nan)

    out = pd.DataFrame(
        {
            "year": years,
            "WA_18yo_Population": df["wa_18yo_population"].astype(int),
            "WA_18yo_Index": df["wa_18yo_index"].astype(float),
            "National_Global_Index": national_index.astype(float),
            "Behavior_Index": behavior_index.astype(float),
            "Demographic_Index": demographic_index.astype(float),
            "Demand_Index": demand_index.astype(float),
            "Capacity_Headcount_Cap": capacity_headcount.astype(float),
            "Capacity_Index": (safe_div(capacity_headcount, base_headcount, default=np.nan) * 100.0).astype(float),
            "Occupied_Headcount": occupied_headcount.astype(float),
            "Occupancy_Index": occupancy_index.astype(float),
            "Revenue_Index": revenue_index.astype(float),
            "Expense_Index": expense_index.astype(float),
            "Debt_Index": debt_index.astype(float),
            "Net_Operating_Index": net_operating_index.astype(float),
            "NOI_Index": noi_index.astype(float),
            "Relative_Coverage_Ratio": relative_coverage_ratio.astype(float),
            "DSCR_Est": dscr_est.astype(float),
            "Safety_Margin_%": safety_margin_pct.astype(float),
        }
    )

    out["Covenant_Breach"] = out["DSCR_Est"] < required_dscr
    return out


def years_until_safety_depleted(df: pd.DataFrame) -> Tuple[str, int | None]:
    """
    Primary KPI: first year where Safety_Margin_% <= 0 (covenant breach),
    reported as years from base year.

    Returns:
    - display string
    - depletion year (or None)
    """
    # Exclude base year when looking for "future" depletion.
    future = df[df["year"] > BASE_YEAR].copy()
    breach = future[future["Safety_Margin_%"] <= 0.0]
    if breach.empty:
        return f"No depletion through {END_YEAR}", None

    y = int(breach.iloc[0]["year"])
    yrs = y - BASE_YEAR
    return f"{y} (in {yrs} years)", y


def value_at_year(df: pd.DataFrame, year: int, col: str, default=np.nan) -> float:
    row = df[df["year"] == year]
    if row.empty:
        return default
    return float(row.iloc[0][col])


# -----------------------------
# Sensitivity (Tornado)
# -----------------------------
def tornado_sensitivity(base_params: Dict[str, object]) -> Tuple[pd.DataFrame, str]:
    """
    Tornado sensitivity on Safety_Margin_% in the target year.

    Inputs per requirement:
    - National Trend (macro non-resident environment)
    - Expense Inflation
    - Debt Peak Multiplier

    Approach:
    - “20% change” is implemented as +/-20% multiplicative shock to the parameter value.
    - For National Trend, we shock the *multiplier* at 2035 (1 + pct), then convert back to pct.
    - For Debt Peak Multiplier, if current debt shape has no peak, effect may be zero (by design).
    """
    target_year = TARGET_YEAR_TORNADO

    # Helper to run the model and extract a single scalar output.
    def safety_with(params_override: Dict[str, object]) -> float:
        p = dict(base_params)
        p.update(params_override)
        df_local = run_model(**p)  # type: ignore[arg-type]
        return value_at_year(df_local, target_year, "Safety_Margin_%", default=np.nan)

    base_safety = safety_with({})

    # --- Parameter 1: National Trend (shock the multiplier 1+pct)
    nat_pct = float(base_params["national_trend_pct_by_2035"])
    nat_mult = 1.0 + nat_pct
    nat_mult_low = clamp(nat_mult * 0.8, 0.5, 1.5)
    nat_mult_high = clamp(nat_mult * 1.2, 0.5, 1.5)
    nat_low = clamp(nat_mult_low - 1.0, -0.30, 0.30)
    nat_high = clamp(nat_mult_high - 1.0, -0.30, 0.30)

    # --- Parameter 2: Expense Inflation (shock the annual rate)
    exp_inf = float(base_params["expense_inflation"])
    exp_inf_low = clamp(exp_inf * 0.8, 0.0, 0.10)
    exp_inf_high = clamp(exp_inf * 1.2, 0.0, 0.10)

    # --- Parameter 3: Debt Peak Multiplier (only meaningful for Cliff/Custom shapes)
    peak_mult = float(base_params["custom_peak_multiplier"])
    peak_mult_low = clamp(peak_mult * 0.8, 0.6, 1.6)
    peak_mult_high = clamp(peak_mult * 1.2, 0.6, 1.6)

    results = []
    # National Trend
    results.append(
        {
            "Parameter": "National/Global HS Grad Trend",
            "Low (20%)": safety_with({"national_trend_pct_by_2035": nat_low}),
            "High (20%)": safety_with({"national_trend_pct_by_2035": nat_high}),
            "Base": base_safety,
        }
    )
    # Expense Inflation
    results.append(
        {
            "Parameter": "Expense Inflation",
            "Low (20%)": safety_with({"expense_inflation": exp_inf_low}),
            "High (20%)": safety_with({"expense_inflation": exp_inf_high}),
            "Base": base_safety,
        }
    )
    # Debt Peak Multiplier (applies only to Custom; for Cliff it remains fixed by model)
    # To keep the tornado aligned with "Debt Peak Multiplier", we only override the peak multiplier
    # and force debt shape to Custom during sensitivity runs if the user is not already using Custom.
    base_shape = str(base_params["debt_shape"])
    if base_shape != "Custom":
        debt_override_common = {"debt_shape": "Custom", "custom_peak_year": int(base_params["custom_peak_year"])}
    else:
        debt_override_common = {}

    results.append(
        {
            "Parameter": "Debt Peak Multiplier",
            "Low (20%)": safety_with({**debt_override_common, "custom_peak_multiplier": peak_mult_low}),
            "High (20%)": safety_with({**debt_override_common, "custom_peak_multiplier": peak_mult_high}),
            "Base": base_safety,
        }
    )

    df_t = pd.DataFrame(results).copy()
    df_t["Low_Delta"] = df_t["Low (20%)"] - df_t["Base"]
    df_t["High_Delta"] = df_t["High (20%)"] - df_t["Base"]
    df_t["Impact_Abs"] = np.maximum(np.abs(df_t["Low_Delta"]), np.abs(df_t["High_Delta"]))
    df_t = df_t.sort_values("Impact_Abs", ascending=False).reset_index(drop=True)

    # Text summary requested
    most = df_t.iloc[0]
    param_name = str(most["Parameter"])
    max_abs_delta = float(most["Impact_Abs"])

    # Prefer a relative % change when base is not tiny, otherwise report percentage points.
    base_val = float(base_safety)
    if np.isfinite(base_val) and abs(base_val) >= 5.0:
        x = safe_div(max_abs_delta, abs(base_val), default=np.nan) * 100.0
        summary = f"The model is most sensitive to {param_name}, where a 20% change drives about {x:.0f}% change in Safety Margin (target year {target_year})."
    else:
        summary = f"The model is most sensitive to {param_name}, where a 20% change shifts Safety Margin by about {max_abs_delta:.1f} percentage points (target year {target_year})."

    return df_t, summary


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="UW HFS Housing Structural Risk Model (Index-Based)",
    layout="wide",
)

st.title("UW HFS Housing Structural Risk Model")
st.caption(
    "All outputs are indices (2025 = 100) and DSCR-derived Safety Margin. "
    "No currency values are computed or exported."
)

# Initialize defaults once
if "initialized" not in st.session_state:
    st.session_state["scenario"] = "Baseline"
    for k, v in SCENARIOS["Baseline"].items():
        st.session_state[k] = v
    st.session_state["initialized"] = True


def apply_scenario_from_state():
    scenario = st.session_state.get("scenario", "Baseline")
    defaults = SCENARIOS.get(scenario, {})
    # If scenario is "Custom (Keep Current Settings)", do nothing.
    for k, v in defaults.items():
        st.session_state[k] = v


with st.sidebar:
    st.header("Controls")

    st.selectbox(
        "Scenario (One-click)",
        options=list(SCENARIOS.keys()),
        key="scenario",
        on_change=apply_scenario_from_state,
        help="Selecting a scenario overwrites the controls below (except the 'Custom' scenario).",
    )

    st.subheader("Demand Engine")

    st.slider(
        "WA Demand Share",
        min_value=0.40,
        max_value=0.90,
        step=0.01,
        key="wa_demand_share",
        help="Weighting between WA OFM pipeline and non-resident macro environment. Both are indices (2025=100).",
    )

    st.slider(
        "National/Global HS Grad Trend (by 2035)",
        min_value=-30,
        max_value=30,
        step=1,
        key="national_trend_pct_by_2035",
        help="Macro proxy for non-resident pipeline (index moves from 100 in 2025 to 100*(1+trend) by 2035, then holds).",
    )

    st.slider(
        "Behavioral Headwind (housing demand) by 2035",
        min_value=-20,
        max_value=10,
        step=1,
        key="behavior_headwind_pct_by_2035",
        help="Optional overlay for changes in college-going, price sensitivity, and take rates. Default 0 for outcome neutrality.",
    )

    st.slider(
        "Haggett Replacement, Net Bed Change (effective 2027)",
        min_value=-500,
        max_value=1000,
        step=25,
        key="haggett_net_beds",
        help="Used only as a physical occupancy cap. Default 0 because this is a replacement and net change may be near zero.",
    )

    st.subheader("Financial Indices")

    st.slider(
        "Rate Escalation (annual)",
        min_value=0.0,
        max_value=6.0,
        step=0.1,
        key="rate_escalation_pct",
        help="Compounding annual rate escalation applied to Revenue Index (not dollars).",
    )

    st.slider(
        "Expense Inflation (annual)",
        min_value=0.0,
        max_value=6.0,
        step=0.1,
        key="expense_inflation_pct",
        help="Compounding annual inflation applied to Expense Index (not dollars).",
    )

    st.subheader("Debt Profile Shape")

    st.selectbox(
        "Debt Profile Shape",
        options=DEBT_SHAPES,
        key="debt_shape",
        help="Debt is modeled as an index shape, not a dollar schedule. Default is Flat for outcome neutrality.",
    )

    if st.session_state["debt_shape"] == "Custom":
        st.slider(
            "Peak Debt Multiplier",
            min_value=1.00,
            max_value=1.50,
            step=0.01,
            key="custom_peak_multiplier",
            help="Debt Index equals 100*multiplier during the peak window (8 years starting at Peak Year).",
        )
        st.slider(
            "Peak Year",
            min_value=2027,
            max_value=2040,
            step=1,
            key="custom_peak_year",
            help="Start year of the 8-year peak window for Custom debt shape.",
        )
    elif st.session_state["debt_shape"] == 'The "Cliff" (Risk)':
        st.caption("Cliff definition: Debt Index = 120 for 2030–2037, then 80 from 2038 onward.")

    with st.expander("Advanced: Base Shares (Index Anchors)", expanded=False):
        st.slider(
            "Operating Expense Share (base year)",
            min_value=30,
            max_value=70,
            step=1,
            key="expense_share_pct",
            help="Used to compute Net Operating Index as Revenue_Index - expense_share*Expense_Index.",
        )
        st.slider(
            "Debt Service Share (base year)",
            min_value=10,
            max_value=50,
            step=1,
            key="debt_share_pct",
            help="Used only for the unanchored Relative_Coverage_Ratio diagnostic (still no dollars).",
        )

    st.subheader("DSCR Anchors (read-only)")
    st.write(f"Base DSCR: {RECONCILED_DATA['financial_ratios']['base_dscr']['value']:.2f}")
    st.write(f"Covenant DSCR: {RECONCILED_DATA['financial_ratios']['required_dscr']['value']:.2f}")

# Collect params (convert user % controls to decimals)
params = {
    "wa_demand_share": float(st.session_state["wa_demand_share"]),
    "national_trend_pct_by_2035": float(st.session_state["national_trend_pct_by_2035"]) / 100.0,
    "behavior_headwind_pct_by_2035": float(st.session_state["behavior_headwind_pct_by_2035"]) / 100.0,
    "haggett_net_beds": int(st.session_state["haggett_net_beds"]),
    "rate_escalation": float(st.session_state["rate_escalation_pct"]) / 100.0,
    "expense_inflation": float(st.session_state["expense_inflation_pct"]) / 100.0,
    "expense_share": float(st.session_state["expense_share_pct"]) / 100.0,
    "debt_share": float(st.session_state["debt_share_pct"]) / 100.0,
    "debt_shape": str(st.session_state["debt_shape"]),
    "custom_peak_multiplier": float(st.session_state.get("custom_peak_multiplier", 1.20)),
    "custom_peak_year": int(st.session_state.get("custom_peak_year", 2030)),
}

df = run_model(**params)

# Primary KPI
depletion_text, depletion_year = years_until_safety_depleted(df)

# Headline metrics
base_dscr = float(RECONCILED_DATA["financial_ratios"]["base_dscr"]["value"])
required_dscr = float(RECONCILED_DATA["financial_ratios"]["required_dscr"]["value"])
dscr_2035 = value_at_year(df, TARGET_YEAR_TORNADO, "DSCR_Est", default=np.nan)
safety_2035 = value_at_year(df, TARGET_YEAR_TORNADO, "Safety_Margin_%", default=np.nan)
min_safety = float(np.nanmin(df["Safety_Margin_%"].to_numpy(dtype=float)))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Years until Safety Margin Depleted", depletion_text)
c2.metric(f"DSCR (Year {TARGET_YEAR_TORNADO})", f"{dscr_2035:.2f}" if np.isfinite(dscr_2035) else "n/a")
c3.metric(f"Safety Margin (Year {TARGET_YEAR_TORNADO})", f"{safety_2035:.0f}%" if np.isfinite(safety_2035) else "n/a")
c4.metric("Minimum Safety Margin (2025–2045)", f"{min_safety:.0f}%")

tabs = st.tabs(["Dashboard", "Sensitivity (Tornado)", "Data Export"])

# -----------------------------
# Dashboard tab
# -----------------------------
with tabs[0]:
    left, right = st.columns(2)

    with left:
        st.subheader("Scissors Graph: Revenue Index vs Expense Index")

        plot_df = df[["year", "Revenue_Index", "Expense_Index"]].copy()
        plot_long = plot_df.melt("year", var_name="Series", value_name="Index")

        if HAS_ALTAIR:
            chart = (
                alt.Chart(plot_long)
                .mark_line(point=False)
                .encode(
                    x=alt.X("year:O", title="Year"),
                    y=alt.Y("Index:Q", title="Index (2025 = 100)"),
                    color=alt.Color("Series:N", title=""),
                    tooltip=["year:O", "Series:N", alt.Tooltip("Index:Q", format=".1f")],
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.line_chart(plot_df.set_index("year")[["Revenue_Index", "Expense_Index"]])

        st.caption("When Expense Index grows faster than Revenue Index, the safety cushion tends to compress.")

    with right:
        st.subheader("Safety Margin Over Time (Percent of Base Cushion)")
        safety_df = df[["year", "Safety_Margin_%"]].copy()

        if HAS_ALTAIR:
            base = (
                alt.Chart(safety_df)
                .mark_line()
                .encode(
                    x=alt.X("year:O", title="Year"),
                    y=alt.Y("Safety_Margin_%:Q", title="Safety Margin (% of base cushion)"),
                    tooltip=["year:O", alt.Tooltip("Safety_Margin_%:Q", format=".1f")],
                )
                .properties(height=320)
            )
            zero_line = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color="#888").encode(y="y:Q")
            st.altair_chart(base + zero_line, use_container_width=True)
        else:
            st.line_chart(safety_df.set_index("year")["Safety_Margin_%"])

        st.caption("100% = same cushion as base year, 0% = covenant breach, negative = below covenant.")

    st.subheader("DSCR (Approximation) vs Covenant")
    dscr_df = df[["year", "DSCR_Est"]].copy()
    dscr_df["Covenant"] = required_dscr

    if HAS_ALTAIR:
        dscr_long = dscr_df.melt("year", var_name="Series", value_name="Value")
        chart = (
            alt.Chart(dscr_long)
            .mark_line()
            .encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("Value:Q", title="DSCR"),
                color=alt.Color("Series:N", title=""),
                tooltip=["year:O", "Series:N", alt.Tooltip("Value:Q", format=".2f")],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.line_chart(dscr_df.set_index("year")[["DSCR_Est", "Covenant"]])

# -----------------------------
# Sensitivity tab
# -----------------------------
with tabs[1]:
    st.subheader(f"Tornado Sensitivity: Safety Margin in {TARGET_YEAR_TORNADO}")

    tornado_base_params = dict(params)  # already decimals and model-ready
    df_t, summary = tornado_sensitivity(tornado_base_params)

    # Build tornado bars as ranges (Low -> High)
    df_t_plot = df_t.copy()
    df_t_plot["Low"] = df_t_plot[["Low (20%)", "High (20%)"]].min(axis=1)
    df_t_plot["High"] = df_t_plot[["Low (20%)", "High (20%)"]].max(axis=1)

    if HAS_ALTAIR:
        chart = (
            alt.Chart(df_t_plot)
            .mark_bar()
            .encode(
                y=alt.Y("Parameter:N", sort="-x", title=""),
                x=alt.X("Low:Q", title="Safety Margin (% of base cushion)"),
                x2="High:Q",
                tooltip=[
                    "Parameter:N",
                    alt.Tooltip("Base:Q", format=".1f"),
                    alt.Tooltip("Low (20%):Q", format=".1f"),
                    alt.Tooltip("High (20%):Q", format=".1f"),
                ],
            )
            .properties(height=200)
        )
        base_rule = (
            alt.Chart(pd.DataFrame({"Base": [float(df_t_plot["Base"].iloc[0])]}))
            .mark_rule(color="#444")
            .encode(x="Base:Q")
        )
        st.altair_chart(chart + base_rule, use_container_width=True)
    else:
        st.dataframe(df_t_plot[["Parameter", "Base", "Low (20%)", "High (20%)"]])

    st.write(summary)

    st.caption(
        "Note: If the current Debt Profile Shape is Flat or Front-Loaded, debt peak sensitivity is evaluated "
        "by temporarily using the Custom debt profile for the sensitivity runs (so the parameter is meaningful)."
    )

# -----------------------------
# Data export tab
# -----------------------------
with tabs[2]:
    st.subheader("Export (Indices Only, No Currency)")

    export_cols = [
        "year",
        "WA_18yo_Population",
        "WA_18yo_Index",
        "National_Global_Index",
        "Behavior_Index",
        "Demographic_Index",
        "Demand_Index",
        "Capacity_Index",
        "Occupancy_Index",
        "Revenue_Index",
        "Expense_Index",
        "Debt_Index",
        "NOI_Index",
        "DSCR_Est",
        "Safety_Margin_%",
        "Covenant_Breach",
    ]

    display_df = df[export_cols].copy()

    # Round for readability while keeping indices, not dollars.
    round_map = {c: 1 for c in export_cols if c not in {"year", "WA_18yo_Population", "Covenant_Breach"}}
    display_df = display_df.round(round_map)

    st.dataframe(display_df, use_container_width=True, height=420)

    csv_buffer = StringIO()
    display_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download CSV (indices only)",
        data=csv_buffer.getvalue(),
        file_name="uw_hfs_housing_indices_only.csv",
        mime="text/csv",
    )

    st.caption(
        "Export includes only indices and DSCR-derived metrics. No absolute currency values are generated or exported."
    )
