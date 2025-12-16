# uw_housing_model.py
"""
UW HFS Housing Structural Risk Model (Index-Based, Outcome Neutral)
v4.2 - Code Review Improvements

Changes in v4.2 (Code Review Implementation):
- Extracted demand index calculation into dedicated function for clarity and testability
- Added comprehensive inline comments for traceability
- Reduced executive jargon in labels and tooltips
- Added plain-English interpretation callouts for KPI cards
- Improved scenario descriptions with "when to use" guidance
- Enhanced type hints for numerical safety functions
- Fixed Custom scenario reset button UX with clearer tooltip
- Documented magic numbers and index normalization logic
- Added colorblind-friendly considerations to chart annotations

Previous fixes (v4.1):
- Charts collapsing under scenarios with a debt-peak band was caused by the
  peak-band layer not sharing the same explicit x-scale domain as the rest of the chart.
- Added "UW Enrollment Trend (Class Size)" slider.

Constraints:
- No absolute dollars are computed, displayed, or exported.
- DSCR approximation logic and index mechanics are retained.
- Outcome neutrality: Baseline remains non-alarmist, flat debt timing.

Run:
    streamlit run uw_housing_model.py
"""

from __future__ import annotations

from io import StringIO
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# STREAMLIT PAGE CONFIG
# Note: Must be the first Streamlit command for stability.
# =============================================================================
st.set_page_config(
    page_title="UW HFS Housing Structural Risk Dashboard (Index-Based)",
    layout="wide",
)

# Altair is typically present in Streamlit environments, but keep a safe fallback.
try:
    import altair as alt

    HAS_ALTAIR = True
except Exception:
    HAS_ALTAIR = False


# =============================================================================
# CONSTANTS AND CONFIGURATION
# Reviewer: These are the foundational parameters for the model.
# =============================================================================

# Time horizon for the model
BASE_YEAR = 2025
END_YEAR = 2045
YEARS = list(range(BASE_YEAR, END_YEAR + 1))

# Base index value used throughout the model
# All indices are normalized to this value in the base year (2025)
BASE_INDEX = 100.0

# =============================================================================
# RECONCILED DATA
# Source: UW HFS financial reports and WA OFM demographic projections
# Note: Financial values here are for BASELINE INDEXING only, not absolute dollars.
# =============================================================================
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
        # Base DSCR from 2022 audited financials, serves as anchor for projections
        "base_dscr": {"value": 1.57, "note": "2022 Actual, serves as DSCR anchor"},
        # Bond covenant minimum requirement
        "required_dscr": {"value": 1.25},
        # Debt service share assumption
        # Code Review Note (v4.2): This 35% figure is from historical budget analysis.
        # Consider updating annually based on actual debt service schedules.
        "debt_service_share": {
            "value": 0.35,
            "note": "Est. debt service as % of base-year indexed revenue (diagnostic only)",
            "source": "Historical budget analysis, FY2022-2024 average",
        },
        "expense_share": {
            "value": 0.50,
            "note": "Est. operating expense as % of base-year indexed revenue",
        },
        "margin_share": {"value": 0.15, "note": "Net margin share (context only)"},
    },
}

# Extract WA OFM data as list of tuples for easier processing
WA_OFM_18YO = [
    (x["year"], x["population"])
    for x in RECONCILED_DATA["demographics"]["wa_18yo_population"]
]

# Debt service timing shape options
DEBT_SHAPES = [
    "Flat (Baseline)",
    "Front-Loaded",
    'The "Cliff" (Risk)',
    "Custom",
]

# =============================================================================
# SCENARIOS
# Code Review Note (v4.2): Scenarios are designed to be outcome-neutral by default.
# The Baseline scenario uses matched inflation/revenue growth with no headwinds.
# =============================================================================
SCENARIOS: Dict[str, Dict[str, object]] = {
    "Baseline": {
        "debt_shape": "Flat (Baseline)",
        "rate_escalation_pct": 2.5,
        "expense_inflation_pct": 2.5,
        "national_trend_pct_by_2035": 0,
        "wa_demand_share": 70,  # percent (in-state share)
        "uw_enrollment_trend_pct": 0,  # UW policy/budget impact on class size
        "behavior_headwind_pct_by_2035": 0,  # housing preference shift
        "haggett_net_beds": 0,
        "expense_share_pct": int(
            RECONCILED_DATA["financial_ratios"]["expense_share"]["value"] * 100
        ),
        "debt_share_pct": int(
            RECONCILED_DATA["financial_ratios"]["debt_service_share"]["value"] * 100
        ),
        "custom_peak_multiplier": 1.20,
        "custom_peak_year": 2030,
        "focus_year": 2035,
    },
    "Structural Squeeze": {
        "debt_shape": 'The "Cliff" (Risk)',
        "rate_escalation_pct": 3.0,
        "expense_inflation_pct": 4.0,
        "national_trend_pct_by_2035": 0,
        "wa_demand_share": 70,
        "uw_enrollment_trend_pct": 0,
        "behavior_headwind_pct_by_2035": 0,
        "haggett_net_beds": 0,
        "expense_share_pct": int(
            RECONCILED_DATA["financial_ratios"]["expense_share"]["value"] * 100
        ),
        "debt_share_pct": int(
            RECONCILED_DATA["financial_ratios"]["debt_service_share"]["value"] * 100
        ),
        "custom_peak_multiplier": 1.20,
        "custom_peak_year": 2030,
        "focus_year": 2035,
    },
    "Demographic Trough": {
        "debt_shape": "Flat (Baseline)",
        "rate_escalation_pct": 2.5,
        "expense_inflation_pct": 2.5,
        "national_trend_pct_by_2035": -10,
        "wa_demand_share": 70,
        "uw_enrollment_trend_pct": 0,
        "behavior_headwind_pct_by_2035": 0,
        "haggett_net_beds": 0,
        "expense_share_pct": int(
            RECONCILED_DATA["financial_ratios"]["expense_share"]["value"] * 100
        ),
        "debt_share_pct": int(
            RECONCILED_DATA["financial_ratios"]["debt_service_share"]["value"] * 100
        ),
        "custom_peak_multiplier": 1.20,
        "custom_peak_year": 2030,
        "focus_year": 2035,
    },
    "Bond Stress Test (Illustrative)": {
        "debt_shape": 'The "Cliff" (Risk)',
        "rate_escalation_pct": 2.5,
        "expense_inflation_pct": 4.5,
        "national_trend_pct_by_2035": -10,
        "wa_demand_share": 70,
        "uw_enrollment_trend_pct": -10,  # Stress test includes enrollment decline
        "behavior_headwind_pct_by_2035": -15,
        "haggett_net_beds": 0,
        "expense_share_pct": int(
            RECONCILED_DATA["financial_ratios"]["expense_share"]["value"] * 100
        ),
        "debt_share_pct": int(
            RECONCILED_DATA["financial_ratios"]["debt_service_share"]["value"] * 100
        ),
        "custom_peak_multiplier": 1.20,
        "custom_peak_year": 2030,
        "focus_year": 2035,
    },
    "Custom (Keep Current Settings)": {},
}

# =============================================================================
# SCENARIO CONTEXT DESCRIPTIONS
# Code Review Update (v4.2): Added "when to use" guidance for executives.
# =============================================================================
SCENARIO_CONTEXT: Dict[str, str] = {
    "Baseline": (
        "**Baseline:** Steady rent and cost growth, flat debt timing, no headwinds. "
        "Use this as your 'expected case' for annual budget planning."
    ),
    "Structural Squeeze": (
        "**Structural Squeeze:** Faster cost growth plus a 2030–2037 debt peak. "
        "Use this to test sensitivity to capital project timing and cost pressures."
    ),
    "Demographic Trough": (
        "**Demographic Trough:** WA pipeline follows state projections, out-of-state pipeline shrinks 10% by 2035. "
        "Use this to understand exposure to national enrollment trends."
    ),
    "Bond Stress Test (Illustrative)": (
        "**Stress Test:** Combined cost growth, debt peak, and weaker demand. "
        "Use this to show bondholders/auditors that you've tested worst-case scenarios."
    ),
    "Custom (Keep Current Settings)": (
        "**Custom:** Preserves your current slider settings without automatic resets. "
        "Use this when fine-tuning specific assumptions."
    ),
}


# =============================================================================
# NUMERICAL SAFETY HELPERS
# Code Review Note (v4.2): Added type hints for clarity on expected inputs.
# =============================================================================


def safe_div(
    n: np.ndarray | float,
    d: np.ndarray | float,
    default: float = np.nan,
) -> np.ndarray | float:
    """
    Division that never throws divide-by-zero and never returns inf/-inf.

    Args:
        n: Numerator (scalar or numpy array)
        d: Denominator (scalar or numpy array)
        default: Value to use when division is undefined

    Returns:
        Result of n/d with undefined values replaced by default
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(n, d)

    if np.isscalar(out):
        return out if np.isfinite(out) else default

    out = np.where(np.isfinite(out), out, default)
    return out


def clamp(
    x: np.ndarray | float,
    lo: float | None = None,
    hi: float | None = None,
) -> np.ndarray | float:
    """
    Clamp values to a range, handling None bounds gracefully.

    Args:
        x: Value(s) to clamp
        lo: Lower bound (None = no lower bound)
        hi: Upper bound (None = no upper bound)

    Returns:
        Clamped value(s)
    """
    lo = -np.inf if lo is None else lo
    hi = np.inf if hi is None else hi
    return np.clip(x, lo, hi)


def linear_progress(
    years: np.ndarray | list,
    start_year: int,
    end_year: int,
) -> np.ndarray:
    """
    Calculate linear progress from 0 at start_year to 1 at end_year, clamped outside.

    Used for phasing in trends over time (e.g., "10% reduction BY 2035").

    Args:
        years: Array of years to calculate progress for
        start_year: Year when progress = 0
        end_year: Year when progress = 1

    Returns:
        Array of progress values [0, 1]
    """
    years = np.asarray(years, dtype=float)
    denom = max(1.0, float(end_year - start_year))
    prog = (years - float(start_year)) / denom
    return clamp(prog, 0.0, 1.0)


def finite_minmax(vals: np.ndarray) -> Tuple[float, float]:
    """
    Get min/max of finite values only, ignoring NaN and Inf.

    Args:
        vals: Array of values

    Returns:
        Tuple of (min, max) or (nan, nan) if no finite values
    """
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan)
    return float(np.min(v)), float(np.max(v))


def padded_domain(
    vals: np.ndarray,
    pad_abs: float,
    force_include: Tuple[float, float] | None = None,
    clamp_low: float | None = None,
    clamp_high: float | None = None,
) -> Tuple[float, float] | None:
    """
    Calculate Y-axis domain that uses chart space efficiently while including key thresholds.

    Args:
        vals: Data values to fit
        pad_abs: Absolute padding to add above/below data range
        force_include: Optional (low, high) values that must be included in domain
        clamp_low: Optional minimum for the domain lower bound
        clamp_high: Optional maximum for the domain upper bound

    Returns:
        Tuple of (domain_min, domain_max) or None if no valid data
    """
    vmin, vmax = finite_minmax(vals)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None

    lo = vmin - pad_abs
    hi = vmax + pad_abs

    if force_include is not None:
        lo = min(lo, float(force_include[0]))
        hi = max(hi, float(force_include[1]))

    if clamp_low is not None:
        lo = max(lo, float(clamp_low))
    if clamp_high is not None:
        hi = min(hi, float(clamp_high))

    if hi <= lo:
        hi = lo + max(1e-6, pad_abs)

    return (lo, hi)


# =============================================================================
# DEMAND ENGINE COMPONENTS
# Code Review Note (v4.2): Separated into distinct functions for clarity.
# The demand model has THREE independent components:
#   1. Demographics: How many college-age students exist (WA OFM + national trends)
#   2. Enrollment: How many UW admits (policy, budget, acceptance rates)
#   3. Housing Preference: What fraction choose on-campus housing (take rate)
# =============================================================================


def build_ofm_df() -> pd.DataFrame:
    """
    Build dataframe of WA 18-year-old population projections indexed to base year.

    Returns:
        DataFrame with columns: year, wa_18yo_population, wa_18yo_index
    """
    df = pd.DataFrame(WA_OFM_18YO, columns=["year", "wa_18yo_population"]).copy()
    df = df[(df["year"] >= BASE_YEAR) & (df["year"] <= END_YEAR)].sort_values("year")
    base_pop = float(df.loc[df["year"] == BASE_YEAR, "wa_18yo_population"].iloc[0])
    df["wa_18yo_index"] = (
        safe_div(df["wa_18yo_population"], base_pop, default=0.0) * BASE_INDEX
    )
    return df.reset_index(drop=True)


def build_national_index(years: np.ndarray, pct_by_2035: float) -> np.ndarray:
    """
    Build out-of-state enrollment outlook index.

    This captures national/international enrollment trends separate from WA demographics.
    Ramps linearly to target by 2035, then holds constant.

    Args:
        years: Array of years
        pct_by_2035: Percent change by 2035 (e.g., -0.10 for -10%)

    Returns:
        Array of index values (base year = 100)
    """
    prog = linear_progress(years, BASE_YEAR, 2035)
    idx = BASE_INDEX * (1.0 + pct_by_2035 * prog)
    return clamp(idx, 0.0, None)


def build_behavior_index(years: np.ndarray, pct_by_2035: float) -> np.ndarray:
    """
    Build housing preference shift index (take rate / competition factor).

    This captures changes in the fraction of students who choose on-campus housing,
    driven by off-campus competition, remote learning trends, etc.
    Ramps linearly to target by 2035, then holds constant.

    Args:
        years: Array of years
        pct_by_2035: Percent change by 2035 (e.g., -0.15 for -15%)

    Returns:
        Array of index values (base year = 100)
    """
    prog = linear_progress(years, BASE_YEAR, 2035)
    idx = BASE_INDEX * (1.0 + pct_by_2035 * prog)
    return clamp(idx, 0.0, None)


def build_enrollment_index_constant(years: np.ndarray, pct_level: float) -> np.ndarray:
    """
    Build UW enrollment trend index (freshman class size / admissions policy).

    This captures UW-specific policy decisions affecting class size:
    - Budget-driven enrollment caps
    - Changes in admission selectivity
    - Program additions or cuts

    Implemented as a constant level multiplier (not ramped) because policy
    changes typically take effect immediately rather than phasing in.

    Args:
        years: Array of years
        pct_level: Percent change from baseline (e.g., -0.10 for -10%)

    Returns:
        Array of index values (base year = 100)
    """
    idx = BASE_INDEX * (1.0 + pct_level)
    return np.full(len(years), clamp(idx, 0.0, None), dtype=float)


def compute_demand_index(
    wa_18yo_index: np.ndarray,
    national_index: np.ndarray,
    enrollment_index: np.ndarray,
    behavior_index: np.ndarray,
    wa_share: float,
) -> np.ndarray:
    """
    Compute composite demand index from all demand drivers.

    Code Review Note (v4.2): This function was extracted from run_model() for
    clarity and testability. The multiplicative model correctly separates:
    - "Fewer students EXIST" → demographic_blend and enrollment_index
    - "Fewer students CHOOSE housing" → behavior_index

    Mathematical Model:
        Demand = (Demographic_blend × Enrollment × Behavior) / 100²

    Why divide by 10,000?
        We multiply THREE base-100 indices together. To return a base-100 result:
        (100 × 100 × 100) / 10,000 = 100
        This is equivalent to: BASE_INDEX^(n_factors - 1) where n_factors = 3

    Args:
        wa_18yo_index: WA demographic index (base 100)
        national_index: Out-of-state enrollment outlook index (base 100)
        enrollment_index: UW enrollment policy index (base 100)
        behavior_index: Housing preference index (base 100)
        wa_share: Fraction of demand from in-state students (0.0 to 1.0)

    Returns:
        Composite demand index (base 100)
    """
    # Blend WA and out-of-state demographic/enrollment trends
    demographic_blend = wa_share * wa_18yo_index + (1.0 - wa_share) * national_index

    # Multiplicative combination of all factors
    # Note: Using explicit constant for the normalization divisor
    # n_factors = 3 (demographic_blend, enrollment, behavior)
    # divisor = BASE_INDEX ** (n_factors - 1) = 100 ** 2 = 10,000
    NORMALIZATION_DIVISOR = 10_000.0  # = BASE_INDEX ** 2

    raw = demographic_blend * enrollment_index * behavior_index
    demand_index = safe_div(raw, NORMALIZATION_DIVISOR, default=0.0)

    return clamp(demand_index, 0.0, None)


def build_capacity_headcount(years: np.ndarray, haggett_net_beds: int) -> np.ndarray:
    """
    Build physical capacity headcount over time.

    Capacity is used ONLY as a physical cap on occupancy headcount.
    It does not create demand - it just limits how many students can be housed.

    Args:
        years: Array of years
        haggett_net_beds: Net bed change from Haggett replacement (effective 2027)

    Returns:
        Array of capacity headcounts by year
    """
    base_operating = int(
        RECONCILED_DATA["housing_portfolio"]["totals"]["total_operating_capacity"][
            "value"
        ]
    )
    overflow = int(
        RECONCILED_DATA["housing_portfolio"]["totals"]["overflow_beds"]["value"]
    )
    cap_base = float(base_operating + overflow)

    cap = np.full(len(years), cap_base, dtype=float)
    cap = np.where(np.asarray(years) >= 2027, cap + float(haggett_net_beds), cap)
    return clamp(cap, 0.0, None)


# =============================================================================
# DEBT SERVICE TIMING
# =============================================================================


def build_debt_index(
    years: np.ndarray,
    shape: str,
    custom_peak_multiplier: float,
    custom_peak_year: int,
) -> np.ndarray:
    """
    Build debt service timing index (no dollars, timing pattern only).

    The debt index represents WHEN debt service payments are higher or lower,
    not the absolute dollar amounts. This allows testing sensitivity to
    capital project timing without exposing actual debt schedules.

    Args:
        years: Array of years
        shape: One of DEBT_SHAPES
        custom_peak_multiplier: For Custom shape, peak multiplier (e.g., 1.20 = +20%)
        custom_peak_year: For Custom shape, start year of 8-year peak window

    Returns:
        Array of debt index values (base year = 100)
    """
    years = np.asarray(years, dtype=int)
    debt = np.full(len(years), BASE_INDEX, dtype=float)

    if shape == "Flat (Baseline)":
        return debt

    if shape == "Front-Loaded":
        # Debt service declines from 100 to 80 by 2035, then holds at 80
        start, end = BASE_YEAR, 2035
        slope = (80.0 - 100.0) / max(1, (end - start))
        for i, y in enumerate(years):
            if y <= end:
                debt[i] = 100.0 + slope * (y - start)
            else:
                debt[i] = 80.0
        return clamp(debt, 1.0, None)

    if shape == 'The "Cliff" (Risk)':
        # Elevated debt service 2030-2037, then drops to 80
        # This models a scenario where major capital projects create a payment spike
        for i, y in enumerate(years):
            if 2030 <= y <= 2037:
                debt[i] = 120.0
            elif y >= 2038:
                debt[i] = 80.0
            else:
                debt[i] = 100.0
        return clamp(debt, 1.0, None)

    if shape == "Custom":
        # User-defined peak window
        peak_mult = float(custom_peak_multiplier)
        peak_mult = clamp(peak_mult, 0.5, 2.0)
        peak_start = int(custom_peak_year)
        peak_end = peak_start + 7  # 8-year window
        for i, y in enumerate(years):
            if peak_start <= y <= peak_end:
                debt[i] = 100.0 * peak_mult
            elif y > peak_end:
                debt[i] = 80.0
            else:
                debt[i] = 100.0
        return clamp(debt, 1.0, None)

    return debt


def get_debt_peak_window(
    debt_shape: str,
    custom_peak_year: int,
    custom_peak_multiplier: float,
) -> Tuple[int, int] | None:
    """
    Identify debt peak pressure window for chart shading (UI helper only).

    Args:
        debt_shape: Selected debt shape
        custom_peak_year: Custom peak start year
        custom_peak_multiplier: Custom peak multiplier

    Returns:
        Tuple of (start_year, end_year) or None if no peak
    """
    if debt_shape == 'The "Cliff" (Risk)':
        return (2030, 2037)
    if debt_shape == "Custom" and float(custom_peak_multiplier) > 1.0001:
        start = int(custom_peak_year)
        return (start, start + 7)
    return None


# =============================================================================
# CORE MODEL
# =============================================================================


def run_model(
    wa_demand_share: float,
    national_trend_pct_by_2035: float,
    uw_enrollment_trend_pct: float,
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
    Run the structural risk model and return year-by-year projections.

    Key constraints:
    - No absolute dollars. Indices only.
    - All indices are base-100 (2025 = 100)

    DSCR approximation (directional, anchored):
        DSCR(t) ≈ Base_DSCR × (NOI_Index(t) / Debt_Index(t))

    Args:
        wa_demand_share: Fraction of demand from in-state (0.0 to 1.0)
        national_trend_pct_by_2035: Out-of-state trend by 2035 (decimal, e.g., -0.10)
        uw_enrollment_trend_pct: UW enrollment policy impact (decimal, e.g., -0.10)
        behavior_headwind_pct_by_2035: Housing preference shift by 2035 (decimal)
        haggett_net_beds: Net bed change from Haggett (integer)
        rate_escalation: Annual rent increase (decimal, e.g., 0.025 for 2.5%)
        expense_inflation: Annual cost growth (decimal, e.g., 0.025 for 2.5%)
        expense_share: Operating expense as fraction of revenue (0.0 to 1.0)
        debt_share: Debt service as fraction of revenue (0.0 to 1.0)
        debt_shape: Debt timing pattern name
        custom_peak_multiplier: For custom debt shape
        custom_peak_year: For custom debt shape

    Returns:
        DataFrame with all projection metrics by year
    """
    df = build_ofm_df()
    years = df["year"].to_numpy(dtype=int)
    t = (years - BASE_YEAR).astype(int)  # Years since base year

    # Build demand components
    national_index = build_national_index(years, national_trend_pct_by_2035)
    behavior_index = build_behavior_index(years, behavior_headwind_pct_by_2035)
    enrollment_index = build_enrollment_index_constant(years, uw_enrollment_trend_pct)

    # Compute composite demand index
    # Code Review Note (v4.2): Extracted to separate function for clarity
    wa_share = float(clamp(wa_demand_share, 0.0, 1.0))
    demand_index = compute_demand_index(
        wa_18yo_index=df["wa_18yo_index"].to_numpy(dtype=float),
        national_index=national_index,
        enrollment_index=enrollment_index,
        behavior_index=behavior_index,
        wa_share=wa_share,
    )

    # Also compute demographic blend for display purposes
    demographic_index = (
        wa_share * df["wa_18yo_index"].to_numpy(dtype=float)
        + (1.0 - wa_share) * national_index
    )

    # Convert demand index to headcount, then cap by physical capacity
    base_headcount = float(
        RECONCILED_DATA["housing_portfolio"]["occupancy"]["current_headcount"]["value"]
    )
    demand_headcount = base_headcount * safe_div(demand_index, BASE_INDEX, default=0.0)

    capacity_headcount = build_capacity_headcount(years, haggett_net_beds)
    occupied_headcount = np.minimum(demand_headcount, capacity_headcount)
    occupied_headcount = clamp(occupied_headcount, 0.0, None)

    occupancy_index = (
        safe_div(occupied_headcount, base_headcount, default=0.0) * BASE_INDEX
    )

    # Revenue and expense indices with compound growth
    # Code Review Note (v4.2): Verified that compound growth is bounded by slider caps
    # Max expense growth (10%) over 20 years = 6.7x, which is realistic for stress testing
    revenue_index = occupancy_index * np.power(1.0 + float(rate_escalation), t)
    expense_index = BASE_INDEX * np.power(1.0 + float(expense_inflation), t)

    # Debt timing index
    debt_index = build_debt_index(
        years=years,
        shape=debt_shape,
        custom_peak_multiplier=custom_peak_multiplier,
        custom_peak_year=custom_peak_year,
    )

    # Net operating index (revenue minus expenses, as indices)
    exp_share = float(clamp(expense_share, 0.0, 1.0))
    net_operating_index = revenue_index - (exp_share * expense_index)

    # Rebase NOI to 2025 = 100 for stable ratio math
    net_operating_base = BASE_INDEX * (1.0 - exp_share)
    noi_index = (
        safe_div(net_operating_index, net_operating_base, default=np.nan) * BASE_INDEX
    )

    # DSCR estimation (anchored to base year actual)
    base_dscr = float(RECONCILED_DATA["financial_ratios"]["base_dscr"]["value"])
    dscr_est = base_dscr * safe_div(noi_index, debt_index, default=np.nan)

    # Safety margin calculation
    required_dscr = float(RECONCILED_DATA["financial_ratios"]["required_dscr"]["value"])
    base_cushion = base_dscr - required_dscr
    safety_margin_pct = (
        safe_div((dscr_est - required_dscr), base_cushion, default=np.nan) * 100.0
    )

    # Relative coverage ratio (diagnostic)
    debt_sh = float(clamp(debt_share, 0.0, 1.0))
    relative_coverage_ratio = safe_div(
        net_operating_index, (debt_sh * debt_index), default=np.nan
    )

    # Build output dataframe
    out = pd.DataFrame(
        {
            "year": years,
            "WA_18yo_Population": df["wa_18yo_population"].astype(int),
            "WA_18yo_Index": df["wa_18yo_index"].astype(float),
            "National_Global_Index": national_index.astype(float),
            "Enrollment_Index": enrollment_index.astype(float),
            "Behavior_Index": behavior_index.astype(float),
            "Demographic_Index": demographic_index.astype(float),
            "Demand_Index": demand_index.astype(float),
            "Capacity_Headcount_Cap": capacity_headcount.astype(float),
            "Capacity_Index": (
                safe_div(capacity_headcount, base_headcount, default=np.nan)
                * BASE_INDEX
            ).astype(float),
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
    out["Headroom_Above_Min_DSCR"] = out["DSCR_Est"] - required_dscr
    return out


# =============================================================================
# MODEL RESULT HELPERS
# =============================================================================


def value_at_year(
    df: pd.DataFrame, year: int, col: str, default: float = np.nan
) -> float:
    """Get value of a column at a specific year."""
    row = df[df["year"] == year]
    if row.empty:
        return default
    return float(row.iloc[0][col])


def first_breach_year(df: pd.DataFrame) -> int | None:
    """Find the first year where covenant is breached, or None if never breached."""
    future = df[df["year"] >= BASE_YEAR].copy()
    breach = future[future["Covenant_Breach"]]
    if breach.empty:
        return None
    return int(breach.iloc[0]["year"])


def worst_year_by(df: pd.DataFrame, col: str) -> int | None:
    """Find the year with the minimum value of a column."""
    s = df[["year", col]].copy()
    s = s[np.isfinite(s[col])]
    if s.empty:
        return None
    i = int(s[col].idxmin())
    return int(df.loc[i, "year"])


# =============================================================================
# UI HELPERS
# =============================================================================


def fmt_num(x: float, fmt: str, na: str = "n/a") -> str:
    """Format a number, returning na string if not finite."""
    return fmt.format(x) if np.isfinite(x) else na


def scenario_defaults_for(name: str) -> Dict[str, object]:
    """Get default parameter values for a scenario."""
    return dict(SCENARIOS.get(name, {}))


def apply_scenario_defaults(name: str) -> None:
    """Apply scenario defaults to session state."""
    defaults = scenario_defaults_for(name)
    for k, v in defaults.items():
        st.session_state[k] = v


def on_scenario_change() -> None:
    """Callback when scenario selection changes."""
    apply_scenario_defaults(st.session_state.get("scenario", "Baseline"))


def get_headroom_interpretation(headroom: float) -> str:
    """
    Get plain-English interpretation of headroom value.

    Code Review Addition (v4.2): Provides executive-friendly context for KPI values.

    Args:
        headroom: Headroom above minimum DSCR

    Returns:
        Emoji + interpretation string
    """
    if not np.isfinite(headroom):
        return "Unable to calculate"
    if headroom > 0.3:
        return "🟢 Comfortable buffer. Normal operations."
    elif headroom > 0.1:
        return "🟡 Adequate buffer. Continue monitoring."
    elif headroom > 0:
        return "🟠 Thin margin. Increased vigilance recommended."
    else:
        return "🔴 Below requirement. Corrective action needed."


def get_dscr_interpretation(dscr: float, required: float) -> str:
    """
    Get plain-English interpretation of DSCR value.

    Code Review Addition (v4.2): Provides executive-friendly context for KPI values.

    Args:
        dscr: Estimated DSCR value
        required: Required minimum DSCR

    Returns:
        Emoji + interpretation string
    """
    if not np.isfinite(dscr):
        return "Unable to calculate"
    ratio = dscr / required if required > 0 else 0
    if ratio >= 1.25:
        return f"🟢 {((ratio - 1) * 100):.0f}% above requirement"
    elif ratio >= 1.0:
        return f"🟡 {((ratio - 1) * 100):.0f}% above requirement"
    else:
        return f"🔴 {((1 - ratio) * 100):.0f}% below requirement"


# =============================================================================
# ALTAIR CHART HELPERS
# =============================================================================


def year_axis() -> alt.Axis:
    """Create consistent year axis with fixed ticks."""
    return alt.Axis(values=list(range(BASE_YEAR, END_YEAR + 1, 2)), format="d")


def x_year(title: str = "Year") -> alt.X:
    """
    Create X encoding for year with explicit domain.

    Code Review Note (v4.1/4.2): Explicit scale domain is critical for stable
    rendering across layered charts. Without this, the peak band layer can
    cause other layers to collapse.
    """
    return alt.X(
        "year:Q",
        title=title,
        axis=year_axis(),
        scale=alt.Scale(domain=[BASE_YEAR, END_YEAR], nice=False),
    )


def peak_band_layer(peak_window: Tuple[int, int] | None) -> alt.Chart | None:
    """
    Create debt-peak shading band for charts.

    Code Review Note (v4.1): The band MUST share the same x-scale domain as
    other layers. Without this, Vega-Lite can infer incorrect shared domain.

    Args:
        peak_window: Tuple of (start_year, end_year) or None

    Returns:
        Altair chart layer or None
    """
    if peak_window is None:
        return None

    band_df = pd.DataFrame(
        {"year_start": [peak_window[0]], "year_end": [peak_window[1]]}
    )
    return (
        alt.Chart(band_df)
        .mark_rect(opacity=0.08, color="#f59e0b")
        .encode(
            x=alt.X(
                "year_start:Q",
                scale=alt.Scale(domain=[BASE_YEAR, END_YEAR], nice=False),
                axis=None,
            ),
            x2=alt.X2("year_end:Q"),
        )
    )


# =============================================================================
# PAGE HEADER
# =============================================================================

st.title("UW HFS Housing Structural Risk Dashboard")
st.caption(
    "Index-based, outcome-neutral decision support. No currency values are computed or exported."
)

base_dscr = float(RECONCILED_DATA["financial_ratios"]["base_dscr"]["value"])
required_dscr = float(RECONCILED_DATA["financial_ratios"]["required_dscr"]["value"])

# About section with plain-English explanations
with st.expander("About This Model (Plain English)", expanded=False):
    st.markdown(
        f"""
**What is a bond covenant?**
A covenant is a promise in the bond agreement. For HFS, the key covenant requires us to maintain
adequate **coverage** for our debt payments.

**What is "coverage" (DSCR)?**
DSCR stands for **Debt Service Coverage Ratio**. Think of it as:
> "For every dollar we owe in debt payments, how many dollars do we have available?"

The bond requires us to keep this ratio above **{required_dscr:.2f}** (meaning we have ${required_dscr:.2f}
available for every $1.00 we owe).

**Why do we need a safety buffer?**
Because our income depends on enrollment, pricing, and costs, all of which can change. If coverage
drops below the minimum, it's a "covenant breach" which triggers remediation requirements.

**About the numbers in this model:**
All values are shown as **indices** (2025 = 100) rather than dollar amounts. This focuses attention
on trends and structural relationships rather than specific budget numbers.
        """.strip()
    )

with st.expander("How to Use This Dashboard", expanded=False):
    st.markdown(
        """
**Quick Start:**
1. Start with **Baseline** (should show healthy operations with no crisis)
2. Try **Structural Squeeze** to see cost pressure effects
3. Use **Bond Stress Test** to see what a breach scenario looks like

**Key Indicators:**
- **Coverage Ratio**: Above {:.2f} = meeting requirement
- **Safety Cushion**: Above 0 = meeting requirement, below 0 = breach

**Tips:**
- Use the "Focus Year" slider to examine specific planning horizons
- Click "Jump to worst year" to find the most vulnerable point
- The "Assumptions at a Glance" expander shows your current settings
        """.format(required_dscr).strip()
    )

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if "initialized" not in st.session_state:
    st.session_state["scenario"] = "Baseline"
    for k, v in SCENARIOS["Baseline"].items():
        st.session_state[k] = v
    st.session_state["initialized"] = True

# Migration safeguard from older versions (WA share may have been stored as decimal)
if "wa_demand_share" in st.session_state:
    v = st.session_state["wa_demand_share"]
    if isinstance(v, float) and 0.0 <= v <= 1.0:
        st.session_state["wa_demand_share"] = int(round(v * 100))

# Ensure new keys exist for users coming from older session state
if "uw_enrollment_trend_pct" not in st.session_state:
    st.session_state["uw_enrollment_trend_pct"] = 0

if "focus_year" not in st.session_state:
    st.session_state["focus_year"] = 2035


# =============================================================================
# SIDEBAR CONTROLS
# Code Review Note (v4.2): Updated labels and tooltips for executive audience
# =============================================================================

with st.sidebar:
    st.header("Controls")

    st.selectbox(
        "Scenario (One-Click Presets)",
        options=list(SCENARIOS.keys()),
        key="scenario",
        on_change=on_scenario_change,
        help="Load pre-configured assumptions for common planning scenarios.",
    )

    scenario_name = str(st.session_state.get("scenario", "Baseline"))
    st.info(SCENARIO_CONTEXT.get(scenario_name, "Scenario loaded."))

    # Code Review Fix (v4.2): Improved tooltip for Custom scenario
    defaults_exist = len(scenario_defaults_for(scenario_name)) > 0
    reset_help = (
        "Reverts sliders to defaults for the selected scenario."
        if defaults_exist
        else "Disabled for 'Custom' because it preserves your current settings."
    )
    if st.button(
        "Reset to Scenario Defaults",
        disabled=not defaults_exist,
        help=reset_help,
        use_container_width=True,
    ):
        apply_scenario_defaults(scenario_name)
        st.success("Scenario defaults reloaded.")

    st.subheader("Focus Year")

    st.slider(
        "Year for KPI Snapshot",
        min_value=BASE_YEAR,
        max_value=END_YEAR,
        step=1,
        key="focus_year",
        help=(
            "Controls which year is shown in the KPI cards at the top. "
            "Default is 2035 as a common 10-year planning horizon."
        ),
    )

    # -------------------------------------------------------------------------
    # Enrollment & Preference Section
    # Code Review Note (v4.2): Renamed for clarity, updated tooltips
    # -------------------------------------------------------------------------
    st.subheader("Student Demand Drivers")

    st.slider(
        "In-State vs. Out-of-State Mix",
        min_value=20,
        max_value=90,
        step=1,
        key="wa_demand_share",
        format="%d%% in-state",
        help=(
            "What portion of housing demand comes from Washington residents vs. "
            "out-of-state/international students?\n\n"
            "Example: 70% means 70% of demand follows WA population trends, "
            "30% follows national/international trends."
        ),
    )

    st.slider(
        "UW Class Size Change",
        min_value=-20,
        max_value=10,
        step=1,
        key="uw_enrollment_trend_pct",
        format="%d%%",
        help=(
            "Changes to freshman class size from UW policy decisions "
            "(budget cuts, admission selectivity, program changes).\n\n"
            "This is separate from demographic trends and housing preferences.\n\n"
            "Example: -10% means 10% fewer freshmen admitted across all years."
        ),
    )

    st.slider(
        "Out-of-State Enrollment Outlook (by 2035)",
        min_value=-50,
        max_value=50,
        step=1,
        key="national_trend_pct_by_2035",
        format="%d%%",
        help=(
            "Expected change in out-of-state/international student interest by 2035.\n\n"
            "Example: -10% means the out-of-state pipeline shrinks 10% by 2035, "
            "then holds steady after."
        ),
    )

    st.slider(
        "Housing Preference Shift (by 2035)",
        min_value=-50,
        max_value=20,
        step=1,
        key="behavior_headwind_pct_by_2035",
        format="%d%%",
        help=(
            "Change in the fraction of students who choose on-campus housing.\n\n"
            "Driven by: off-campus competition, remote learning, lifestyle preferences.\n\n"
            "Example: -15% means 15% fewer students choose housing by 2035, "
            "even if enrollment stays the same."
        ),
    )

    st.slider(
        "New Bed Capacity (effective 2027)",
        min_value=-500,
        max_value=1500,
        step=25,
        key="haggett_net_beds",
        help=(
            "Net change in housing capacity from construction projects.\n\n"
            "This only changes how many students CAN be housed. "
            "It does not create additional demand."
        ),
    )

    # -------------------------------------------------------------------------
    # Prices & Costs Section
    # -------------------------------------------------------------------------
    st.subheader("Prices & Costs")

    st.slider(
        "Annual Rent Increase",
        min_value=0.0,
        max_value=8.0,
        step=0.1,
        key="rate_escalation_pct",
        format="%.1f%%",
        help=(
            "Average annual increase in housing rates.\n\n"
            "Compounds year-over-year. Higher rates increase revenue "
            "but may affect demand if out of line with market."
        ),
    )

    st.slider(
        "Annual Cost Growth",
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        key="expense_inflation_pct",
        format="%.1f%%",
        help=(
            "Average annual increase in operating costs "
            "(salaries, utilities, supplies, maintenance).\n\n"
            "Compounds year-over-year. If cost growth exceeds revenue growth, "
            "coverage will decline."
        ),
    )

    # -------------------------------------------------------------------------
    # Debt Section
    # -------------------------------------------------------------------------
    st.subheader("Debt Payment Timing")

    st.selectbox(
        "Debt Service Pattern",
        options=DEBT_SHAPES,
        key="debt_shape",
        help=(
            "Models WHEN debt payments are higher or lower over time.\n\n"
            "• Flat: Steady payments throughout\n"
            "• Front-Loaded: Higher payments early, declining over time\n"
            '• The "Cliff": Payment spike 2030-2037 (e.g., major project)\n'
            "• Custom: Define your own peak window"
        ),
    )

    if st.session_state["debt_shape"] == "Custom":
        st.slider(
            "Peak Payment Multiplier",
            min_value=1.00,
            max_value=1.80,
            step=0.01,
            key="custom_peak_multiplier",
            help="How much higher are payments during the peak? (1.20 = 20% higher)",
        )
        st.slider(
            "Peak Start Year",
            min_value=2027,
            max_value=2040,
            step=1,
            key="custom_peak_year",
            help="First year of the 8-year peak payment window.",
        )
    elif st.session_state["debt_shape"] == 'The "Cliff" (Risk)':
        st.caption("Cliff: Payments are 20% higher from 2030-2037, then drop 20% from 2038 onward.")

    # -------------------------------------------------------------------------
    # Advanced Settings
    # -------------------------------------------------------------------------
    with st.expander("Advanced: Base Year Cost Structure", expanded=False):
        st.caption(
            "These settings define the starting cost structure. "
            "Change only if you have updated financial data."
        )
        st.slider(
            "Operating Expense Share",
            min_value=30,
            max_value=70,
            step=1,
            key="expense_share_pct",
            format="%d%%",
            help="Operating expenses as percentage of revenue in the base year.",
        )
        st.slider(
            "Debt Service Share (Reference)",
            min_value=10,
            max_value=50,
            step=1,
            key="debt_share_pct",
            format="%d%%",
            help=(
                "Debt service as percentage of revenue in the base year. "
                "Used for the relative coverage diagnostic."
            ),
        )

    with st.expander("Reference: Bond Requirements", expanded=False):
        st.write(f"**Current Coverage Ratio (2022 actual):** {base_dscr:.2f}")
        st.write(f"**Minimum Required (covenant):** {required_dscr:.2f}")
        st.write(
            f"**Current Safety Cushion:** {base_dscr - required_dscr:.2f} "
            f"({((base_dscr / required_dscr) - 1) * 100:.0f}% above minimum)"
        )


# =============================================================================
# COLLECT PARAMETERS AND RUN MODEL
# =============================================================================

params = {
    "wa_demand_share": float(st.session_state["wa_demand_share"]) / 100.0,
    "national_trend_pct_by_2035": float(st.session_state["national_trend_pct_by_2035"])
    / 100.0,
    "uw_enrollment_trend_pct": float(st.session_state["uw_enrollment_trend_pct"])
    / 100.0,
    "behavior_headwind_pct_by_2035": float(
        st.session_state["behavior_headwind_pct_by_2035"]
    )
    / 100.0,
    "haggett_net_beds": int(st.session_state["haggett_net_beds"]),
    "rate_escalation": float(st.session_state["rate_escalation_pct"]) / 100.0,
    "expense_inflation": float(st.session_state["expense_inflation_pct"]) / 100.0,
    "expense_share": float(st.session_state["expense_share_pct"]) / 100.0,
    "debt_share": float(st.session_state["debt_share_pct"]) / 100.0,
    "debt_shape": str(st.session_state["debt_shape"]),
    "custom_peak_multiplier": float(
        st.session_state.get("custom_peak_multiplier", 1.20)
    ),
    "custom_peak_year": int(st.session_state.get("custom_peak_year", 2030)),
}

df = run_model(**params)

# Extract focus year and key metrics
focus_year = int(st.session_state.get("focus_year", 2035))
focus_year = int(clamp(focus_year, BASE_YEAR, END_YEAR))

breach_year = first_breach_year(df)
worst_year = worst_year_by(df, "Headroom_Above_Min_DSCR")

peak_window = get_debt_peak_window(
    debt_shape=params["debt_shape"],
    custom_peak_year=params["custom_peak_year"],
    custom_peak_multiplier=params["custom_peak_multiplier"],
)

# Focus-year values
dscr_focus = value_at_year(df, focus_year, "DSCR_Est", default=np.nan)
headroom_focus = value_at_year(df, focus_year, "Headroom_Above_Min_DSCR", default=np.nan)

# Worst-year values
headroom_min = float(np.nanmin(df["Headroom_Above_Min_DSCR"].to_numpy(dtype=float)))

dscr_ok_focus = bool(np.isfinite(dscr_focus) and dscr_focus >= required_dscr)
headroom_ok_focus = bool(np.isfinite(headroom_focus) and headroom_focus >= 0.0)

# =============================================================================
# COMPLIANCE BANNER
# =============================================================================

if breach_year is None:
    st.success(
        f"✓ Bond requirement is met in all years ({BASE_YEAR}–{END_YEAR}) under current assumptions."
    )
else:
    st.error(
        f"⚠ Bond requirement is breached starting in **{breach_year}** under current assumptions. "
        "Review the charts below to identify the primary driver (demand, costs, or debt timing)."
    )

# =============================================================================
# KPI CARDS
# Code Review Update (v4.2): Added plain-English interpretations
# =============================================================================

k1, k2, k3, k4 = st.columns(4)

with k1:
    if breach_year is None:
        st.metric("Compliance Status", f"Healthy through {END_YEAR}")
    else:
        st.metric("Compliance Status", f"Breach in {breach_year}")
    st.caption("Based on current scenario assumptions")

with k2:
    st.metric(f"Coverage Ratio ({focus_year})", fmt_num(dscr_focus, "{:.2f}"))
    # Code Review Addition (v4.2): Plain-English interpretation
    st.caption(get_dscr_interpretation(dscr_focus, required_dscr))

with k3:
    st.metric(f"Safety Cushion ({focus_year})", fmt_num(headroom_focus, "{:+.2f}"))
    # Code Review Addition (v4.2): Plain-English interpretation
    st.caption(get_headroom_interpretation(headroom_focus))

with k4:
    st.metric("Worst-Year Cushion", fmt_num(headroom_min, "{:+.2f}"))
    if worst_year is not None:
        st.caption(f"Occurs in {worst_year}")

# Jump to worst year button
if worst_year is not None:
    if st.button("Jump to worst year (update Focus Year)", use_container_width=False):
        st.session_state["focus_year"] = worst_year
        st.rerun()

# Assumptions summary
with st.expander("Assumptions at a Glance", expanded=False):
    wa_pct = int(st.session_state["wa_demand_share"])
    nonwa_pct = 100 - wa_pct
    st.markdown(
        f"""
**Scenario:** {scenario_name}

**Student Demand:**
- Student mix: {wa_pct}% in-state, {nonwa_pct}% out-of-state
- UW class size change: {int(st.session_state["uw_enrollment_trend_pct"])}%
- Out-of-state outlook (by 2035): {int(st.session_state["national_trend_pct_by_2035"])}%
- Housing preference shift (by 2035): {int(st.session_state["behavior_headwind_pct_by_2035"])}%
- New bed capacity: {int(st.session_state["haggett_net_beds"]):+,} (effective 2027)

**Prices & Costs:**
- Annual rent increase: {float(st.session_state["rate_escalation_pct"]):.1f}%
- Annual cost growth: {float(st.session_state["expense_inflation_pct"]):.1f}%

**Debt Timing:** {st.session_state["debt_shape"]}
        """.strip()
    )
    if st.session_state["debt_shape"] == "Custom":
        st.markdown(
            f"- Custom peak: {float(st.session_state['custom_peak_multiplier']):.0%} "
            f"starting {int(st.session_state['custom_peak_year'])}"
        )

# =============================================================================
# DASHBOARD TAB
# =============================================================================

tabs = st.tabs(["Dashboard", "Data Export"])

with tabs[0]:
    # -------------------------------------------------------------------------
    # Demand / Occupancy Chart
    # -------------------------------------------------------------------------
    st.subheader("Housing Demand: Beds Filled Over Time")

    occ_df = df[["year", "Occupancy_Index", "Capacity_Index"]].copy()
    occ_df = occ_df.rename(
        columns={
            "Occupancy_Index": "Beds Filled",
            "Capacity_Index": "Bed Capacity",
        }
    )
    occ_long = occ_df.melt("year", var_name="Series", value_name="Index")

    if HAS_ALTAIR:
        band = peak_band_layer(peak_window)

        y_dom = padded_domain(
            occ_long["Index"].to_numpy(dtype=float),
            pad_abs=5.0,
            force_include=(100.0, 100.0),
            clamp_low=0.0,
        )

        base = (
            alt.Chart(occ_long)
            .mark_line(point=False)
            .encode(
                x=x_year(),
                y=alt.Y(
                    "Index:Q",
                    title="Index (2025 = 100)",
                    scale=alt.Scale(domain=y_dom) if y_dom else alt.Undefined,
                ),
                color=alt.Color("Series:N", title=""),
                tooltip=[
                    alt.Tooltip("year:Q", format="d", title="Year"),
                    "Series:N",
                    alt.Tooltip("Index:Q", format=".1f", title="Index"),
                ],
            )
            .properties(height=280)
        )

        ref_100 = (
            alt.Chart(pd.DataFrame({"y": [100.0]}))
            .mark_rule(color="#666", strokeDash=[4, 4])
            .encode(y="y:Q")
        )

        layers = []
        if band is not None:
            layers.append(band)
        layers.extend([base, ref_100])

        chart = alt.layer(*layers).resolve_scale(x="shared", y="shared")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.line_chart(occ_df.set_index("year"))

    st.caption(
        "If Beds Filled declines while costs rise, coverage becomes harder to maintain. "
        "The 100 line represents today's occupancy level."
    )

    # -------------------------------------------------------------------------
    # Operating Buffer (NOI) Chart
    # -------------------------------------------------------------------------
    st.subheader("Operating Buffer: Cash Available for Debt Payments")

    noi_df = df[["year", "NOI_Index"]].copy()
    noi_df = noi_df.rename(columns={"NOI_Index": "Operating Buffer"})

    if HAS_ALTAIR:
        band = peak_band_layer(peak_window)

        y_dom = padded_domain(
            noi_df["Operating Buffer"].to_numpy(dtype=float),
            pad_abs=10.0,
            force_include=(0.0, 100.0),
        )

        line = (
            alt.Chart(noi_df)
            .mark_line()
            .encode(
                x=x_year(),
                y=alt.Y(
                    "Operating Buffer:Q",
                    title="Index (100 = today's buffer)",
                    scale=alt.Scale(domain=y_dom) if y_dom else alt.Undefined,
                ),
                tooltip=[
                    alt.Tooltip("year:Q", format="d", title="Year"),
                    alt.Tooltip("Operating Buffer:Q", format=".1f", title="Buffer Index"),
                ],
            )
            .properties(height=260)
        )

        ref_100 = (
            alt.Chart(pd.DataFrame({"y": [100.0]}))
            .mark_rule(color="#666", strokeDash=[4, 4])
            .encode(y="y:Q")
        )
        ref_0 = (
            alt.Chart(pd.DataFrame({"y": [0.0]}))
            .mark_rule(color="#888")
            .encode(y="y:Q")
        )

        layers = []
        if band is not None:
            layers.append(band)
        layers.extend([line, ref_100, ref_0])

        chart = alt.layer(*layers).resolve_scale(x="shared", y="shared")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.line_chart(noi_df.set_index("year"))

    st.caption(
        "This shows how much cash is available for debt payments relative to today. "
        "Below 0 means operating costs exceed revenue (structural deficit)."
    )

    # -------------------------------------------------------------------------
    # Bond Compliance Charts
    # -------------------------------------------------------------------------
    st.subheader("Bond Compliance: Coverage Ratio vs. Requirement")

    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("**Coverage Ratio Over Time**")

        dscr_df = df[["year", "DSCR_Est"]].copy()
        dscr_df["Minimum Required"] = required_dscr
        dscr_df = dscr_df.rename(columns={"DSCR_Est": "Coverage Ratio"})
        dscr_long = dscr_df.melt("year", var_name="Series", value_name="Value")

        if HAS_ALTAIR:
            band = peak_band_layer(peak_window)

            y_dom = padded_domain(
                dscr_long["Value"].to_numpy(dtype=float),
                pad_abs=0.25,
                force_include=(required_dscr, required_dscr),
                clamp_low=0.0,
            )

            lines = (
                alt.Chart(dscr_long)
                .mark_line()
                .encode(
                    x=x_year(),
                    y=alt.Y(
                        "Value:Q",
                        title="Coverage Ratio",
                        scale=alt.Scale(domain=y_dom) if y_dom else alt.Undefined,
                    ),
                    color=alt.Color(
                        "Series:N",
                        title="",
                        sort=["Coverage Ratio", "Minimum Required"],
                    ),
                    strokeDash=alt.StrokeDash(
                        "Series:N",
                        title="",
                        sort=["Coverage Ratio", "Minimum Required"],
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("year:Q", format="d", title="Year"),
                        "Series:N",
                        alt.Tooltip("Value:Q", format=".2f"),
                    ],
                )
                .properties(height=280)
            )

            layers = []
            if band is not None:
                layers.append(band)
            layers.append(lines)

            if breach_year is not None:
                breach_rule = (
                    alt.Chart(pd.DataFrame({"year": [breach_year]}))
                    .mark_rule(color="#b91c1c", strokeDash=[6, 3], opacity=0.9)
                    .encode(x=x_year(title="Year"))
                )
                layers.append(breach_rule)

            chart = alt.layer(*layers).resolve_scale(x="shared", y="shared")
            st.altair_chart(chart, use_container_width=True)
        else:
            st.line_chart(
                dscr_df.set_index("year")[["Coverage Ratio", "Minimum Required"]]
            )

        st.caption(
            "Coverage Ratio must stay above the Minimum Required line. "
            "Crossing below triggers covenant breach."
        )

    with c_right:
        st.markdown("**Safety Cushion Over Time**")

        head_df = df[["year", "Headroom_Above_Min_DSCR"]].copy()
        head_df = head_df.rename(columns={"Headroom_Above_Min_DSCR": "Safety Cushion"})

        if HAS_ALTAIR:
            band = peak_band_layer(peak_window)

            y_dom = padded_domain(
                head_df["Safety Cushion"].to_numpy(dtype=float),
                pad_abs=0.25,
                force_include=(0.0, 0.0),
            )

            # Code Review Note (v4.2): Green/red shading with pattern consideration
            # Using opacity and line to ensure readability for colorblind users
            area_pos = (
                alt.Chart(head_df)
                .transform_filter(alt.datum["Safety Cushion"] >= 0)
                .mark_area(opacity=0.20, color="#16a34a")
                .encode(
                    x=x_year(),
                    y=alt.Y(
                        "Safety Cushion:Q",
                        title="Cushion (DSCR points above minimum)",
                        scale=alt.Scale(domain=y_dom) if y_dom else alt.Undefined,
                    ),
                    y2=alt.value(0),
                )
            )

            area_neg = (
                alt.Chart(head_df)
                .transform_filter(alt.datum["Safety Cushion"] < 0)
                .mark_area(opacity=0.20, color="#dc2626")
                .encode(
                    x=x_year(),
                    y=alt.Y("Safety Cushion:Q", title="Cushion (DSCR points above minimum)"),
                    y2=alt.value(0),
                )
            )

            line = (
                alt.Chart(head_df)
                .mark_line(color="#111827", strokeWidth=2)
                .encode(
                    x=x_year(),
                    y=alt.Y(
                        "Safety Cushion:Q",
                        title="Cushion (DSCR points above minimum)",
                        scale=alt.Scale(domain=y_dom) if y_dom else alt.Undefined,
                    ),
                    tooltip=[
                        alt.Tooltip("year:Q", format="d", title="Year"),
                        alt.Tooltip("Safety Cushion:Q", format="+.2f", title="Cushion"),
                    ],
                )
                .properties(height=280)
            )

            zero_rule = (
                alt.Chart(pd.DataFrame({"y": [0.0]}))
                .mark_rule(color="#444", strokeWidth=1.5)
                .encode(y="y:Q")
            )

            # Code Review Addition (v4.2): Label for zero line
            zero_label = (
                alt.Chart(pd.DataFrame({"year": [END_YEAR - 2], "y": [0.05], "text": ["← Minimum"]}))
                .mark_text(align="left", fontSize=10, color="#666")
                .encode(x="year:Q", y="y:Q", text="text:N")
            )

            layers = []
            if band is not None:
                layers.append(band)
            layers.extend([area_pos, area_neg, line, zero_rule, zero_label])

            if breach_year is not None:
                breach_rule = (
                    alt.Chart(pd.DataFrame({"year": [breach_year]}))
                    .mark_rule(color="#b91c1c", strokeDash=[6, 3], opacity=0.9)
                    .encode(x=x_year())
                )
                layers.append(breach_rule)

            chart = alt.layer(*layers).resolve_scale(x="shared", y="shared")
            st.altair_chart(chart, use_container_width=True)
        else:
            st.line_chart(head_df.set_index("year"))

        st.caption(
            "The simplest compliance signal: above 0 = meeting requirement, "
            "below 0 = breach. The black line shows the cushion over time."
        )

    # -------------------------------------------------------------------------
    # Optional: Revenue vs Cost Chart
    # -------------------------------------------------------------------------
    with st.expander("Additional Detail: Revenue vs. Cost Trends", expanded=False):
        sb_df = df[["year", "Revenue_Index", "Expense_Index"]].copy()
        sb_df = sb_df.rename(
            columns={"Revenue_Index": "Revenue", "Expense_Index": "Costs"}
        )
        sb_long = sb_df.melt("year", var_name="Series", value_name="Index")

        if HAS_ALTAIR:
            band = peak_band_layer(peak_window)

            y_dom = padded_domain(
                sb_long["Index"].to_numpy(dtype=float),
                pad_abs=10.0,
                force_include=(100.0, 100.0),
                clamp_low=0.0,
            )

            lines = (
                alt.Chart(sb_long)
                .mark_line()
                .encode(
                    x=x_year(),
                    y=alt.Y(
                        "Index:Q",
                        title="Index (2025 = 100)",
                        scale=alt.Scale(domain=y_dom) if y_dom else alt.Undefined,
                    ),
                    color=alt.Color("Series:N", title="", sort=["Revenue", "Costs"]),
                    tooltip=[
                        alt.Tooltip("year:Q", format="d", title="Year"),
                        "Series:N",
                        alt.Tooltip("Index:Q", format=".1f"),
                    ],
                )
                .properties(height=260)
            )

            ref_100 = (
                alt.Chart(pd.DataFrame({"y": [100.0]}))
                .mark_rule(color="#666", strokeDash=[4, 4])
                .encode(y="y:Q")
            )

            layers = []
            if band is not None:
                layers.append(band)
            layers.extend([lines, ref_100])

            chart = alt.layer(*layers).resolve_scale(x="shared", y="shared")
            st.altair_chart(chart, use_container_width=True)
        else:
            st.line_chart(sb_df.set_index("year"))

        st.caption(
            "Classic 'scissors' view: if Costs grow faster than Revenue, "
            "the gap erodes operating buffer over time."
        )


# =============================================================================
# DATA EXPORT TAB
# =============================================================================

with tabs[1]:
    st.subheader("Export Projection Data")
    st.caption(
        "All values are indices (2025 = 100) or derived ratios. "
        "No dollar amounts are included to maintain data privacy."
    )

    export_cols = [
        "year",
        "WA_18yo_Population",
        "WA_18yo_Index",
        "National_Global_Index",
        "Enrollment_Index",
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
        "Headroom_Above_Min_DSCR",
        "Safety_Margin_%",
        "Covenant_Breach",
    ]

    # Friendly column names for export
    export_rename = {
        "WA_18yo_Population": "WA 18yo Population",
        "WA_18yo_Index": "WA Demographic Index",
        "National_Global_Index": "Out-of-State Index",
        "Enrollment_Index": "Enrollment Policy Index",
        "Behavior_Index": "Housing Preference Index",
        "Demographic_Index": "Blended Demographic Index",
        "Demand_Index": "Total Demand Index",
        "Capacity_Index": "Capacity Index",
        "Occupancy_Index": "Occupancy Index",
        "Revenue_Index": "Revenue Index",
        "Expense_Index": "Expense Index",
        "Debt_Index": "Debt Timing Index",
        "NOI_Index": "Operating Buffer Index",
        "DSCR_Est": "Coverage Ratio (Est)",
        "Headroom_Above_Min_DSCR": "Safety Cushion",
        "Safety_Margin_%": "Safety Margin %",
        "Covenant_Breach": "Breach Flag",
    }

    display_df = df[export_cols].copy()

    round_map = {
        c: 2
        for c in export_cols
        if c not in {"year", "WA_18yo_Population", "Covenant_Breach"}
    }
    round_map.update(
        {
            "WA_18yo_Index": 1,
            "National_Global_Index": 1,
            "Enrollment_Index": 1,
            "Behavior_Index": 1,
            "Demographic_Index": 1,
            "Demand_Index": 1,
            "Capacity_Index": 1,
            "Occupancy_Index": 1,
            "Revenue_Index": 1,
            "Expense_Index": 1,
            "Debt_Index": 1,
            "NOI_Index": 1,
            "Safety_Margin_%": 1,
            "DSCR_Est": 2,
            "Headroom_Above_Min_DSCR": 2,
        }
    )

    display_df = display_df.round(round_map)
    display_df = display_df.rename(columns=export_rename)

    st.dataframe(display_df, use_container_width=True, height=420)

    csv_buffer = StringIO()
    display_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download CSV",
        data=csv_buffer.getvalue(),
        file_name="uw_hfs_housing_projection.csv",
        mime="text/csv",
    )

    with st.expander("Column Definitions", expanded=False):
        st.markdown(
            """
| Column | Description |
|--------|-------------|
| WA 18yo Population | Washington state 18-year-old population (OFM projection) |
| WA Demographic Index | WA population indexed to 2025 = 100 |
| Out-of-State Index | Out-of-state/international enrollment outlook index |
| Enrollment Policy Index | UW class size changes from policy decisions |
| Housing Preference Index | Fraction of students choosing on-campus housing |
| Blended Demographic Index | Weighted blend of WA and out-of-state |
| Total Demand Index | Combined demand from all factors |
| Capacity Index | Physical housing capacity |
| Occupancy Index | Actual beds filled (demand capped by capacity) |
| Revenue Index | Revenue from housing operations |
| Expense Index | Operating cost growth |
| Debt Timing Index | When debt payments are higher/lower |
| Operating Buffer Index | Cash available for debt service |
| Coverage Ratio (Est) | DSCR estimate (must stay above 1.25) |
| Safety Cushion | DSCR points above minimum requirement |
| Safety Margin % | Cushion as % of base-year cushion |
| Breach Flag | TRUE if coverage ratio below requirement |
            """
        )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.caption(
    "UW HFS Housing Structural Risk Model v4.2 | "
    "Index-based projections for strategic planning | "
    "Questions? Contact HFS Finance"
)
