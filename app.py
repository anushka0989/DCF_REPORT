"""
DCF Engine — Multi-Scenario Redevelopment Work Paper
Run with:  streamlit run app.py
"""

import io
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------
# THEME
# ----------------------------------------------------------------------
NAVY = "#182A44"
BRONZE = "#A5792C"
PAPER = "#F5F2E9"
PANEL = "#FFFFFF"
RULE = "#D8D2BE"
GREEN = "#3F6B4C"
RED = "#8B3A3A"
INK_SOFT = "#4B5567"

SCHEDULE_REF = {"residential": "Sch-1", "commercial": "Sch-2", "industrial": "Sch-3"}
LABELS = {"residential": "Residential", "commercial": "Commercial", "industrial": "Industrial"}

# 1 sq.m = 10.7639 sq.ft — internal calculations always run in sq.ft / ₹ per sq.ft.
SQM_TO_SQFT = 10.7639
UNIT_LABEL = {"sqft": "sq.ft", "sqm": "sq.m"}


def convert_area(value, from_unit, to_unit):
    if from_unit == to_unit:
        return value
    return value / SQM_TO_SQFT if to_unit == "sqm" else value * SQM_TO_SQFT


def convert_rate(value, from_unit, to_unit):
    """Converts a ₹-per-unit-area rate (sale rate / construction cost)."""
    if from_unit == to_unit:
        return value
    return value * SQM_TO_SQFT if to_unit == "sqm" else value / SQM_TO_SQFT

st.set_page_config(page_title="DCF Engine — Work Paper", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {PAPER}; }}
    h1, h2, h3, .stMarkdown, label, .stTabs [data-baseweb="tab"] {{
        font-family: Georgia, 'Times New Roman', serif !important;
        color: {NAVY};
    }}
    div[data-testid="stMetric"] {{
        background-color: {PANEL};
        border: 1px solid {RULE};
        border-radius: 3px;
        padding: 10px 14px;
    }}
    div[data-testid="stMetricLabel"] {{ color: {INK_SOFT} !important; font-size: 0.75rem !important; }}
    div[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace !important; }}
    .stTabs [aria-selected="true"] {{ background-color: {NAVY} !important; color: {PAPER} !important; }}
    .stButton>button {{
        background-color: {NAVY}; color: {PAPER}; border-radius: 2px; border: none;
        font-family: Georgia, serif;
    }}
    .schedule-ref {{ color: {BRONZE}; font-size: 0.7rem; letter-spacing: 0.06em; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# CALCULATION ENGINE
# ----------------------------------------------------------------------
def npv_at_rate(years, cfs, rate):
    return sum(cf / (1 + rate) ** y for y, cf in zip(years, cfs))


def calc_irr(years, cfs):
    """Bisection IRR solver — returns None if cash flow never changes sign."""
    low, high = -0.95, 5.0
    f_low, f_high = npv_at_rate(years, cfs, low), npv_at_rate(years, cfs, high)
    if f_low * f_high > 0:
        return None
    for _ in range(80):
        mid = (low + high) / 2
        f_mid = npv_at_rate(years, cfs, mid)
        if abs(f_mid) < 1:
            return mid
        if f_low * f_mid < 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return (low + high) / 2


def compute_schedule(sc, tenure, dev_rate, auth_rate, apply_escalation=True,
                      land_cost=0.0, land_payment_year=1, professional_fee_pct=0.0):
    area_per_constr_yr = sc["saleable_area"] / sc["construction_years"] if sc["construction_years"] else 0
    area_per_sales_yr = sc["saleable_area"] / sc["sales_years"] if sc["sales_years"] else 0

    rehab_area = sc.get("rehab_area", 0)
    rehab_cy = sc["construction_years"]  # rehab construction rides on the same construction window
    area_per_rehab_constr_yr = rehab_area / rehab_cy if rehab_cy else 0
    handover_year = sc.get("rehab_handover_year", rehab_cy) or rehab_cy

    # Professional fees are assessed on the (unescalated) base project cost — land + base construction —
    # consistent with how a feasibility estimate typically states them, and paid alongside land.
    base_construction_cost = sc["saleable_area"] * sc["construction_cost"] + rehab_area * sc.get("rehab_construction_cost", 0)
    professional_fee_amount = professional_fee_pct / 100 * (land_cost + base_construction_cost)

    rows = []
    for y in range(0, tenure + 1):
        constr_active = 1 <= y <= sc["construction_years"]
        cost_esc = (1 + sc["cost_escalation"] / 100) ** (y - 1) if apply_escalation else 1.0
        construction_cost = area_per_constr_yr * sc["construction_cost"] * cost_esc if constr_active else 0

        sales_active = sc["sales_start_year"] <= y < sc["sales_start_year"] + sc["sales_years"]
        price_esc = (1 + sc["price_escalation"] / 100) ** (y - 1) if apply_escalation else 1.0
        revenue = area_per_sales_yr * sc["sale_rate"] * price_esc if sales_active else 0

        other_costs = revenue * sc["other_cost_pct"] / 100
        gst = revenue * sc.get("gst_pct", 0) / 100
        other_revenue = sc.get("other_revenue", 0) * 1e7 if y == sc.get("other_revenue_year", 1) else 0

        authority_payment = 0.0
        if sc["authority_mode"] == "share":
            authority_payment = revenue * sc["revenue_share_pct"] / 100
        elif sc["authority_mode"] == "premium" and y == 1:
            authority_payment = sc["premium_amount"] * 1e7

        # --- Previous stakeholders (existing allottees) ---
        rehab_cost = area_per_rehab_constr_yr * sc.get("rehab_construction_cost", 0) * cost_esc if constr_active else 0
        rehab_revenue = rehab_area * sc.get("rehab_rate", 0) if y == handover_year else 0
        transit_compensation = sc.get("transit_compensation", 0) * 1e7 if y == sc.get("transit_year", 1) else 0

        # --- Land & professional fees ---
        land_cost_this_year = land_cost if y == land_payment_year else 0.0
        professional_fee_this_year = professional_fee_amount if y == land_payment_year else 0.0

        developer_cf = (revenue + other_revenue - construction_cost - other_costs - gst - authority_payment
                         + rehab_revenue - rehab_cost - transit_compensation
                         - land_cost_this_year - professional_fee_this_year)
        rows.append(dict(Year=y, Revenue=revenue, OtherRevenue=other_revenue, ConstructionCost=construction_cost,
                          OtherCosts=other_costs, GST=gst, AuthorityPayment=authority_payment,
                          RehabRevenue=rehab_revenue, RehabCost=rehab_cost,
                          TransitCompensation=transit_compensation, LandCost=land_cost_this_year,
                          ProfessionalFee=professional_fee_this_year, DeveloperCF=developer_cf))

    df = pd.DataFrame(rows)
    dev_r, auth_r = dev_rate / 100, auth_rate / 100
    npv_developer = sum(r.DeveloperCF / (1 + dev_r) ** r.Year for r in df.itertuples())
    npv_authority = sum(r.AuthorityPayment / (1 + auth_r) ** r.Year for r in df.itertuples())
    irr = calc_irr(df["Year"].tolist(), df["DeveloperCF"].tolist())

    df["DiscountedCF"] = [r.DeveloperCF / (1 + dev_r) ** r.Year for r in df.itertuples()]
    df["CumulativeDiscountedCF_Cr"] = (df["DiscountedCF"].cumsum() / 1e7).round(3)

    net_stakeholder_cost = df["RehabCost"].sum() + df["TransitCompensation"].sum() - df["RehabRevenue"].sum()

    return dict(rows=df, npv_developer=npv_developer, npv_authority=npv_authority, irr=irr,
                net_stakeholder_cost=net_stakeholder_cost, land_cost=land_cost,
                professional_fee=professional_fee_amount)


def default_scenario(area, rate, cost, cy, ss, sy):
    return dict(
        saleable_area=area, sale_rate=rate, price_escalation=5.0,
        construction_cost=cost, cost_escalation=6.0, construction_years=cy,
        sales_start_year=ss, sales_years=sy, other_cost_pct=8.0,
        authority_mode="share", revenue_share_pct=25.0, premium_amount=40.0,
        rehab_area=0.0, rehab_rate=0.0, rehab_construction_cost=cost,
        rehab_handover_year=cy, transit_compensation=0.0, transit_year=1,
        gst_pct=0.0, other_revenue=0.0, other_revenue_year=ss,
    )


# ----------------------------------------------------------------------
# SESSION STATE INIT
# ----------------------------------------------------------------------
if "project_name" not in st.session_state:
    st.session_state.project_name = "Redevelopment Project"
    st.session_state.tenure = 8
    st.session_state.developer_rate = 15.0
    st.session_state.authority_rate = 8.0
    st.session_state.land_area = 15000.0          # canonical sq.ft
    st.session_state.land_rate = 9300.0           # canonical ₹/sq.ft
    st.session_state.land_payment_year = 0
    st.session_state.professional_fee_pct = 0.0
    st.session_state.apply_escalation = True
    st.session_state.scenarios = {
        "residential": default_scenario(120000, 9000, 3200, 3, 2, 5),
        "commercial": default_scenario(80000, 12000, 4200, 3, 2, 4),
        "industrial": default_scenario(150000, 4500, 2200, 2, 2, 6),
    }

scenarios = st.session_state.scenarios


def cr(v):
    return f"{v / 1e7:,.2f}"


def pctfmt(v):
    return "n/a" if v is None else f"{v * 100:.1f}%"


# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.markdown(f"<div class='schedule-ref'>DISCOUNTED CASH FLOW — WORK PAPER</div>", unsafe_allow_html=True)
st.session_state.project_name = st.text_input("Project name", st.session_state.project_name, label_visibility="collapsed")
st.markdown(
    f"<div style='color:{INK_SOFT};font-family:IBM Plex Mono,monospace;font-size:12px;'>"
    f"Prepared {date.today().strftime('%d %b %Y')} · Multi-scenario redevelopment analysis</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

if "area_unit" not in st.session_state:
    st.session_state.area_unit = "sqft"
if "area_unit_prev" not in st.session_state:
    st.session_state.area_unit_prev = "sqft"

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.session_state.tenure = st.number_input("Project Tenure (yrs)", min_value=1, max_value=30,
                                                value=st.session_state.tenure, step=1,
                                                help="Every schedule starts at Year 0 (today, undiscounted) followed by Years 1 through this value.")
with c2:
    st.session_state.developer_rate = st.number_input("Developer Discount Rate (%)", min_value=0.0,
                                                        value=st.session_state.developer_rate, step=0.5)
with c3:
    st.session_state.authority_rate = st.number_input("Authority Discount Rate (%)", min_value=0.0,
                                                        value=st.session_state.authority_rate, step=0.5)
with c4:
    unit = st.radio("Area / Rate Unit", ["sqft", "sqm"],
                     index=["sqft", "sqm"].index(st.session_state.area_unit),
                     format_func=lambda u: UNIT_LABEL[u], horizontal=True, key="area_unit")

# If the unit was just switched, convert every already-entered area/rate widget value
# in place so the underlying figures (and every calculation) stay unchanged.
if unit != st.session_state.area_unit_prev:
    old_unit = st.session_state.area_unit_prev
    area_fields = ("area", "rehab_area")
    rate_fields = ("rate", "ccost", "rehab_rate", "rehab_ccost")
    for k in scenarios:
        for field in area_fields + rate_fields:
            wkey = f"{k}_{field}"
            if wkey in st.session_state:
                converter = convert_area if field in area_fields else convert_rate
                st.session_state[wkey] = converter(st.session_state[wkey], old_unit, unit)
    for wkey, converter in [("land_area_widget", convert_area), ("land_rate_widget", convert_rate)]:
        if wkey in st.session_state:
            st.session_state[wkey] = converter(st.session_state[wkey], old_unit, unit)
    st.session_state.area_unit_prev = unit

tenure = st.session_state.tenure
dev_rate = st.session_state.developer_rate
auth_rate = st.session_state.authority_rate
unit_label = UNIT_LABEL[unit]

st.markdown(f"**Land Cost & Statutory** &nbsp;<span class='schedule-ref'>applies once, across every scenario — same site</span>", unsafe_allow_html=True)
l1, l2, l3, l4, l5 = st.columns(5)
with l1:
    land_area_input = st.number_input(f"Land Area ({unit_label})",
                                       value=convert_area(st.session_state.land_area, "sqft", unit),
                                       step=100.0, key="land_area_widget")
    st.session_state.land_area = convert_area(land_area_input, unit, "sqft")
with l2:
    land_rate_input = st.number_input(f"Land Rate (₹/{unit_label})",
                                       value=convert_rate(st.session_state.land_rate, "sqft", unit),
                                       step=500.0, key="land_rate_widget")
    st.session_state.land_rate = convert_rate(land_rate_input, unit, "sqft")
with l3:
    st.session_state.land_payment_year = st.number_input("Land Payment Year", min_value=0,
                                                           value=int(st.session_state.land_payment_year), step=1,
                                                           help="0 = today (undiscounted), 1 = end of Year 1, etc.")
with l4:
    st.session_state.professional_fee_pct = st.number_input("Professional Fees (% of Land+Constr.)", min_value=0.0,
                                                              value=st.session_state.professional_fee_pct, step=0.5)
with l5:
    st.session_state.apply_escalation = st.checkbox("Apply Annual Escalation", value=st.session_state.apply_escalation,
                                                      help="Turn off to match a static, single-point feasibility estimate (no year-on-year price/cost growth).")

land_cost = st.session_state.land_area * st.session_state.land_rate
land_payment_year = st.session_state.land_payment_year
professional_fee_pct = st.session_state.professional_fee_pct
apply_escalation = st.session_state.apply_escalation
st.caption(f"Computed Land Cost: ₹ {cr(land_cost)} Cr, payable in Year {land_payment_year}")

# ----------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------
tabs = st.tabs(["Overview", "Residential · Sch-1", "Commercial · Sch-2", "Industrial · Sch-3", "Comparison · Sch-4"])

with tabs[0]:
    st.markdown(
        """
This work paper computes a discounted cash flow for a land redevelopment project across three land-use
scenarios — **Residential**, **Commercial** and **Industrial** — each on its own tab. Set the project
tenure and the two discount rates above: the **developer's rate** reflects commercial risk/WACC, the
**authority's rate** reflects a public-sector social discount rate.

Within each scenario, revenue is recognised over the sales window and construction cost over the
construction window, both escalated annually. The authority's return is modelled either as a percentage
revenue share each year, or a one-time upfront premium in Year 1 — set per scenario. Developer net cash
flow, NPV, and IRR are computed automatically; the Comparison tab ranks the three scenarios and lets you
export the full audit schedule to Excel.

*Figures are illustrative — replace the seeded defaults with values from your Knight Frank / Satguru
valuation reports before relying on the output.*
        """
    )

results = {}
scenario_tabs = {"residential": tabs[1], "commercial": tabs[2], "industrial": tabs[3]}

for key, tab in scenario_tabs.items():
    with tab:
        sc = scenarios[key]
        left, right = st.columns([1, 2])

        with left:
            st.markdown(f"**Assumptions — {LABELS[key]}**")
            area_input = st.number_input(f"Saleable Area ({unit_label})",
                                          value=convert_area(float(sc["saleable_area"]), "sqft", unit),
                                          step=1000.0 if unit == "sqft" else 100.0, key=f"{key}_area")
            sc["saleable_area"] = convert_area(area_input, unit, "sqft")

            rate_input = st.number_input(f"Sale Rate, Yr-1 (₹/{unit_label})",
                                          value=convert_rate(float(sc["sale_rate"]), "sqft", unit),
                                          step=100.0, key=f"{key}_rate")
            sc["sale_rate"] = convert_rate(rate_input, unit, "sqft")

            sc["price_escalation"] = st.number_input("Price Escalation (%/yr)", value=sc["price_escalation"], step=0.5, key=f"{key}_pesc")

            ccost_input = st.number_input(f"Construction Cost (₹/{unit_label})",
                                           value=convert_rate(float(sc["construction_cost"]), "sqft", unit),
                                           step=100.0, key=f"{key}_ccost")
            sc["construction_cost"] = convert_rate(ccost_input, unit, "sqft")
            sc["cost_escalation"] = st.number_input("Cost Escalation (%/yr)", value=sc["cost_escalation"], step=0.5, key=f"{key}_cesc")
            sc["construction_years"] = st.number_input("Construction Period (yrs)", min_value=1, value=int(sc["construction_years"]), step=1, key=f"{key}_cy")
            sc["sales_start_year"] = st.number_input("Sales Start (yr)", min_value=1, value=int(sc["sales_start_year"]), step=1, key=f"{key}_ss")
            sc["sales_years"] = st.number_input("Sales Duration (yrs)", min_value=1, value=int(sc["sales_years"]), step=1, key=f"{key}_sy")
            sc["other_cost_pct"] = st.number_input("Marketing/Other (% of rev)", value=sc["other_cost_pct"], step=0.5, key=f"{key}_other")

            mode_label = st.radio("Authority Arrangement", ["Revenue Share", "Upfront Premium"],
                                   index=0 if sc["authority_mode"] == "share" else 1, key=f"{key}_mode")
            sc["authority_mode"] = "share" if mode_label == "Revenue Share" else "premium"
            if sc["authority_mode"] == "share":
                sc["revenue_share_pct"] = st.number_input("Revenue Share to Authority (%)", value=sc["revenue_share_pct"], step=1.0, key=f"{key}_share")
            else:
                sc["premium_amount"] = st.number_input("Upfront Premium, Yr-1 (₹ Cr)", value=sc["premium_amount"], step=5.0, key=f"{key}_premium")

            st.markdown(f"**Previous Stakeholders — {LABELS[key]}**")
            st.caption("Existing allottees/lessees who surrender their current space and receive rehabilitation area at a concessional rate, separate from the market-rate sale above.")

            rehab_area_input = st.number_input(f"Rehabilitation Area ({unit_label})",
                                                value=convert_area(float(sc["rehab_area"]), "sqft", unit),
                                                step=1000.0 if unit == "sqft" else 100.0, key=f"{key}_rehab_area")
            sc["rehab_area"] = convert_area(rehab_area_input, unit, "sqft")

            rehab_rate_input = st.number_input(f"Rehabilitation Rate charged to Stakeholders (₹/{unit_label})",
                                                value=convert_rate(float(sc["rehab_rate"]), "sqft", unit),
                                                step=100.0, key=f"{key}_rehab_rate",
                                                help="Set to 0 for free rehabilitation (typical of most schemes).")
            sc["rehab_rate"] = convert_rate(rehab_rate_input, unit, "sqft")

            rehab_ccost_input = st.number_input(f"Rehabilitation Construction Cost (₹/{unit_label})",
                                                 value=convert_rate(float(sc["rehab_construction_cost"]), "sqft", unit),
                                                 step=100.0, key=f"{key}_rehab_ccost",
                                                 help="Cost to build the rehabilitation area — defaults to the same rate as market construction, adjust if specification differs.")
            sc["rehab_construction_cost"] = convert_rate(rehab_ccost_input, unit, "sqft")

            sc["rehab_handover_year"] = st.number_input("Rehabilitation Handover Year", min_value=1,
                                                          value=int(sc["rehab_handover_year"]), step=1, key=f"{key}_rehab_year",
                                                          help="Year the rehabilitation revenue (if any) is recognised — usually at end of construction, ahead of market sales.")
            sc["transit_compensation"] = st.number_input("One-time Transit/Shifting Compensation (₹ Cr)",
                                                           value=sc["transit_compensation"], step=1.0, key=f"{key}_transit",
                                                           help="Lump-sum paid to existing stakeholders for alternate accommodation during construction.")
            sc["transit_year"] = st.number_input("Transit Compensation Year", min_value=0,
                                                  value=int(sc["transit_year"]), step=1, key=f"{key}_transit_year")

            st.markdown(f"**Other Revenue & Levies — {LABELS[key]}**")
            sc["other_revenue"] = st.number_input("Other/Miscellaneous Revenue (₹ Cr)",
                                                    value=sc["other_revenue"], step=0.5, key=f"{key}_otherrev",
                                                    help="Car parking, amenity, or other saleable income not captured in the main sale rate above.")
            sc["other_revenue_year"] = st.number_input("Other Revenue — Year", min_value=1,
                                                         value=int(sc["other_revenue_year"]), step=1, key=f"{key}_otherrev_year")
            sc["gst_pct"] = st.number_input("GST / Statutory Levies (% of revenue)", min_value=0.0,
                                             value=sc["gst_pct"], step=0.5, key=f"{key}_gst")

        result = compute_schedule(sc, tenure, dev_rate, auth_rate, apply_escalation=apply_escalation,
                                   land_cost=land_cost, land_payment_year=land_payment_year,
                                   professional_fee_pct=professional_fee_pct)
        results[key] = result

        with right:
            m1, m2, m3 = st.columns(3)
            m1.metric("NPV — Developer", f"₹ {cr(result['npv_developer'])} Cr", f"@ {dev_rate:.1f}%")
            m2.metric("NPV — Authority", f"₹ {cr(result['npv_authority'])} Cr", f"@ {auth_rate:.1f}%")
            m3.metric("Developer IRR", pctfmt(result["irr"]))

            m4, m5 = st.columns(2)
            m4.metric("Land Cost (deducted)", f"₹ {cr(result['land_cost'])} Cr")
            if professional_fee_pct > 0:
                m5.metric("Professional Fees (deducted)", f"₹ {cr(result['professional_fee'])} Cr")

            if sc["rehab_area"] > 0 or sc["transit_compensation"] > 0:
                st.metric("Net Cost of Stakeholder Rehabilitation", f"₹ {cr(result['net_stakeholder_cost'])} Cr",
                          help="Rehab construction cost + transit compensation, net of any amount recovered from stakeholders. Already deducted from Developer NPV above.")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=result["rows"]["Year"], y=result["rows"]["CumulativeDiscountedCF_Cr"],
                fill="tozeroy", mode="lines", line=dict(color=BRONZE, width=2),
                fillcolor="rgba(165,121,44,0.18)", name="Cumulative Discounted CF",
            ))
            fig.add_hline(y=0, line_color=INK_SOFT, line_width=1)
            fig.update_layout(
                height=220, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                font=dict(family="IBM Plex Mono, monospace", size=11, color=INK_SOFT),
                xaxis=dict(title="Year", gridcolor=RULE), yaxis=dict(title="₹ Cr", gridcolor=RULE),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            cols_to_show = ["Year", "Revenue"]
            col_labels = ["Year", "Revenue (₹Cr)"]
            if sc["other_revenue"] > 0:
                cols_to_show.append("OtherRevenue")
                col_labels.append("Other Revenue (₹Cr)")
            cols_to_show += ["ConstructionCost", "OtherCosts"]
            col_labels += ["Construction (₹Cr)", "Marketing/Other (₹Cr)"]
            if sc["gst_pct"] > 0:
                cols_to_show.append("GST")
                col_labels.append("GST/Levies (₹Cr)")
            cols_to_show.append("AuthorityPayment")
            col_labels.append("Authority Pmt (₹Cr)")
            if sc["rehab_area"] > 0 or sc["transit_compensation"] > 0:
                cols_to_show += ["RehabRevenue", "RehabCost", "TransitCompensation"]
                col_labels += ["Rehab Revenue (₹Cr)", "Rehab Cost (₹Cr)", "Transit Comp. (₹Cr)"]
            if land_cost > 0:
                cols_to_show.append("LandCost")
                col_labels.append("Land Cost (₹Cr)")
            if professional_fee_pct > 0:
                cols_to_show.append("ProfessionalFee")
                col_labels.append("Professional Fee (₹Cr)")
            cols_to_show.append("DeveloperCF")
            col_labels.append("Developer Net CF (₹Cr)")

            display_df = result["rows"][cols_to_show].copy()
            for col in cols_to_show[1:]:
                display_df[col] = (display_df[col] / 1e7).round(2)
            display_df.columns = col_labels
            st.dataframe(display_df, use_container_width=True, hide_index=True)

with tabs[4]:
    # ensure results computed for all scenarios even if a tab wasn't visited
    for key in scenarios:
        if key not in results:
            results[key] = compute_schedule(scenarios[key], tenure, dev_rate, auth_rate, apply_escalation=apply_escalation,
                                              land_cost=land_cost, land_payment_year=land_payment_year,
                                              professional_fee_pct=professional_fee_pct)

    best = max(results, key=lambda k: results[k]["npv_developer"])
    st.markdown(f"**Highest developer NPV:** :green[{LABELS[best]}] (₹ {cr(results[best]['npv_developer'])} Cr)")

    comp_df = pd.DataFrame([
        dict(Scenario=LABELS[k], Ref=SCHEDULE_REF[k],
             NPV_Developer=round(results[k]["npv_developer"] / 1e7, 2),
             NPV_Authority=round(results[k]["npv_authority"] / 1e7, 2),
             NetStakeholderCost=round(results[k]["net_stakeholder_cost"] / 1e7, 2),
             IRR=pctfmt(results[k]["irr"]))
        for k in scenarios
    ])

    fig = go.Figure()
    fig.add_trace(go.Bar(x=comp_df["Scenario"], y=comp_df["NPV_Developer"], name="NPV Developer", marker_color=BRONZE))
    fig.add_trace(go.Bar(x=comp_df["Scenario"], y=comp_df["NPV_Authority"], name="NPV Authority", marker_color=NAVY))
    fig.update_layout(
        barmode="group", height=300, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(family="Georgia, serif", size=12, color=NAVY),
        yaxis=dict(title="₹ Cr", gridcolor=RULE), legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        comp_df.rename(columns={"NPV_Developer": "NPV Developer (₹Cr)", "NPV_Authority": "NPV Authority (₹Cr)",
                                 "NetStakeholderCost": "Net Stakeholder Cost (₹Cr)"}),
        use_container_width=True, hide_index=True,
    )

    # ---------------- Excel export ----------------
    def build_excel():
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            wb = writer.book
            title_fmt = wb.add_format({"bold": True, "font_size": 13, "font_color": NAVY})
            hdr_fmt = wb.add_format({"bold": True, "bg_color": NAVY, "font_color": "white", "border": 1})
            num_fmt = wb.add_format({"num_format": "#,##0.00", "border": 1})
            neg_fmt = wb.add_format({"num_format": "#,##0.00;(#,##0.00)", "border": 1, "font_color": RED})

            summary = comp_df.rename(columns={"NPV_Developer": "NPV Developer (₹Cr)", "NPV_Authority": "NPV Authority (₹Cr)",
                                               "NetStakeholderCost": "Net Stakeholder Cost (₹Cr)"})
            summary.to_excel(writer, sheet_name="Summary", index=False, startrow=4)
            ws = writer.sheets["Summary"]
            ws.write(0, 0, st.session_state.project_name, title_fmt)
            ws.write(1, 0, f"Project Tenure: {tenure} yrs")
            ws.write(2, 0, f"Developer Rate: {dev_rate}%  |  Authority Rate: {auth_rate}%  |  Area/Rate Unit: {unit_label}")
            for col_num, col_name in enumerate(summary.columns):
                ws.write(4, col_num, col_name, hdr_fmt)
            ws.set_column(0, 5, 20)

            for key in scenarios:
                df = results[key]["rows"].copy()
                money_cols = ["Revenue", "OtherRevenue", "ConstructionCost", "OtherCosts", "GST", "AuthorityPayment",
                              "RehabRevenue", "RehabCost", "TransitCompensation", "LandCost", "ProfessionalFee",
                              "DeveloperCF", "DiscountedCF"]
                for c in money_cols:
                    df[c] = (df[c] / 1e7).round(3)
                df = df.drop(columns=["CumulativeDiscountedCF_Cr"]).rename(columns={
                    "Revenue": "Revenue (₹Cr)", "OtherRevenue": "Other Revenue (₹Cr)",
                    "ConstructionCost": "Construction Cost (₹Cr)",
                    "OtherCosts": "Marketing/Other (₹Cr)", "GST": "GST/Levies (₹Cr)",
                    "AuthorityPayment": "Authority Payment (₹Cr)",
                    "RehabRevenue": "Rehab Revenue (₹Cr)", "RehabCost": "Rehab Construction Cost (₹Cr)",
                    "TransitCompensation": "Transit Compensation (₹Cr)",
                    "LandCost": "Land Cost (₹Cr)", "ProfessionalFee": "Professional Fee (₹Cr)",
                    "DeveloperCF": "Developer Net CF (₹Cr)", "DiscountedCF": "Discounted CF (₹Cr)",
                })
                sheet_name = SCHEDULE_REF[key]
                df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)
                wsx = writer.sheets[sheet_name]
                wsx.write(0, 0, f"{LABELS[key]} — {sheet_name}", title_fmt)
                for col_num, col_name in enumerate(df.columns):
                    wsx.write(2, col_num, col_name, hdr_fmt)
                wsx.set_column(0, len(df.columns) - 1, 18)
                r = results[key]
                last_row = 3 + len(df)
                wsx.write(last_row, 0, "NPV Developer (₹Cr)")
                wsx.write(last_row, 1, round(r["npv_developer"] / 1e7, 2))
                wsx.write(last_row + 1, 0, "NPV Authority (₹Cr)")
                wsx.write(last_row + 1, 1, round(r["npv_authority"] / 1e7, 2))
                wsx.write(last_row + 2, 0, "Net Stakeholder Rehabilitation Cost (₹Cr)")
                wsx.write(last_row + 2, 1, round(r["net_stakeholder_cost"] / 1e7, 2))
                wsx.write(last_row + 3, 0, "Land Cost (₹Cr)")
                wsx.write(last_row + 3, 1, round(r["land_cost"] / 1e7, 2))
                wsx.write(last_row + 4, 0, "Professional Fee (₹Cr)")
                wsx.write(last_row + 4, 1, round(r["professional_fee"] / 1e7, 2))
                wsx.write(last_row + 5, 0, "Developer IRR")
                wsx.write(last_row + 5, 1, pctfmt(r["irr"]))
        buffer.seek(0)
        return buffer

    excel_bytes = build_excel()
    st.download_button(
        "⬇ Export to Excel (audit schedule)",
        data=excel_bytes,
        file_name=f"{st.session_state.project_name.replace(' ', '_')}_DCF_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.markdown(
    f"<div style='text-align:center;color:{INK_SOFT};font-family:IBM Plex Mono,monospace;font-size:10px;margin-top:20px;'>"
    "Internal work paper — all figures in Indian Rupees, ₹ in Crore unless stated. "
    "NPV computed on annual discrete cash flows.</div>",
    unsafe_allow_html=True,
)