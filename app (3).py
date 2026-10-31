import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import datetime
import io
import math
import xlsxwriter
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Set page config
st.set_page_config(
    page_title="Hispanic Heritage Historical Data Explorer & STEM Lab",
    page_icon="🌎",
    layout="wide"
)

# --- CONSTANTS & MAPPINGS ---
CURRENT_YEAR = datetime.datetime.now().year
DEFAULT_START_YEAR = CURRENT_YEAR - 70
DEFAULT_END_YEAR = CURRENT_YEAR - 1

COUNTRIES = {
    "MX": "Mexico",
    "BR": "Brazil (Latin American Comparison)",
    "AR": "Argentina",
    "CO": "Colombia",
    "CL": "Chile",
    "PE": "Peru",
    "VE": "Venezuela",
    "EC": "Ecuador",
    "UY": "Uruguay"
}

INDICATORS = {
    "SP.DYN.LE00.IN": {"name": "Life Expectancy at Birth", "unit": "Years"},
    "SL.UEM.TOTL.NE.ZS": {"name": "Unemployment Rate (National Estimate)", "unit": "% of Total Labor Force"},
    "NY.GDP.PCAP.CD": {"name": "GDP per Capita (Current US$)", "unit": "Current US$"},
    "SP.DYN.CBRT.IN": {"name": "Crude Birth Rate", "unit": "Per 1,000 People"},
    "SM.POP.NETM": {"name": "Net Migration", "unit": "Net Total People"},
    "VC.IHR.PSRC.P5": {"name": "Intentional Homicide Rate", "unit": "Per 100,000 People"},
    "FP.CPI.TOTL.ZG": {"name": "Consumer Price Inflation", "unit": "Annual %"},
    "GC.DOD.TOTL.GD.ZS": {"name": "Central Government Debt", "unit": "% of GDP"},
    "FI.RES.TOTL.CD": {"name": "Total Reserves (includes Gold)", "unit": "Current US$"},
    "NE.RSB.GNFS.ZS": {"name": "External Balance on Goods and Services", "unit": "% of GDP"}
}

CONTEXT_PRESETS = {
    "Economic Forecasting": {
        "var_x": "Time (Years from Base)",
        "var_y": "Indicator Output",
        "unit_x": "Years",
        "unit_y": "Units",
        "desc": "Analysis of macroeconomic trends and long-term indicators."
    },
    "Business": {
        "var_x": "Time / Units Produced",
        "var_y": "Revenue / Profit",
        "unit_x": "Months",
        "unit_y": "USD ($)",
        "desc": "Optimizing business profit margins and evaluating marginal revenue."
    },
    "Engineering": {
        "var_x": "Load / Time",
        "var_y": "Stress / Displacement",
        "unit_x": "Seconds",
        "unit_y": "Meters",
        "desc": "Structural dynamics and physical system modeling."
    },
    "Medical": {
        "var_x": "Time Post-Administration",
        "var_y": "Drug Concentration",
        "unit_x": "Hours",
        "unit_y": "mg/L",
        "desc": "Pharmacokinetics and physiological clearance modeling."
    },
    "Sports Science": {
        "var_x": "Angle / Time",
        "var_y": "Joint Torque / Velocity",
        "unit_x": "Seconds",
        "unit_y": "Nm",
        "desc": "Biomechanics, prosthetic joint trajectory, and athletic performance."
    },
    "Biology / Bioengineering": {
        "var_x": "Time",
        "var_y": "Cell Population",
        "unit_x": "Hours",
        "unit_y": "Cells/mL",
        "desc": "Biomass growth dynamics and substrate utilization."
    },
    "Weather / Climate": {
        "var_x": "Time",
        "var_y": "Temperature / Anomaly",
        "unit_x": "Days",
        "unit_y": "°C",
        "desc": "Atmospheric variations and seasonal trend analysis."
    },
    "Custom": {
        "var_x": "Independent Variable (x)",
        "var_y": "Dependent Variable (y)",
        "unit_x": "x-units",
        "unit_y": "y-units",
        "desc": "User-defined mathematical dynamic model."
    }
}

# --- WORLD BANK DATA FETCHING ---
@st.cache_data(ttl=3600)
def fetch_world_bank_data(country_code, indicator_code, start_year, end_year):
    url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?date={start_year}:{end_year}&format=json&per_page=1000"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if len(data) > 1 and data[1]:
                records = []
                for entry in data[1]:
                    yr = int(entry['date'])
                    val = entry['value']
                    if val is not None:
                        records.append({"Year": yr, "Value": float(val)})
                df = pd.DataFrame(records).sort_values("Year").reset_index(drop=True)
                return df, None
            else:
                return pd.DataFrame(columns=["Year", "Value"]), "No observations available for this selection."
        else:
            return pd.DataFrame(columns=["Year", "Value"]), f"API Error: HTTP {res.status_code}"
    except Exception as e:
        return pd.DataFrame(columns=["Year", "Value"]), f"Request failed: {str(e)}"

# --- POLYNOMIAL FITTING & DERIVATIVES ---
def fit_polynomial(df, degree):
    if len(df) < degree + 1:
        return None
    base_year = int(df["Year"].min())
    t = df["Year"].values - base_year
    y = df["Value"].values
    
    coeffs = np.polyfit(t, y, degree)
    poly = np.poly1d(coeffs)
    poly_d1 = np.polyder(poly, 1)
    poly_d2 = np.polyder(poly, 2)
    
    y_pred = poly(t)
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    mae = np.mean(np.abs(residuals))
    
    return {
        "coeffs": coeffs,
        "poly": poly,
        "poly_d1": poly_d1,
        "poly_d2": poly_d2,
        "base_year": base_year,
        "r2": r2,
        "mae": mae,
        "residuals": residuals,
        "mean_y": np.mean(y),
        "count": len(y),
        "min_year": df["Year"].min(),
        "max_year": df["Year"].max()
    }

def format_poly_latex(coeffs, var_name="t"):
    degree = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        p = degree - i
        if abs(c) < 1e-12:
            continue
        c_str = f"{c:+.4e}" if abs(c) < 0.001 or abs(c) > 10000 else f"{c:+.4f}"
        if p == 0:
            terms.append(f"{c_str}")
        elif p == 1:
            terms.append(f"{c_str} \cdot {var_name}")
        else:
            terms.append(f"{c_str} \cdot {var_name}^{{{p}}}")
    return " ".join(terms) if terms else "0"

# --- MAIN APP LAYOUT ---
def main():
    st.title("🌎 Hispanic Heritage Historical Data Explorer & STEM Lab")
    st.caption("Authentic World Bank Data | Multi-Pathway Mathematical Analysis | Calculus & Visual Simulations")
    
    # Sidebar
    st.sidebar.header("🕹️ Control Panel")
    
    pathway = st.sidebar.selectbox(
        "Assigned Mathematics Pathway",
        ["Calculus", "Precalculus", "Honors Algebra 2"]
    )
    
    country_code = st.sidebar.selectbox(
        "Select Country",
        options=list(COUNTRIES.keys()),
        format_func=lambda x: f"{COUNTRIES[x]} ({x})"
    )
    
    indicator_code = st.sidebar.selectbox(
        "Select Indicator",
        options=list(INDICATORS.keys()),
        format_func=lambda x: INDICATORS[x]["name"]
    )
    
    st.sidebar.subheader("Model & Sampling Controls")
    degree = st.sidebar.slider("Polynomial Fit Degree (d)", min_value=3, max_value=8, value=3)
    sample_inc = st.sidebar.slider("Graph Sample Increment (Years)", min_value=1, max_value=10, value=1)
    horizon = st.sidebar.slider("Forward Horizon (Years)", min_value=0, max_value=50, value=15)
    
    compare_country = st.sidebar.selectbox(
        "Comparison Country (Same Indicator)",
        options=["None"] + list(COUNTRIES.keys()),
        format_func=lambda x: "None" if x == "None" else f"{COUNTRIES[x]} ({x})"
    )
    
    printer_mode = st.sidebar.checkbox("Printer-Friendly View Panel")
    
    # Data Fetching
    raw_df, err = fetch_world_bank_data(country_code, indicator_code, DEFAULT_START_YEAR, DEFAULT_END_YEAR)
    
    if err:
        st.warning(f"World Bank API Notice: {err}")
    
    st.subheader(f"1. Historical Data: {COUNTRIES[country_code]} — {INDICATORS[indicator_code]['name']}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**Indicator Unit:** {INDICATORS[indicator_code]['unit']}")
        st.markdown(f"**Observed Count:** {len(raw_df)} observations recorded between {DEFAULT_START_YEAR} and {DEFAULT_END_YEAR}.")
        st.markdown(f"**Official Source:** [World Bank Open Data API](http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}) | Retrieved: {datetime.date.today().strftime('%B %Y')}")
    
    if "data_table" not in st.session_state or st.session_state.get("current_key") != f"{country_code}_{indicator_code}":
        st.session_state.data_table = raw_df.copy()
        st.session_state.current_key = f"{country_code}_{indicator_code}"
        st.session_state.is_edited = False

    with col2:
        if st.button("Reset Data to Source"):
            st.session_state.data_table = raw_df.copy()
            st.session_state.is_edited = False
            st.rerun()

    st.write("#### Validated Data Table (Editable)")
    if st.session_state.is_edited:
        st.info("⚠️ Note: You are currently working with a User-Modified Scenario.")
        
    edited_df = st.data_editor(
        st.session_state.data_table,
        num_rows="dynamic",
        key="editor"
    )
    
    if not edited_df.equals(st.session_state.data_table):
        st.session_state.data_table = edited_df
        st.session_state.is_edited = True
        st.rerun()

    working_df = st.session_state.data_table.dropna().sort_values("Year").reset_index(drop=True)
    
    if len(working_df) < degree + 1:
        st.error(f"Insufficient data to fit polynomial of degree {degree}. Need at least {degree+1} observations, but got {len(working_df)}.")
        return

    fit_res = fit_polynomial(working_df, degree)
    poly = fit_res["poly"]
    base_year = fit_res["base_year"]

    st.divider()
    st.header("2. Mathematical Model & Fit Analysis")
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Degree (d)", degree)
    m_col2.metric("In-Sample R²", f"{fit_res['r2']:.4f}")
    m_col3.metric("In-Sample MAE", f"{fit_res['mae']:.4f}")
    m_col4.metric("Base Year (t=0)", base_year)

    poly_latex = format_poly_latex(fit_res["coeffs"])
    st.latex(f"f(t) = {poly_latex} \quad \text{{where }} t = \text{{Year}} - {base_year}")

    st.divider()
    st.header("3. STEM Context & Calculus Interactive Lab")
    
    ctx_choice = st.selectbox("Select STEM Context", list(CONTEXT_PRESETS.keys()))
    ctx_info = CONTEXT_PRESETS[ctx_choice]
    
    st.info(f"**Context Description:** {ctx_info['desc']}")
    
    c_col1, c_col2 = st.columns(2)
    eval_year = c_col1.slider("Evaluate Year (Single Point Evaluation)", 1950, 2100, DEFAULT_END_YEAR)
    eval_t = eval_year - base_year
    
    val_est = poly(eval_t)
    d1_est = fit_res["poly_d1"](eval_t)
    d2_est = fit_res["poly_d2"](eval_t)

    c_col2.metric(f"Estimated {INDICATORS[indicator_code]['name']} ({eval_year})", f"{val_est:.2f} {INDICATORS[indicator_code]['unit']}")
    c_col2.metric("First Derivative f'(t) [Velocity/Rate]", f"{d1_est:.4f} {INDICATORS[indicator_code]['unit']}/yr")
    c_col2.metric("Second Derivative f''(t) [Acceleration]", f"{d2_est:.4f} {INDICATORS[indicator_code]['unit']}/yr²")

    st.subheader(f"Mathematical Analysis ({pathway} Mode)")
    if pathway == "Honors Algebra 2":
        st.markdown(f"""
        - **Domain:** Validated historical range [{fit_res['min_year']}, {fit_res['max_year']}].
        - **End Behavior:** As $t \to \infty$, $f(t) \to {'+\infty' if fit_res['coeffs'][0] > 0 else '-\infty'}$.
        - **Average Rate of Change:** Measure of overall growth/decline across the observation window.
        """)
    elif pathway == "Precalculus":
        st.markdown(f"""
        - **Polynomial Degree Impact:** Degree {degree} allows up to {degree-1} turning points.
        - **Concavity Trend:** $f''(t) > 0$ indicates accelerating growth (concave up), while $f''(t) < 0$ indicates decelerating growth (concave down).
        """)
    else:
        st.markdown(f"""
        - **Derivative Interpretation:** $f'({eval_year}) = {d1_est:.4f}$ represents the instantaneous rate of change of {INDICATORS[indicator_code]['name']} in {eval_year}.
        - **Concavity & Inflection:** $f''({eval_year}) = {d2_est:.4f}$ describes whether the rate of change is increasing or decreasing.
        """)

    st.divider()
    st.header("4. Visualizations & Simulations")

    years_sampled = working_df["Year"].values[::sample_inc]
    values_sampled = working_df["Value"].values[::sample_inc]
    
    t_smooth = np.linspace(working_df["Year"].min() - base_year, working_df["Year"].max() + horizon - base_year, 300)
    years_smooth = t_smooth + base_year
    y_smooth = poly(t_smooth)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(years_sampled, values_sampled, color="blue", label="Sampled Observations", zorder=3)
    ax.plot(years_smooth, y_smooth, color="red", label=f"Polynomial Fit (Degree {degree})", linewidth=2)
    ax.axvline(x=fit_res['max_year'], color='gray', linestyle='--', label='Historical Data Boundary')
    ax.plot(eval_year, val_est, 'go', markersize=10, label=f'Evaluated Point ({eval_year})')

    if compare_country != "None" and compare_country != country_code:
        comp_df, _ = fetch_world_bank_data(compare_country, indicator_code, DEFAULT_START_YEAR, DEFAULT_END_YEAR)
        if not comp_df.empty:
            ax.plot(comp_df["Year"], comp_df["Value"], color="green", linestyle=":", label=f"Comparison: {COUNTRIES[compare_country]}")

    ax.set_xlabel("Year")
    ax.set_ylabel(INDICATORS[indicator_code]['unit'])
    ax.set_title(f"{INDICATORS[indicator_code]['name']} Trend & Extrapolation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.divider()
    st.header("5. Export Reports & Data")
    
    exp_col1, exp_col2, exp_col3 = st.columns(3)

    csv_data = working_df.to_csv(index=False).encode('utf-8')
    exp_col1.download_button("📥 Download Data (CSV)", csv_data, f"{country_code}_{indicator_code}.csv", "text/csv")

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        working_df.to_excel(writer, sheet_name='Working Data', index=False)
        summary_df = pd.DataFrame([
            {"Metric": "Country", "Value": COUNTRIES[country_code]},
            {"Metric": "Indicator", "Value": INDICATORS[indicator_code]['name']},
            {"Metric": "Polynomial Degree", "Value": degree},
            {"Metric": "R-squared", "Value": fit_res['r2']},
            {"Metric": "MAE", "Value": fit_res['mae']},
            {"Metric": "Evaluated Year", "Value": eval_year},
            {"Metric": "Evaluated Value", "Value": val_est}
        ])
        summary_df.to_excel(writer, sheet_name='Model Summary', index=False)
    exp_col2.download_button("📥 Download Workbook (.xlsx)", excel_buffer.getvalue(), f"{country_code}_{indicator_code}_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Hispanic Heritage Historical Data & STEM Calculus Report", styles['Title']),
        Spacer(1, 12),
        Paragraph(f"<b>Country:</b> {COUNTRIES[country_code]}", styles['Normal']),
        Paragraph(f"<b>Indicator:</b> {INDICATORS[indicator_code]['name']}", styles['Normal']),
        Paragraph(f"<b>Mathematics Pathway:</b> {pathway}", styles['Normal']),
        Paragraph(f"<b>Polynomial Degree:</b> {degree}", styles['Normal']),
        Paragraph(f"<b>R-squared:</b> {fit_res['r2']:.4f} | <b>MAE:</b> {fit_res['mae']:.4f}", styles['Normal']),
        Spacer(1, 12),
        Paragraph(f"<b>Evaluated Year ({eval_year}):</b> {val_est:.2f} {INDICATORS[indicator_code]['unit']}", styles['Heading2']),
        Paragraph(f"f'({eval_year}) = {d1_est:.4f} | f''({eval_year}) = {d2_est:.4f}", styles['Normal']),
        Spacer(1, 12),
        Paragraph("Report generated automatically by Hispanic Heritage Data Explorer App.", styles['Italic'])
    ]
    doc.build(story)
    exp_col3.download_button("📥 Download Summary PDF", pdf_buffer.getvalue(), f"{country_code}_{indicator_code}_report.pdf", "application/pdf")

if __name__ == "__main__":
    main()
