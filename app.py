import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="SolarScope | Space Weather Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
    :root { color-scheme: dark; }
    .stApp { background: radial-gradient(ellipse at top left, #10233d 0%, #080d18 45%, #070a12 100%); }
    .block-container {padding-top: 1.25rem; padding-bottom: 2.5rem; max-width: 1500px;}
    [data-testid="stSidebar"] {background: #0b1322; border-right: 1px solid #1d304a;}
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {color: #e6f1ff;}
    .hero {
        padding: 1.7rem 1.9rem; border-radius: 20px;
        background: linear-gradient(115deg, #101b33 0%, #173d64 55%, #116b78 100%);
        border: 1px solid rgba(132, 202, 255, .24);
        color: white; margin-bottom: 1.1rem;
        box-shadow: 0 12px 34px rgba(0,0,0,.22);
    }
    .hero h1 {margin: 0; font-size: 2.45rem; letter-spacing: -.04em;}
    .hero p {margin: .45rem 0 0 0; color: #d5e8ff; font-size: 1rem;}
    .hero .eyebrow {text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; color: #7ee7d5; font-weight: 700; margin-bottom: .5rem;}
    [data-testid="stMetric"] {background: rgba(18, 33, 54, .78); border: 1px solid #243b59; padding: 14px 16px; border-radius: 14px;}
    [data-testid="stMetricLabel"] {color: #a9c2df;}
    [data-testid="stMetricValue"] {color: #f3f8ff;}
    div[data-baseweb="tab-list"] {gap: 8px;}
    button[data-baseweb="tab"] {background: #111d30; border-radius: 10px 10px 0 0; padding: 10px 16px;}
    .small-note {font-size: .85rem; color: #9eb1c9;}
    div[data-testid="stAlert"] {border-radius: 12px;}
    .section-note {color: #a9c2df; font-size: .93rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">SPACE WEATHER · DATA INTELLIGENCE</div>
  <h1>☀️ SolarScope</h1>
  <p>Explore geomagnetic activity, uncover solar patterns, and evaluate time-series forecasts.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Data loading ----------
DEFAULT_FILE = Path(__file__).resolve().parent / "kpdata.txt"

@st.cache_data(show_spinner=False)
def load_gfz_file(file_bytes: bytes) -> pd.DataFrame:
    """Load the GFZ combined Kp/ap/Ap/SN/F10.7 whitespace-delimited format."""
    raw = pd.read_csv(
        io.BytesIO(file_bytes),
        sep=r"\s+",
        header=None,
        comment="#",
        engine="python",
    )
    if raw.shape[1] < 28:
        raise ValueError(
            f"Expected at least 28 columns for the selected GFZ combined format; "
            f"found {raw.shape[1]}. Please check that you uploaded the combined "
            f"'Geomagnetic and solar indices (Kp, ap, Ap, SN, F10.7)' file."
        )

    data = pd.DataFrame()
    data["Date"] = pd.to_datetime(
        {"year": raw[0], "month": raw[1], "day": raw[2]},
        errors="coerce",
    )
    # GFZ combined format: eight 3-hour Kp values (7–14), eight ap values (15–22),
    # daily Ap (23), sunspot number (24), F10.7 (25), centered 81-day F10.7 (26),
    # and a quality/status flag (27).
    data["Ap"] = pd.to_numeric(raw[23], errors="coerce")
    data["Sunspot Number"] = pd.to_numeric(raw[24], errors="coerce")
    data["F10.7 Flux"] = pd.to_numeric(raw[25], errors="coerce")
    data["F10.7 81-day Mean"] = pd.to_numeric(raw[26], errors="coerce")
    data["Quality Flag"] = pd.to_numeric(raw[27], errors="coerce")

    data = data.dropna(subset=["Date"]).sort_values("Date")
    data = data.drop_duplicates(subset=["Date"], keep="last")
    data = data.set_index("Date").asfreq("D")
    # Treat the documented special Ap value 400 as missing, not a real observation.
    data.loc[data["Ap"] >= 400, "Ap"] = np.nan
    return data

def score_forecast(actual, predicted):
    aligned = pd.concat(
        [pd.Series(actual, name="actual"), pd.Series(predicted, name="predicted")],
        axis=1,
    ).dropna()
    if aligned.empty:
        return np.nan, np.nan
    mae = mean_absolute_error(aligned["actual"], aligned["predicted"])
    rmse = np.sqrt(mean_squared_error(aligned["actual"], aligned["predicted"]))
    return mae, rmse

def make_forecast(data, train_end, test_start, test_end, order, log_transform=False):
    target = data.loc[:train_end, "Ap"].dropna()
    actual = data.loc[test_start:test_end, "Ap"].dropna()
    if len(target) < 100 or len(actual) == 0:
        raise ValueError("Not enough valid observations in the selected training/test periods.")
    if log_transform:
        fitted = ARIMA(np.log1p(target), order=order).fit()
        pred = np.expm1(fitted.forecast(steps=len(actual)))
    else:
        fitted = ARIMA(target, order=order).fit()
        pred = fitted.forecast(steps=len(actual))
    pred.index = actual.index
    return actual, pred, fitted

with st.sidebar:
    st.header("Dashboard controls")
    uploaded_file = st.file_uploader(
        "Upload GFZ combined-index file",
        type=["txt", "dat", "csv"],
        help="Upload the whitespace-delimited GFZ file saved as kpdata.txt.",
    )
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        source_label = uploaded_file.name
    elif DEFAULT_FILE.exists():
        file_bytes = DEFAULT_FILE.read_bytes()
        source_label = DEFAULT_FILE.name
        st.success("Loaded kpdata.txt from the app folder.")
    else:
        file_bytes = None
        source_label = ""

    st.divider()
    st.caption("Forecast settings")
    model_order = st.selectbox(
        "ARIMA order (p, d, q)",
        options=["(1, 0, 1)", "(1, 0, 0)", "(0, 0, 1)", "(2, 0, 1)"],
        index=0,
        help="Start with (1, 0, 1), the model tested in the case study.",
    )
    train_end_year = st.selectbox("Training data through year", [2022, 2023, 2024], index=1)
    forecast_horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=90, value=30, step=7)

if file_bytes is None:
    st.info(
        "Upload your GFZ `kpdata.txt` file using the sidebar, or place it beside `app.py` "
        "and run the app from that folder."
    )
    st.markdown("""
    ### What this dashboard includes
    - Daily geomagnetic activity and solar-index exploration
    - Yearly averages and high-activity days
    - ADF stationarity test
    - Naive baseline vs ARIMA forecast evaluation
    - Downloadable cleaned data and forecast outputs
    """)
    st.stop()

try:
    data = load_gfz_file(file_bytes)
except Exception as exc:
    st.error(f"Could not read this file: {exc}")
    st.stop()

if data["Ap"].dropna().empty:
    st.error("No valid Ap observations were found. Check that this is the GFZ combined-index file.")
    st.stop()

# ---------- Filters ----------
min_date = data.index.min().date()
max_date = data.index.max().date()
with st.sidebar:
    selected_dates = st.date_input(
        "Explore date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_date, end_date = pd.Timestamp(selected_dates[0]), pd.Timestamp(selected_dates[1])
else:
    start_date, end_date = pd.Timestamp(min_date), pd.Timestamp(max_date)

view = data.loc[start_date:end_date].copy()
ap_valid = view["Ap"].dropna()
all_ap = data["Ap"].dropna()

st.caption(f"Data window: {min_date:%d %b %Y} – {max_date:%d %b %Y} · Source: {source_label} · Use the sidebar to filter dates and forecast settings.")

# ---------- KPI row ----------
daily_count = int(ap_valid.shape[0])
mean_ap = float(ap_valid.mean()) if daily_count else np.nan
max_ap = float(ap_valid.max()) if daily_count else np.nan
max_date = ap_valid.idxmax().strftime("%d %b %Y") if daily_count else "—"
active_days = int((ap_valid >= 30).sum()) if daily_count else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Valid daily records", f"{daily_count:,}")
k2.metric("Mean Ap", f"{mean_ap:.2f}" if np.isfinite(mean_ap) else "—")
k3.metric("Peak Ap", f"{max_ap:.0f}" if np.isfinite(max_ap) else "—")
k4.metric("Peak date", max_date)
k5.metric("Days with Ap ≥ 30", f"{active_days:,}")

tab_overview, tab_patterns, tab_forecast, tab_data = st.tabs(
    ["🌍 Overview", "📈 Patterns & Statistics", "🔮 Forecast Lab", "🗂️ Data Explorer"]
)

with tab_overview:
    st.subheader("Geomagnetic activity over time")
    if not ap_valid.empty:
        chart_data = view.reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_data["Date"], y=chart_data["Ap"],
            mode="lines", name="Daily Ap",
            line=dict(width=1, color="#6fb7ff"),
            connectgaps=False,
        ))
        rolling = view["Ap"].rolling(30, min_periods=10).mean()
        fig.add_trace(go.Scatter(
            x=rolling.index, y=rolling.values,
            mode="lines", name="30-day rolling mean",
            line=dict(width=2.5, color="#55e0c0"),
        ))
        high_days = view[view["Ap"] >= 30]["Ap"].dropna()
        if not high_days.empty:
            fig.add_trace(go.Scatter(
                x=high_days.index, y=high_days.values,
                mode="markers", name="Ap ≥ 30 (exploratory flag)",
                marker=dict(size=7, color="#ffb45e", line=dict(width=1, color="#171d2a")),
            ))
        fig.update_layout(
            title="Daily geomagnetic activity",
            xaxis_title="Date", yaxis_title="Daily Ap index",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(8,15,27,0.55)", hovermode="x unified",
            height=450, margin=dict(l=10, r=10, t=55, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(130,160,190,.12)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(130,160,190,.12)")
        st.plotly_chart(fig, use_container_width=True)

        yearly = (
            data["Ap"].loc[start_date:end_date]
            .resample("YE").mean()
            .dropna()
            .rename("Mean Ap")
            .reset_index()
        )
        yearly["Year"] = yearly["Date"].dt.year
        fig_year = px.bar(
            yearly, x="Year", y="Mean Ap",
            title="Yearly mean geomagnetic activity",
            labels={"Mean Ap": "Mean Ap"},
        )
        fig_year.update_layout(height=330, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,15,27,0.55)")
        st.plotly_chart(fig_year, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Highest-activity days in selected period**")
        top_days = view[["Ap", "Sunspot Number", "F10.7 Flux"]].dropna(subset=["Ap"]).nlargest(10, "Ap")
        st.dataframe(top_days.reset_index(), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**How to interpret Ap**")
        st.write(
            "Ap is a daily geomagnetic activity index. Higher values indicate stronger "
            "geomagnetic disturbance. It is a useful space-weather indicator, but it does "
            "not by itself predict local aurora visibility or guarantee a geomagnetic storm."
        )
        st.caption("The threshold Ap ≥ 30 is used here as an exploratory dashboard flag, not as an official storm classification.")

with tab_patterns:
    st.subheader("Patterns, distributions and stationarity")
    c1, c2 = st.columns(2)
    with c1:
        hist = px.histogram(
            view.dropna(subset=["Ap"]), x="Ap", nbins=45,
            title="Distribution of daily Ap values",
        )
        hist.update_layout(height=350, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,15,27,0.55)")
        st.plotly_chart(hist, use_container_width=True)
    with c2:
        monthly = view["Ap"].groupby(view.index.month).mean().reindex(range(1, 13))
        month_df = pd.DataFrame({"Month": range(1, 13), "Mean Ap": monthly.values})
        month_df["Month"] = pd.to_datetime(month_df["Month"], format="%m").dt.strftime("%b")
        mfig = px.bar(month_df, x="Month", y="Mean Ap", title="Average Ap by calendar month")
        mfig.update_layout(height=350, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,15,27,0.55)")
        st.plotly_chart(mfig, use_container_width=True)

    st.markdown("#### Augmented Dickey–Fuller test")
    adf_series = data.loc[start_date:end_date, "Ap"].dropna()
    if len(adf_series) >= 20:
        try:
            adf = adfuller(adf_series, autolag="AIC")
            a1, a2, a3 = st.columns(3)
            a1.metric("ADF statistic", f"{adf[0]:.4f}")
            a2.metric("p-value", f"{adf[1]:.3g}")
            a3.metric("Lags used", str(adf[2]))
            if adf[1] < 0.05:
                st.success("Reject the unit-root null hypothesis at the 5% level.")
            else:
                st.warning("Do not reject the unit-root null hypothesis at the 5% level.")
            st.caption("The ADF result does not by itself prove constant variance or absence of solar-cycle effects.")
        except Exception as exc:
            st.warning(f"ADF test could not be calculated: {exc}")
    else:
        st.info("Select a longer date range to run the ADF test.")

    st.markdown("#### Solar indicators")
    solar_cols = ["Sunspot Number", "F10.7 Flux"]
    solar_view = view[solar_cols].dropna(how="all")
    if not solar_view.empty:
        fig_solar = go.Figure()
        for col in solar_cols:
            fig_solar.add_trace(go.Scatter(
                x=solar_view.index, y=solar_view[col],
                mode="lines", name=col,
            ))
        fig_solar.update_layout(
            title="Solar indicators over time",
            xaxis_title="Date", yaxis_title="Index / flux value",
            height=400, hovermode="x unified", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,15,27,0.55)",
        )
        st.plotly_chart(fig_solar, use_container_width=True)

with tab_forecast:
    st.subheader("Forecast Lab")
    st.write("Train on earlier observations and evaluate predictions on a later holdout period.")
    train_end = pd.Timestamp(f"{train_end_year}-12-31")
    test_start = train_end + pd.Timedelta(days=1)
    test_end = min(data.index.max(), test_start + pd.Timedelta(days=729))
    order = tuple(int(x.strip()) for x in model_order.strip("()").split(","))

    train_series = data.loc[:train_end, "Ap"].dropna()
    test_series = data.loc[test_start:test_end, "Ap"].dropna()

    if len(train_series) < 100 or len(test_series) < 30:
        st.warning("Not enough observations for the selected training end year and test period.")
    else:
        with st.spinner("Fitting ARIMA and generating forecasts..."):
            try:
                fitted = ARIMA(train_series, order=order).fit()
                arima_pred = fitted.forecast(steps=len(test_series))
                arima_pred.index = test_series.index

                # One-step naive prediction uses previous day's actual observation.
                naive_pred = test_series.shift(1)
                naive_pred.iloc[0] = train_series.iloc[-1]

                arima_mae, arima_rmse = score_forecast(test_series, arima_pred)
                naive_mae, naive_rmse = score_forecast(test_series.iloc[1:], naive_pred.iloc[1:])

                m1, m2 = st.columns(2)
                m1.metric("ARIMA MAE", f"{arima_mae:.3f}")
                m2.metric("ARIMA RMSE", f"{arima_rmse:.3f}")

                metrics_df = pd.DataFrame({
                    "Model": ["Naive baseline", f"ARIMA{order}"],
                    "MAE": [naive_mae, arima_mae],
                    "RMSE": [naive_rmse, arima_rmse],
                })
                st.dataframe(metrics_df.round(3), use_container_width=True, hide_index=True)

                comp_fig = go.Figure()
                comp_fig.add_trace(go.Scatter(x=test_series.index, y=test_series, name="Actual Ap", mode="lines"))
                comp_fig.add_trace(go.Scatter(x=arima_pred.index, y=arima_pred, name="ARIMA forecast", mode="lines"))
                comp_fig.update_layout(title="Actual vs ARIMA forecast (holdout period)", xaxis_title="Date", yaxis_title="Ap", height=430, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,15,27,0.55)")
                st.plotly_chart(comp_fig, use_container_width=True)

                # Refit on all available valid observations for future forecast.
                horizon = int(forecast_horizon)
                full_series = data["Ap"].dropna()
                future_fit = ARIMA(full_series, order=order).fit()
                future_values = future_fit.forecast(steps=horizon)
                future_dates = pd.date_range(full_series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
                future_df = pd.DataFrame({"Date": future_dates, "Forecast Ap": np.asarray(future_values)})
                future_fig = px.line(future_df, x="Date", y="Forecast Ap", title=f"Experimental {horizon}-day Ap forecast")
                future_fig.update_layout(height=350, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,15,27,0.55)")
                st.plotly_chart(future_fig, use_container_width=True)
                st.warning(
                    "This is an experimental statistical forecast, not an operational space-weather warning. "
                    "Historical-Ap-only ARIMA forecasts are not reliable predictors of sudden storms or local aurora visibility."
                )

                output = pd.DataFrame({
                    "Date": test_series.index,
                    "Actual_Ap": test_series.values,
                    "Naive_Forecast": naive_pred.values,
                    "ARIMA_Forecast": arima_pred.values,
                })
                st.download_button(
                    "Download holdout forecasts (CSV)",
                    data=output.to_csv(index=False).encode("utf-8"),
                    file_name="solarscope_holdout_forecasts.csv",
                    mime="text/csv",
                )
                st.download_button(
                    "Download model metrics (CSV)",
                    data=metrics_df.to_csv(index=False).encode("utf-8"),
                    file_name="solarscope_model_metrics.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Forecasting failed: {exc}")
                st.info("Try ARIMA order (1, 0, 1), or choose an earlier training end year.")

with tab_data:
    st.subheader("Data Explorer")
    st.caption(f"Source file: {source_label} · GFZ combined geomagnetic and solar indices format")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows in date range", f"{len(view):,}")
    c2.metric("Missing Ap values", f"{int(view['Ap'].isna().sum()):,}")
    c3.metric("Duplicate dates after cleaning", f"{int(view.index.duplicated().sum()):,}")
    st.dataframe(view.reset_index(), use_container_width=True, height=420)
    st.download_button(
        "Download cleaned selected data (CSV)",
        data=view.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="solarscope_cleaned_data.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "SolarScope is an educational time-series analysis dashboard. GFZ data should be cited in the project report. "
    "Forecasts are statistical demonstrations and should not be used for operational safety decisions."
)
