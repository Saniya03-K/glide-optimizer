import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import requests

# importing from our own modules
from physics_engine import calc_performance, get_aircraft_data, get_polar_data, get_max_glide
from optimizer import golden_section

# --- openweathermap api ---
def fetch_wind(city="London"):
    api_key = "59ecb0a05890748b5795a65c07ad69c9"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        if data.get("cod") != 200:
            return None, f"Error: {data.get('message', 'City not found')}"
        ws = data['wind']['speed'] * 1.94384  # convert to knots
        return ws, None
    except Exception as e:
        return None, str(e)

# --- page setup ---
st.set_page_config(
    page_title="Glide Performance Optimizer",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Dynamic Glide Performance Optimizer")
st.markdown("*Multi-Aircraft, Real-Time Physics, and Machine Learning Validation*")

# --- sidebar ---
st.sidebar.header("Flight Conditions")

ac_options = {
    1: "Cessna 152",
    2: "Cessna 172S",
    3: "Piper Cherokee PA-28-140"
}
ac_id = st.sidebar.selectbox(
    "Select Aircraft",
    options=list(ac_options.keys()),
    format_func=lambda x: ac_options[x]
)

ac_data = get_aircraft_data(ac_id)
if ac_data is not None:
    empty_w = ac_data['empty_weight_lbs']
    max_w = ac_data['max_gross_weight_lbs']
    poh_speed = ac_data['best_glide_kias']
else:
    empty_w, max_w, poh_speed = 1500, 2500, 70

alt = st.sidebar.slider("Altitude (ft)", 1000, 12000, 6000, 100)
wt = st.sidebar.slider("Aircraft Weight (lbs)", int(empty_w), int(max_w), int((empty_w + max_w) / 2), 10)

# session state for wind
if "wind_kts" not in st.session_state:
    st.session_state.wind_kts = 10
wind_kts = st.sidebar.slider(
    "Headwind (kts) [+], Tailwind (kts) [-]",
    -20.0, 50.0,
    value=float(st.session_state.wind_kts),
    step=1.0
)
st.session_state.wind_kts = wind_kts

# live weather
st.sidebar.divider()
st.sidebar.subheader("🌦️ Live Weather")
city = st.sidebar.text_input("Enter City", "London")
if st.sidebar.button("Fetch Live Wind"):
    wind_kts, error = fetch_wind(city)
    if error:
        st.sidebar.error(f"Could not fetch wind: {error}")
    else:
        st.sidebar.success(f"Live Wind: **{wind_kts:.1f} kts**")
        st.session_state.wind_kts = wind_kts
        st.rerun()

st.sidebar.divider()

# angle of attack
st.sidebar.subheader("Angle of Attack")
use_opt = st.sidebar.button("🎯 Find Optimal Angle", type="primary")

if use_opt:
    opt_res = golden_section(ac_id=ac_id, alt=alt, weight=wt, wind=float(wind_kts), a=0.0, b=15.0, tol=0.001)
    alpha = opt_res['best_alpha_deg']
    st.sidebar.success(f"Optimal AoA: {alpha:.2f}°")
else:
    alpha = st.sidebar.slider("Angle of Attack (degrees)", 0.0, 15.0, 5.0, 0.1)

# --- main content ---
try:
    res = calc_performance(
        aircraft_id=ac_id,
        altitude_ft=alt,
        alpha_deg=alpha,
        weight_lbs=wt,
        wind_kts=float(wind_kts)
    )

    # top metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Glide Ratio (Current)", f"{res['glide_ratio_air']:.1f}")
    with col2:
        st.metric("Airspeed", f"{res['v_airspeed_kts']:.1f} KIAS")
    with col3:
        st.metric("Time to Ground", f"{res['time_minutes']:.1f} min")
    with col4:
        st.metric("Distance (Ground)", f"{res['dist_ground_nm']:.2f} NM", delta=f"{res['penalty_pct']:.1f}% Penalty")

    # --- POH vs optimizer duel ---
    st.divider()
    max_glide, poh_alpha = get_max_glide(ac_id)
    poh_res = calc_performance(aircraft_id=ac_id, altitude_ft=alt, alpha_deg=poh_alpha, weight_lbs=wt, wind_kts=float(wind_kts))
    opt_res = golden_section(ac_id=ac_id, alt=alt, weight=wt, wind=float(wind_kts), a=0.0, b=15.0, tol=0.001)
    dist_saved = opt_res['best_distance_nm'] - poh_res['dist_ground_nm']

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("🔵 POH Static Angle", f"{poh_res['dist_ground_nm']:.2f} NM", delta=f"{poh_alpha:.1f}°")
    with col_b:
        st.metric("🟢 Optimizer Dynamic Angle", f"{opt_res['best_distance_nm']:.2f} NM", delta=f"{opt_res['best_alpha_deg']:.1f}°")
    with col_c:
        if dist_saved > 0:
            st.metric("🟡 Distance Saved", f"{dist_saved:.2f} NM", delta=f"✅ +{dist_saved:.1f} NM better")
        else:
            st.metric("🟡 Distance Saved", f"{dist_saved:.2f} NM", delta="ℹ️ Same as POH", delta_color="off")

    st.divider()

    # --- sensitivity heatmap ---
    st.subheader("🗺️ Sensitivity Heatmap: Optimal AoA vs Weight & Wind")
    weight_range = np.linspace(empty_w + 50, max_w - 50, 8)
    wind_range = np.linspace(-20, 50, 8)
    heatmap = np.zeros((len(weight_range), len(wind_range)))

    for i, w in enumerate(weight_range):
        for j, wv in enumerate(wind_range):
            try:
                o = golden_section(ac_id=ac_id, alt=alt, weight=w, wind=wv, a=0.0, b=15.0, tol=0.01)
                heatmap[i, j] = o['best_alpha_deg']
            except:
                heatmap[i, j] = np.nan

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    im = ax3.imshow(heatmap, extent=[wind_range.min(), wind_range.max(), weight_range.min(), weight_range.max()],
                    origin='lower', cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=15)
    ax3.set_xlabel('Wind Speed (kts)')
    ax3.set_ylabel('Weight (lbs)')
    ax3.set_title('Optimal Angle of Attack (°)')
    plt.colorbar(im, ax=ax3).set_label('Optimal AoA (°)')
    st.pyplot(fig3)
    st.caption("*Dark green = lower AoA (faster). Dark red = higher AoA (slower).*")

    st.divider()

    # --- optimization vs current ---
    opt_res = golden_section(ac_id=ac_id, alt=alt, weight=wt, wind=float(wind_kts), a=0.0, b=15.0, tol=0.001)
    st.subheader("🔄 Optimization vs Current Performance")
    col5, col6 = st.columns(2)
    with col5:
        st.info(f"**Current AoA**: {alpha:.2f}°\n\nDistance: {res['dist_ground_nm']:.2f} NM")
    with col6:
        st.success(f"**Optimal AoA**: {opt_res['best_alpha_deg']:.2f}°\n\nMax Distance: {opt_res['best_distance_nm']:.2f} NM")

    # --- plot 1: cl/cd curve ---
    st.subheader("📈 Aerodynamic Performance Curve")
    polar_df = get_polar_data(ac_id)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(polar_df['alpha_deg'], polar_df['glide_ratio'], 'b-', label='Cl/Cd (Airfoil)', linewidth=2)
    ax.scatter(alpha, res['glide_ratio_air'], color='red', s=100, zorder=5, label=f'Current (α={alpha:.1f}°)')
    mg, ba = get_max_glide(ac_id)
    ax.scatter(ba, mg, color='green', s=100, zorder=5, label=f'Optimal (α={ba:.1f}°)')
    ax.set_xlabel('Angle of Attack (degrees)')
    ax.set_ylabel('Glide Ratio (Cl/Cd)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    st.pyplot(fig, width=600)

    # --- plot 2: distance vs alpha ---
    st.subheader("📉 Ground Distance vs Angle of Attack")
    alphas = np.linspace(0, 15, 100)
    dists = []
    for a in alphas:
        try:
            tmp = calc_performance(aircraft_id=ac_id, altitude_ft=alt, alpha_deg=a, weight_lbs=wt, wind_kts=float(wind_kts))
            dists.append(tmp['dist_ground_nm'])
        except:
            dists.append(0)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(alphas, dists, 'b-', linewidth=2)
    ax2.scatter(alpha, res['dist_ground_nm'], color='red', s=100, zorder=5, label='Current')
    ax2.scatter(opt_res['best_alpha_deg'], opt_res['best_distance_nm'], color='green', s=100, zorder=5, label='Optimal')
    ax2.set_xlabel('Angle of Attack (degrees)')
    ax2.set_ylabel('Ground Distance (NM)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    st.pyplot(fig2, width=600)

    st.divider()

    # --- ML validation section ---
    st.subheader("🧠 Machine Learning Validation")

    try:
        ml_path = 'data/ml_results.csv'
        if os.path.exists(ml_path):
            ml_df = pd.read_csv(ml_path)
            st.dataframe(ml_df, width='stretch')

            # feature importance comparison
            fi_path = 'data/feature_importance.csv'
            if os.path.exists(fi_path):
                fi_df = pd.read_csv(fi_path)
                fig4, ax4 = plt.subplots(figsize=(10, 5))
                x = np.arange(len(fi_df))
                width = 0.35
                ax4.barh(x - width/2, fi_df['Direct_Importance'], width, label='Direct Model', color='steelblue')
                ax4.barh(x + width/2, fi_df['Residual_Importance'], width, label='Residual Model', color='coral')
                ax4.set_yticks(x)
                ax4.set_yticklabels(fi_df['Feature'])
                ax4.set_xlabel('Importance')
                ax4.set_title('Feature Importance: Direct vs Residual')
                ax4.legend()
                ax4.invert_yaxis()
                st.pyplot(fig4)
                st.caption("*Direct model relies on altitude and wind. Residual model shifts focus to alpha and aircraft features.*")
        else:
            st.info("Run train_models.py to generate ML results.")
    except Exception as e:
        st.error(f"ML results error: {e}")

    st.divider()

    # --- SHAP section ---
    st.subheader("🔍 Explainable AI (SHAP)")

    # summary plots side by side
    col_shap1, col_shap2 = st.columns(2)
    with col_shap1:
        st.markdown("**Direct Model SHAP**")
        st.caption("What the model learned from data alone")
        if os.path.exists('data/shap_summary_direct.png'):
            st.image('data/shap_summary_direct.png', width=500)
    with col_shap2:
        st.markdown("**Residual Model SHAP**")
        st.caption("What the model learned to correct in the formula")
        if os.path.exists('data/shap_summary_residual.png'):
            st.image('data/shap_summary_residual.png', width=500)

    # wind dependence plots
    st.markdown("---")
    st.markdown("**Wind Dependence — Direct vs Residual**")
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        if os.path.exists('data/shap_wind_direct.png'):
            st.image('data/shap_wind_direct.png', width=500)
    with col_w2:
        if os.path.exists('data/shap_wind_residual.png'):
            st.image('data/shap_wind_residual.png', width=500)

    st.caption("*Direct model: wind decreases distance. Residual model: formula already handles wind, so ML correction is smaller.*")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.write("Please adjust the inputs or check the database connection.")