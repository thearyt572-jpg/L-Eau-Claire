"""
L'Eau Claire · 4-page Streamlit App
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import joblib, os, warnings, random, datetime, base64
from PIL import Image
import io
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="L'Eau Claire · Water Quality Monitor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
STATE_COORDS = {
    "ANDHRA PRADESH"   : (15.9129,  79.7400),
    "ASSAM"            : (26.2006,  92.9376),
    "BIHAR"            : (25.0961,  85.3131),
    "CHANDIGARH"       : (30.7333,  76.7794),
    "CHHATTISGARH"     : (21.2787,  81.8661),
    "DELHI"            : (28.7041,  77.1025),
    "GOA"              : (15.2993,  74.1240),
    "GUJARAT"          : (22.2587,  71.1924),
    "HARYANA"          : (29.0588,  76.0856),
    "HIMACHAL PRADESH" : (31.1048,  77.1734),
    "KARNATAKA"        : (15.3173,  75.7139),
    "KERALA"           : (10.8505,  76.2711),
    "LAKSHADWEEP"      : (10.5667,  72.6417),
    "MADHYA PRADESH"   : (22.9734,  78.6569),
    "MANIPUR"          : (24.6637,  93.9063),
    "MEGHALAYA"        : (25.4670,  91.3662),
    "ODISHA"           : (20.9517,  85.0985),
    "PUDUCHERRY"       : (11.9416,  79.8083),
    "PUNJAB"           : (31.1471,  75.3412),
    "RAJASTHAN"        : (27.0238,  74.2179),
    "SIKKIM"           : (27.5330,  88.5122),
    "TAMIL NADU"       : (11.1271,  78.6569),
    "TELANGANA"        : (18.1124,  79.0193),
    "TRIPURA"          : (23.9408,  91.9882),
    "UTTAR PRADESH"    : (26.8467,  80.9462),
    "UTTARAKHAND"      : (30.0668,  79.0193),
    "WEST BENGAL"      : (22.9868,  87.8550),
}

ARTIFACTS_DIR = "model/artifacts"
REPORTS_CSV   = "data/reports.csv"
UPLOADS_DIR   = "uploads"

os.makedirs("data",       exist_ok=True)
os.makedirs(UPLOADS_DIR,  exist_ok=True)


# SAFETY HELPERS
def get_safety_map(fc):
    if fc <= 500:   return "✅ Safe to drink",           "green",  "Safe"
    elif fc <= 5000:return "⚠️ Not safe to drink",       "orange", "Moderate"
    else:           return "🚨 Dangerous — avoid contact","red",    "Unsafe"

def get_safety_predict(fc):
    if fc <= 50:    return "✅ Safe (Class A)",   "#2e9e6e", True
    elif fc <= 500: return "⚠️ Moderate risk",   "#d48f00", False
    else:           return "🚨 High risk unsafe","#c94040", False


# DATA LOADERS
@st.cache_data
def load_data():
    for path in ["dataset 2017-2022.csv",
                 "dataset 2017-2022 - Copy.csv",
                 "/mnt/user-data/uploads/dataset_2017-2022.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            for col in ["State Name","Type Water Body"]:
                df[col] = (df[col].astype(str)
                           .str.replace(r"[\n\r]"," ",regex=True)
                           .str.replace(r"\s+"," ",regex=True)
                           .str.strip().str.upper())
            raw_pairs = [
                ("Temperature",    "Min Temperature",           "Max Temperature"),
                ("DO",             "Min Dissolved Oxygen",      "Max Dissolved Oxygen"),
                ("pH",             "Min pH",                    "Max pH"),
                ("Conductivity",   "Min Conductivity",          "Max Conductivity"),
                ("BOD",            "Min BOD",                   "Max BOD"),
                ("Nitrate_Nitrite","Min Nitrate N + Nitrite N", "Max Nitrate N + Nitrite N"),
                ("Fecal_Coliform", "Min Fecal Coliform",        "Max Fecal Coliform"),
                ("Total_Coliform", "Min Total Coliform",        "Max Total Coliform"),
            ]
            for avg_col,mn,mx in raw_pairs:
                for c in [mn,mx]:
                    df[c] = pd.to_numeric(
                        df[c].astype(str).str.replace(",","",regex=False),errors="coerce")
                df[avg_col] = (df[mn]+df[mx])/2
            df = df.dropna(subset=["Fecal_Coliform","State Name","Type Water Body"])
            df = df[df["Fecal_Coliform"]>=0]
            return df
    return None

@st.cache_resource
def load_artifacts():
    base = ARTIFACTS_DIR
    try:
        model        = joblib.load(f"{base}/finalbest_model_tuned.pkl")
        feature_cols = joblib.load(f"{base}/finalfeature_cols.pkl")
        qt           = joblib.load(f"{base}/qt.pkl")
        sf_path      = f"{base}/state_freq_mapping.pkl"
        state_freq   = joblib.load(sf_path) if os.path.exists(sf_path) else {}
        return model, feature_cols, qt, state_freq, None
    except Exception as e:
        return None, None, None, {}, str(e)

def load_reports():
    if os.path.exists(REPORTS_CSV):
        return pd.read_csv(REPORTS_CSV)
    return pd.DataFrame(columns=[
        "timestamp","reporter","state","water_body","severity",
        "description","lat","lon","photo_path"])

def save_report(record: dict):
    df = load_reports()
    new_row = pd.DataFrame([record])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(REPORTS_CSV, index=False)


# GLOBAL STYLES
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/*  Global baby blue background */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #ddeeff !important;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stAppViewContainer"] > .main {
    background-color: #ddeeff !important;
}
[data-testid="stSidebar"] {
    background-color: #c8e0f4 !important;
}
/* Input widgets soft background */
[data-testid="stForm"], .stNumberInput input, .stSelectbox select,
div[data-baseweb="select"] { background-color: #eef6ff !important; }

.page-title{font-family:'Playfair Display',serif;font-size:2.1rem;font-weight:700;color:#0d3b4f;margin-bottom:0;}
.page-sub{font-size:0.88rem;color:#5a7a99;margin-top:2px;margin-bottom:1.2rem;}
.fact-card{background:linear-gradient(135deg,#cce8ff,#b8d8f5);border-radius:16px;
  padding:24px 28px;border-left:5px solid #3fa8c8;margin-bottom:16px;}
.fact-number{font-family:'Playfair Display',serif;font-size:2.8rem;font-weight:700;color:#0d3b4f;line-height:1;}
.tip-card{background:#eef6ff;border-radius:12px;padding:18px 20px;
  border:1.5px solid #b8d8f5;box-shadow:0 2px 8px rgba(13,59,79,0.07);margin-bottom:10px;}
.tip-icon{font-size:1.6rem;margin-bottom:6px;}
.tip-title{font-weight:600;color:#0d3b4f;font-size:1rem;margin-bottom:4px;}
.tip-text{font-size:0.87rem;color:#3d5a6b;line-height:1.5;}
.severity-high{color:#c94040;font-weight:600;}
.severity-med{color:#d48f00;font-weight:600;}
.severity-low{color:#2e9e6e;font-weight:600;}

/* Result boxes */
.result-box{border-radius:14px;padding:20px 24px;margin-bottom:12px;border:1.5px solid rgba(0,0,0,0.07);}
.result-safe{background:#c8f0d8;border-color:rgba(46,158,110,0.3);}
.result-mod{background:#fff3cd;border-color:rgba(212,143,0,0.3);}
.result-unsafe{background:#fce4e4;border-color:rgba(201,64,64,0.3);}
.result-number{font-family:'Playfair Display',serif;font-size:3rem;font-weight:700;line-height:1;color:#0d3b4f;}
.result-unit{font-size:0.85rem;color:#7a9aaa;margin-top:2px;}
.result-badge{font-size:1.05rem;font-weight:600;margin-top:10px;}
.conf-track{background:#b8d8f5;border-radius:8px;height:10px;overflow:hidden;margin-top:6px;}
.conf-fill{height:100%;border-radius:8px;background:linear-gradient(90deg,#7ecab8,#3fa8c8);}

/* Page-level background overrides for predict result */
.predict-bg-safe   { background-color: #d6f5e3 !important; }
.predict-bg-mod    { background-color: #fff8e1 !important; }
.predict-bg-unsafe { background-color: #fde8e8 !important; }
</style>
""", unsafe_allow_html=True)

# SIDEBAR NAV
with st.sidebar:
    st.markdown("### L'Eau Claire")
    st.caption("Water Quality Monitor · 2017–2022")
    st.divider()
    page = st.radio("Navigate", [
        "🗺️  Water Quality Map",
        "🔮  Predict Contamination",
        "💡  Learn & Tips",
        "🚨  Report Pollution",
    ], label_visibility="collapsed")
    st.divider()

df       = load_data()
model, feature_cols, qt, state_freq, load_err = load_artifacts()
state_list = sorted(state_freq.keys()) if state_freq else sorted(STATE_COORDS.keys())

# Session state for predict page background color
if "predict_bg" not in st.session_state:
    st.session_state.predict_bg = "default"  # default | safe | mod | unsafe

# PAGE 1: WATER QUALITY MAP
if page == "🗺️  Water Quality Map":
    st.session_state.predict_bg = "default"
    st.markdown('<p class="page-title">🗺️ India Water Quality Map</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Drinking water safety across Indian states · Fecal Coliform levels 2017–2022</p>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    c1.success("🟢  Safe  ≤ 500 MPN/100ml")
    c2.warning("🟡  Moderate  501–5,000 MPN/100ml")
    c3.error("🔴  Dangerous  > 5,000 MPN/100ml")
    st.divider()

    with st.sidebar:
        st.header("🔍 Filters")
        if df is not None:
            wb_options    = ["ALL"] + sorted(df["Type Water Body"].unique().tolist())
            selected_wb   = st.selectbox("Water Body Type", wb_options)
            year_options  = ["ALL"] + sorted(df["Year"].dropna().unique().astype(int).tolist())
            selected_year = st.selectbox("Year", year_options)
            st.divider()
            st.header("📊 Safety Summary")
            state_agg  = df.groupby("State Name")["Fecal_Coliform"].median().reset_index()
            safe_n     = (state_agg["Fecal_Coliform"]<=500).sum()
            moderate_n = ((state_agg["Fecal_Coliform"]>500)&(state_agg["Fecal_Coliform"]<=5000)).sum()
            unsafe_n   = (state_agg["Fecal_Coliform"]>5000).sum()
            st.write(f"🟢 Safe states: **{safe_n}**")
            st.write(f"🟡 Moderate states: **{moderate_n}**")
            st.write(f"🔴 Unsafe states: **{unsafe_n}**")
        else:
            selected_wb,selected_year = "ALL","ALL"
            st.warning("Place `dataset 2017-2022.csv` in the app folder.")

    if df is not None:
        plot_df = df.copy()
        if selected_wb   != "ALL": plot_df = plot_df[plot_df["Type Water Body"]==selected_wb]
        if selected_year != "ALL": plot_df = plot_df[plot_df["Year"]==int(selected_year)]
        agg = (plot_df.groupby(["State Name","Type Water Body"])
               .agg(FC=("Fecal_Coliform","median"),count=("Fecal_Coliform","count"))
               .reset_index())
    else:
        agg = pd.DataFrame()

    m = folium.Map(location=[22.5,82.0],zoom_start=5,tiles="CartoDB positron",min_zoom=4,max_zoom=12)
    m.fit_bounds([[6.5,68.0],[35.5,97.5]])

    if not agg.empty:
        import hashlib
        for _,row in agg.iterrows():
            state,wb,fc = row["State Name"],row["Type Water Body"],row["FC"]
            if state not in STATE_COORDS: continue
            lat,lon = STATE_COORDS[state]
            h = int(hashlib.md5(wb.encode()).hexdigest(),16)
            lat += ((h%100)-50)/800; lon += ((h%137)-68)/800
            label,color,status = get_safety_map(fc)
            bg = {"green":"#d4edda","orange":"#fff3cd","red":"#f8d7da"}[color]
            popup_html = f"""
            <div style="font-family:Arial,sans-serif;min-width:200px;padding:4px">
              <b style="font-size:15px">📍 {state.title()}</b><hr style="margin:4px 0">
              <table style="width:100%;font-size:13px">
                <tr><td><b>Water Body</b></td><td>{wb.title()}</td></tr>
                <tr><td><b>Fecal Coliform</b></td><td>{fc:,.0f} MPN/100ml</td></tr>
                <tr><td><b>Records</b></td><td>{row['count']:,}</td></tr>
              </table>
              <div style="margin-top:8px;padding:6px;border-radius:6px;
                   background:{bg};font-weight:bold;font-size:13px;text-align:center">{label}</div>
            </div>"""
            folium.CircleMarker(location=[lat,lon],radius=12,color="white",weight=1.5,
                fill=True,fill_color=color,fill_opacity=0.85,
                popup=folium.Popup(popup_html,max_width=260),
                tooltip=folium.Tooltip(f"<b>{state.title()}</b> — {wb.title()}<br>{label}",sticky=True),
            ).add_to(m)

    # Overlay pollution reports on map
    reports_df = load_reports()
    if not reports_df.empty:
        for _,r in reports_df.iterrows():
            try:
                folium.Marker(
                    location=[float(r["lat"]),float(r["lon"])],
                    icon=folium.Icon(color="red",icon="warning-sign",prefix="glyphicon"),
                    tooltip=f"⚠️ Report: {r['water_body']} — {r['severity']}",
                    popup=folium.Popup(
                        f"<b>🚨 Pollution Report</b><br>"
                        f"<b>State:</b> {r['state']}<br>"
                        f"<b>Water Body:</b> {r['water_body']}<br>"
                        f"<b>Severity:</b> {r['severity']}<br>"
                        f"<b>Details:</b> {r['description']}<br>"
                        f"<b>Reported:</b> {r['timestamp']}",
                        max_width=250)
                ).add_to(m)
            except: pass

    st_folium(m,width="100%",height=620,returned_objects=[])

    if not agg.empty:
        st.divider()
        st.subheader("📋 Full Data Table")
        display = agg.copy()
        display["Safety"]   = display["FC"].apply(lambda x: get_safety_map(x)[2])
        display["State"]    = display["State Name"].str.title()
        display["Water Body"] = display["Type Water Body"].str.title()
        display["Fecal Coliform (MPN/100ml)"] = display["FC"].round(0).astype(int)
        display = display[["State","Water Body","Fecal Coliform (MPN/100ml)","count","Safety"]]
        display.columns = ["State","Water Body","Fecal Coliform (MPN/100ml)","Records","Safety"]
        display = display.sort_values("Fecal Coliform (MPN/100ml)",ascending=False).reset_index(drop=True)
        st.dataframe(display,use_container_width=True,hide_index=True)

    st.caption("Source: CPCB India · Safe threshold: WHO & Indian Standards")



# PAGE 2: PREDICT
elif page == "🔮  Predict Contamination":
    # Dynamic background based on last prediction
    _bg_colors = {
        "safe"   : "#d6f5e3",   # soft baby green
        "mod"    : "#fff8e1",   # soft warm yellow
        "unsafe" : "#fde8e8",   # soft red
        "default": "#ddeeff",   # same baby blue as rest of app
    }
    _bg = _bg_colors.get(st.session_state.predict_bg, "#ddeeff")
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stAppViewContainer"],
    [data-testid="stApp"] {{
        background-color: {_bg} !important;
        transition: background-color 0.6s ease;
    }}
    </style>""", unsafe_allow_html=True)

    st.markdown('<p class="page-title">🔮 Predict Fecal Coliform</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Enter water parameters · Extra Trees model (tuned) · Threshold: 50 MPN/100 mL (CPCB Class A)</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Model Info")
        st.metric("Algorithm",   "Extra Trees")
        st.metric("CV R²",       "~0.30")
        st.metric("Safe limit",  "50 MPN/100mL")
        if load_err:
            st.error(f"Model not loaded:\n{load_err}")

    col_left,col_right = st.columns([1.1,1],gap="large")

    with col_left:
        st.markdown("#### 📥 Water Parameters")
        c1,c2 = st.columns(2)
        with c1:
            temp    = st.number_input("🌡️ Temperature (°C)",    0.0,  45.0,  28.0, 0.5)
            ph      = st.number_input("⚗️ pH",                   0.0,  14.0,   7.2, 0.1)
            bod     = st.number_input("🧫 BOD (mg/L)",           0.0, 300.0,   3.0, 0.1)
        with c2:
            do      = st.number_input("💨 Dissolved O₂ (mg/L)", 0.0,  20.0,   6.5, 0.1)
            cond    = st.number_input("⚡ Conductivity (µS/cm)", 0.0,50000.0, 420.0,10.0)
            nitrate = st.number_input("🌿 Nitrate/Nitrite (mg/L)",0.0,100.0,  1.2, 0.1)
        wbt = st.selectbox("🌊 Water Body Type", ["LAKE","POND","TANK","WETLAND"])
        predict_btn = st.button("🔮 Predict Fecal Coliform", type="primary", use_container_width=True)

    with col_right:
        st.markdown("#### 📊 Result")
        if predict_btn:
            if model is None:
                st.error("Model artifacts not found. See sidebar.")
            else:
                do_sqrt          = np.sqrt(max(do,0))
                conductivity_log = np.log1p(cond)
                bod_log          = np.log1p(bod)
                nitrate_log      = np.log1p(nitrate)
                bod_temp_log     = np.log1p(bod*temp)
                bod_cond_log     = np.log1p(bod*cond)
                nitrate_temp     = nitrate*temp
                is_monsoon       = 0   # not used no month input
                state_enc        = float(np.mean(list(state_freq.values()))) if state_freq else 0.03
                all_wbt          = ["POND","TANK","WETLAND"]
                wbt_cols         = {f"Water_Body_Type_{cat}": int(wbt==cat) for cat in all_wbt}
                row = {"Temperature":temp,"DO_sqrt":do_sqrt,"pH":ph,
                       "Conductivity_log":conductivity_log,"BOD_log":bod_log,
                       "Nitrate_Nitrite_log":nitrate_log,"BOD_Temp_log":bod_temp_log,
                       "BOD_Conductivity_log":bod_cond_log,"Nitrate_Temp":nitrate_temp,
                       "Is_Monsoon":is_monsoon,"State_freq":state_enc,**wbt_cols}
                X_in = pd.DataFrame([row])
                for col in feature_cols:
                    if col not in X_in.columns: X_in[col]=0
                X_in = X_in[feature_cols]
                y_qt   = model.predict(X_in)
                y_pred = float(qt.inverse_transform(y_qt.reshape(-1,1)).ravel().clip(0)[0])
                tree_preds = np.array([t.predict(X_in)[0] for t in model.estimators_])
                confidence = float(max(0,min(100,100-np.std(tree_preds)*40)))
                label,color,is_safe = get_safety_predict(y_pred)
                box_cls = "result-safe" if is_safe else ("result-mod" if y_pred<=500 else "result-unsafe")
                # Update background color based on result
                st.session_state.predict_bg = "safe" if is_safe else ("mod" if y_pred<=500 else "unsafe")
                st.markdown(f"""
                <div class="result-box {box_cls}">
                  <div class="result-number">{y_pred:,.0f}</div>
                  <div class="result-unit">MPN / 100 mL</div>
                  <div class="result-badge">{label}</div>
                </div>""",unsafe_allow_html=True)
                st.markdown(f"""
                <div style="margin-bottom:16px">
                  <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#3d5a6b;margin-bottom:4px">
                    <span>Model Confidence</span><span><b>{confidence:.0f}%</b></span>
                  </div>
                  <div class="conf-track"><div class="conf-fill" style="width:{confidence}%"></div></div>
                </div>""",unsafe_allow_html=True)
                st.markdown("**Input summary**")
                sc = st.columns(3)
                for i,(l,v) in enumerate([("Temp",f"{temp}°C"),("DO",f"{do}mg/L"),("pH",f"{ph}"),
                                           ("BOD",f"{bod}mg/L"),("Cond",f"{cond:.0f}"),("NO₃",f"{nitrate}mg/L")]):
                    with sc[i%3]: st.metric(l,v)
                st.markdown("---")
                st.markdown("**Top feature importances**")
                imp = pd.Series(model.feature_importances_,index=feature_cols).sort_values(ascending=False).head(10)
                imp_df = imp.reset_index(); imp_df.columns=["Feature","Importance"]
                imp_df["Feature"] = imp_df["Feature"].str.replace("_"," ")
                st.bar_chart(imp_df.set_index("Feature"),height=240)

        else:
            st.markdown("""
            <div style="border:2px dashed #c9b99a;border-radius:14px;padding:40px 24px;
                        text-align:center;color:#7a9aaa;">
              <div style="font-size:2.5rem;margin-bottom:12px">🌊</div>
              <div style="font-size:1rem;font-weight:500">Fill in parameters and click <b>Predict</b></div>
              <div style="font-size:0.8rem;margin-top:12px">Safe threshold: 50 MPN/100 mL</div>
            </div>""",unsafe_allow_html=True)
            if df is not None:
                st.markdown("---")
                st.markdown("**📈 Historical median Fecal Coliform by Year**")
                yearly = df.groupby("Year")["Fecal_Coliform"].median().reset_index()
                yearly.columns = ["Year","Median FC"]
                st.line_chart(yearly.set_index("Year"),height=200)

    st.caption("Model: Extra Trees (tuned) · CPCB India 2017–2022")



# PAGE3:  LEARN & TIPS
elif page == "💡  Learn & Tips":
    st.markdown('<p class="page-title">💡 Learn & Tips</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Water facts, safety tips, which water to drink, and how to save water</p>', unsafe_allow_html=True)

    FACTS = [
        ("3.4M",   "people die each year from water-related diseases making it one of the leading causes of death globally."),
        ("1 in 3", "people globally do not have access to safe drinking water at home."),
        ("80%",    "of diseases in developing countries are linked to unsafe water and poor sanitation."),
        ("50L",    "per day is the minimum recommended by the UN for basic human needs: drinking, cooking, hygiene."),
        ("There is about the same amount of water on Earth now as there was millions of years ago."),
        ("Earth holds about 1.386 billion cubic kilometers of water."),
        ("Of all the water on the planet, about 2.5 – 2.75 % is freshwater, and an even smaller fraction — around 0.3 % — is surface water in rivers and lakes that people use most directly."),
        ("Only roughly 0.014 % of all water on Earth is both fresh and easily accessible in surface and near-surface sources that people can use directly."),
        ("More than 68 % of freshwater is stored in ice sheets and glaciers, making these frozen reservoirs the planet’s largest freshwater pool."),
        ("About 30 % of freshwater is stored as groundwater in aquifers below the surface."),
        ("Antarctica and Greenland together hold the vast majority of the world’s ice-bound freshwater, but human-accessible fresh water is spread across glaciers, groundwater, lakes and rivers."),
        ("oughly 2.1 % of all Earth’s water is frozen in glaciers and ice caps — a bigger share than older figures suggested."),
    ]

    # Random fact card
    st.markdown("### Water Fact of the Day")
    if "fact_idx" not in st.session_state:
        st.session_state.fact_idx = random.randint(0, len(FACTS)-1)
    num, text = FACTS[st.session_state.fact_idx]
    st.markdown(f"""
    <div class="fact-card">
      <div class="fact-number">{num}</div>
      <div style="font-size:1.05rem;color:#1a6e8a;margin-top:8px;line-height:1.5">{text}</div>
    </div>""", unsafe_allow_html=True)
    if st.button(" Next Fact"):
        st.session_state.fact_idx = random.randint(0, len(FACTS)-1)
        st.rerun()

    st.divider()

    # Which water to drink
    tab1, tab2, tab3 = st.tabs([" Which Water to Drink?", " Safety Tips", " Water Saving"])

    with tab1:
        st.markdown("### Which Water Source is Safest?")
        water_types = [
            ("🏔️", "Mountain Spring Water",  "Excellent",  "#2e9e6e",
             "Naturally filtered through rock. Very low contamination risk. Best choice if available."),
            ("🚰", "Municipal Tap Water",     "Good",       "#3fa8c8",
             "Treated and chlorinated. Safe in most Indian cities. Always boil if unsure."),
            ("🪣", "Filtered/RO Water",       "Good",       "#3fa8c8",
             "RO removes dissolved solids, bacteria, and viruses. Reliable for daily use."),
            ("💧", "Bottled Water",           "Good",       "#7ecab8",
             "Generally safe but check BIS certification (IS 14543). Avoid if seal is broken."),
            ("🌊", "River Water (untreated)", "Dangerous",  "#c94040",
             "High risk of Fecal Coliform, BOD, and heavy metals. Never drink without treatment."),
            ("🌿", "Pond / Lake (untreated)", "Dangerous",  "#c94040",
             "Stagnant water breeds bacteria. Fecal Coliform often exceeds 5,000 MPN/100ml."),
            ("🪨", "Groundwater / Borewells", "Moderate",   "#d48f00",
             "Risk of arsenic, fluoride, and nitrate contamination. Always test before drinking."),
            ("🌧️", "Rainwater (harvested)",  "Moderate",   "#d48f00",
             "Clean at source but contaminated by rooftops. Filter and disinfect before use."),
        ]
        for icon, name, safety, color, desc in water_types:
            st.markdown(f"""
            <div class="tip-card" style="border-left:4px solid {color}">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
                <span style="font-size:1.5rem">{icon}</span>
                <span style="font-weight:600;font-size:1rem;color:#0d3b4f">{name}</span>
                <span style="margin-left:auto;background:{color};color:white;
                  padding:2px 10px;border-radius:12px;font-size:0.78rem;font-weight:600">{safety}</span>
              </div>
              <div style="font-size:0.87rem;color:#3d5a6b">{desc}</div>
            </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("### 🛡️ Water Safety Tips")
        tips = [
            ("🔥", "Always Boil Suspicious Water",
             "Boil water for at least 1 minute to kill bacteria, viruses, and parasites. At high altitudes, boil for 3 minutes."),
            ("🧪", "Test Your Water Annually",
             "Home water testing kits can detect pH, chlorine, nitrates, and bacteria. Test borewell water every 6 months."),
            ("🚿", "Clean Your Storage Tanks",
             "Water tanks are breeding grounds for bacteria. Clean and disinfect overhead tanks every 3–6 months."),
            ("🔍", "Check for Signs of Contamination",
             "Discolored water, unusual smell, or strange taste are red flags. Don't drink — report to your local authority."),
            ("📦", "Proper Storage",
             "Store water in covered, food-grade containers. Avoid plastic containers exposed to sunlight."),
            ("💊", "Use Water Purification Tablets",
             "When travelling or during emergencies, water purification tablets (chlorine/iodine) are effective and cheap."),
            ("🌡️", "Monsoon Extra Caution",
             "Fecal Coliform spikes during monsoon season (June–September). Switch to RO or boiled water during this period."),
            ("🏥", "Know the Symptoms",
             "Diarrhoea, vomiting, stomach cramps, and fever after drinking water = possible contamination. Seek medical help."),
        ]
        col_a, col_b = st.columns(2)
        for i, (icon, title, text) in enumerate(tips):
            with (col_a if i%2==0 else col_b):
                st.markdown(f"""
                <div class="tip-card">
                  <div class="tip-icon">{icon}</div>
                  <div class="tip-title">{title}</div>
                  <div class="tip-text">{text}</div>
                </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown("### ♻️ Water Conservation Tips")

        st.markdown("#### 🏠 At Home")
        home_tips = [
            ("🚿","Shorter Showers","Cutting shower time by 2 minutes saves ~30 litres per shower."),
            ("🪣","Fix Leaks Fast","A dripping tap wastes up to 20,000 litres per year. Fix it today."),
            ("🌿","Water Plants Wisely","Water in the early morning or evening to reduce evaporation by 50%."),
            ("🧺","Full Loads Only","Run washing machines and dishwashers only when full — saves 30–50 litres per cycle."),
            ("🚽","Dual-Flush Toilets","Switching to dual-flush saves up to 67% of toilet water usage."),
        ]
        for icon, title, text in home_tips:
            st.markdown(f"""
            <div class="tip-card" style="border-left:4px solid #7ecab8">
              <b>{icon} {title}</b>
              <div class="tip-text">{text}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### 🌾 Water Stress by Region")
        if df is not None:
            state_avg = (df.groupby("State Name")["Fecal_Coliform"]
                         .median().sort_values(ascending=False).head(10).reset_index())
            state_avg.columns = ["State","Median FC (MPN/100ml)"]
            state_avg["State"] = state_avg["State"].str.title()
            st.markdown("**Top 10 most contaminated states (median Fecal Coliform):**")
            st.bar_chart(state_avg.set_index("State"), height=280)
            st.caption("Higher contamination = more urgent need for water safety measures.")

    st.caption("Sources: WHO, UNICEF, CPCB India, UN Water")


# PAGE 4 — REPORT POLLUTION
elif page == "🚨  Report Pollution":
    st.markdown('<p class="page-title">🚨 Report Pollution</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Help your community — report polluted water bodies. Reports are shown on the map.</p>', unsafe_allow_html=True)

    col_form, col_map = st.columns([1, 1.2], gap="large")

    with col_form:
        st.markdown("#### 📝 Submit a Report")

        with st.form("report_form", clear_on_submit=True):
            reporter  = st.text_input("Your Name (optional)", placeholder="Anonymous")
            state_r   = st.selectbox("State", sorted(STATE_COORDS.keys()),
                                     format_func=lambda x: x.title())
            wb_r      = st.selectbox("Water Body Type",
                                     ["River","Lake","Pond","Tank","Wetland","Canal","Other"])
            severity  = st.select_slider("Severity",
                options=["Low — minor discoloration",
                         "Medium — bad smell / foam",
                         "High — dead fish / visible waste",
                         "Critical — chemical spill / industrial discharge"])
            desc      = st.text_area("Describe what you observed",
                placeholder="e.g. Dark oily water near the river bank, strong chemical smell...")
            photo     = st.file_uploader("📷 Upload a photo (optional)",
                                         type=["jpg","jpeg","png","webp"])
            submitted = st.form_submit_button("🚨 Submit Report", type="primary", use_container_width=True)

        if submitted:
            if not desc.strip():
                st.error("Please describe what you observed.")
            else:
                lat, lon = STATE_COORDS.get(state_r, (22.5, 82.0))
                # Small random offset so reports don't stack exactly
                lat += random.uniform(-0.3, 0.3)
                lon += random.uniform(-0.3, 0.3)

                photo_path = ""
                if photo:
                    fname = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{photo.name}"
                    fpath = os.path.join(UPLOADS_DIR, fname)
                    img   = Image.open(photo)
                    img.save(fpath)
                    photo_path = fpath

                record = {
                    "timestamp"  : datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "reporter"   : reporter.strip() or "Anonymous",
                    "state"      : state_r,
                    "water_body" : wb_r,
                    "severity"   : severity.split("—")[0].strip(),
                    "description": desc.strip(),
                    "lat"        : round(lat, 4),
                    "lon"        : round(lon, 4),
                    "photo_path" : photo_path,
                }
                save_report(record)
                st.success("✅ Report submitted! It will appear on the map.")
                if photo:
                    st.image(photo, caption="Uploaded photo", width=300)

        # Stats
        reports_df = load_reports()
        if not reports_df.empty:
            st.divider()
            st.markdown("#### 📊 Report Statistics")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Reports", len(reports_df))
            c2.metric("States Covered", reports_df["state"].nunique())
            high = (reports_df["severity"].str.contains("High|Critical",case=False,na=False)).sum()
            c3.metric("High/Critical", int(high))

    with col_map:
        st.markdown("#### 🗺️ Pollution Reports Map")
        reports_df = load_reports()

        m3 = folium.Map(location=[22.5, 82.0], zoom_start=5,
                        tiles="CartoDB positron", min_zoom=4, max_zoom=12)
        m3.fit_bounds([[6.5, 68.0], [35.5, 97.5]])

        SEVERITY_COLORS = {
            "Low":      "green",
            "Medium":   "orange",
            "High":     "red",
            "Critical": "darkred",
        }

        if not reports_df.empty:
            for _, r in reports_df.iterrows():
                try:
                    sev   = r["severity"].split("—")[0].strip() if "—" in str(r["severity"]) else str(r["severity"])
                    color = SEVERITY_COLORS.get(sev, "red")
                    folium.Marker(
                        location=[float(r["lat"]), float(r["lon"])],
                        icon=folium.Icon(color=color, icon="warning-sign", prefix="glyphicon"),
                        tooltip=f"⚠️ {r['water_body']} — {sev} severity",
                        popup=folium.Popup(
                            f"<div style='font-family:Arial;min-width:180px'>"
                            f"<b>🚨 {sev} Severity</b><br>"
                            f"<b>State:</b> {str(r['state']).title()}<br>"
                            f"<b>Water body:</b> {r['water_body']}<br>"
                            f"<b>Reported by:</b> {r['reporter']}<br>"
                            f"<b>When:</b> {r['timestamp']}<br><br>"
                            f"<i>{r['description']}</i></div>",
                            max_width=260),
                    ).add_to(m3)
                except: pass
            st.caption(f"Showing {len(reports_df)} report(s). Click markers for details.")
        else:
            st.caption("No reports yet. Be the first to report a pollution issue!")

        st_folium(m3, width="100%", height=520, returned_objects=[])

        # Recent reports table
        if not reports_df.empty:
            st.markdown("#### 📋 Recent Reports")
            show = reports_df[["timestamp","state","water_body","severity","description","reporter"]].copy()
            show.columns = ["Time","State","Water Body","Severity","Description","Reporter"]
            show["State"] = show["State"].str.title()
            show = show.sort_values("Time", ascending=False).head(10).reset_index(drop=True)
            st.dataframe(show, use_container_width=True, hide_index=True)

    st.caption("Reports are stored locally in data/reports.csv · Photos saved in uploads/")