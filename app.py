"""
commL'Eau Claire · 5-page Streamlit App
Run with: streamlit run app.py
"""

import os
import joblib
import gdown
import streamlit as st

# Add these missing imports to fix the undefined variable errors:
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import random
from datetime import datetime
from PIL import Image

# === CONFIG ===
DRIVE_URL = "https://drive.google.com/drive/folders/11ueQ6B1vR2pNTkkn5FDdzmiqGTtkzstr?usp=sharing"
MODEL_DIR = "model/artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")


@st.cache_resource
def load_model():

    # create folder if not exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # if model not downloaded yet → download folder
    if not os.path.exists(MODEL_PATH):

        st.info("Downloading model from Google Drive...")

        gdown.download_folder(
            url=DRIVE_URL,
            output=MODEL_DIR,
            quiet=False,
            use_cookies=False
        )

        st.success("Model downloaded successfully!")

    # load model
    model = joblib.load(MODEL_PATH)

    return model


model = load_model()

# CONSTANTS
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

ARTIFACTS_DIR = r"D:\mini_water_predict\model\artifacts"
REPORTS_CSV   = r"D:\mini_water_predict\data\reports.csv"
UPLOADS_DIR   = "uploads"

for folder in ["data", UPLOADS_DIR]:
    try:
        os.makedirs(folder, exist_ok=True)
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# GIF ASSETS
# ══════════════════════════════════════════════════════════════════════════════
GIF = {
    "sidebar":        "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdzBraHpoZ3RmNmdiemhtem43aTR2M2VoMzNqbmE5Ym5xNTJmdGNvayZlcD12MV9naWZzX3NlYXJjaCZjdD1n/Sbsf1Zsamps1zgZyZj/giphy.gif",
    "no_data":        "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdzBraHpoZ3RmNmdiemhtem43aTR2M2VoMzNqbmE5Ym5xNTJmdGNvayZlcD12MV9naWZzX3NlYXJjaCZjdD1n/m5KDfLMRIS2CwCXVKG/giphy.gif",
    "predict_idle":   "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdzBraHpoZ3RmNmdiemhtem43aTR2M2VoMzNqbmE5Ym5xNTJmdGNvayZlcD12MV9naWZzX3NlYXJjaCZjdD1n/IKFZJeAA5A8OOumLHx/giphy.gif",
    "water_cycle":    "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdzBraHpoZ3RmNmdiemhtem43aTR2M2VoMzNqbmE5Ym5xNTJmdGNvayZlcD12MV9naWZzX3NlYXJjaCZjdD1n/7mSgaTrLetqpOdnZ9q/giphy.gif",
    "report_hero":    "https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3cXhvajhzd3NkeGtpM2x4NHZ1ZGJ4NTF2d3EwbmRsczZtOGxicDFrdCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/bgOQ2Mx4uLnsoyyIg9/giphy.gif",
    "tip_boil":       "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbW43OXRpaHg0bDYwMWNsdzY4eDAxeHZ5b2ZzYWhrbnhlN3YwYXZobiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/vnNHyroyEwb4s/giphy.gif",
    "tip_test":       "https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3bnM3MWQzcG0zb3hpNHVnYXNleWxzdGJkZ3h2M2ZlNnMzNmhjM3l2dyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/OalJFGFBD4D2xn61qb/giphy.gif",
    "tip_tank":       "https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3YW1hb2c1dTNqODl0a3lpdXNxOHZ3OHR5cmV0cjA0OXB2NjE3bWgzMyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/5bCYrRAZ3b5HtQ5HuJ/giphy.gif",
    "tip_check":      "https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3ZW45bmV6N2VuN3J1ZzlsNGlqdzYyNzJiYTd1MHQ0MGd2Y3gxeGN0NCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/gLWZMy6cpK4Cca2MpA/giphy.gif",
    "tip_store":      "https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3aGlnZHhkOTV3aHVibmZjamYzb2RvZjVpcDZpMDNkMWZrczhzeWd2cyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/4uKGppPmeVBCzwfnMK/giphy.gif",
    "tip_tablet":     "https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3NWxkdTdpZjBlNXBoNWtqYmkydGZqc3ZlZmNyYTVmcTYxMTJrcjVkZiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/VehoU0h2Rl8Gc/giphy.gif",
    "tip_monsoon":    "https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3aGlnZHhkOTV3aHVibmZjamYzb2RvZjVpcDZpMDNkMWZrczhzeWd2cyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/Q2koSsz3l42ZIiboyW/giphy.gif",
    "tip_symptoms":   "https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3dmN4b3pwZnhzcmppZTF3ZDE1OHhvbTM1NGx1NHA3bHV3bnZhcTJ5NiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/LI6TgnchmtJ60/giphy.gif",
    "src_mountain":   "https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3eW10OTVjOGFtYjBzMWpqdmkycmkyZjc2ZGpucHQ3anVueDNpZnJkZiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/bc7Ae3mL7ZAEo/giphy.gif",
    "src_tap":        "https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3azNiZ2V5bTAyMTA1bGpvczJlanB3OGQ0OTNycjZveGZoaTIyZXY2YiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/fVYhJVclr91E4/giphy.gif",
    "src_ro":         "https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3azNiZ2V5bTAyMTA1bGpvczJlanB3OGQ0OTNycjZveGZoaTIyZXY2YiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/Z4oPJW36axa32zBA6C/giphy.gif",
    "src_bottle":     "https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3dmN4b3pwZnhzcmppZTF3ZDE1OHhvbTM1NGx1NHA3bHV3bnZhcTJ5NiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/TJpp166ImcK1a/giphy.gif",
    "src_river":      "https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3azNiZ2V5bTAyMTA1bGpvczJlanB3OGQ0OTNycjZveGZoaTIyZXY2YiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/VQ5vmfZWGnXqCH5vul/giphy.gif",
    "src_pond":       "https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3YXZ6ZTR1YTlkbHpwaHVnajM1eDBtaGRvcGtyNm9tYXBjdTMzeDB4NyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/SFQiIj7DXwl3ypIV5B/giphy.gif",
    "src_ground":     "https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3N2hjMWllanRkcW9wOGN2MW56eGh2Y2xjdnJiNDV6dTRqcGJ5ZjZsdSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/9TKBUr2RwHLd8ehAjg/giphy.gif",
    "src_rain":       "https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3ZW45bmV6N2VuN3J1ZzlsNGlqdzYyNzJiYTd1MHQ0MGd2Y3gxeGN0NCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/G0Odfjd78JTpu/giphy.gif",
}

def gif(key, size=40):
    return f'<img src="{GIF[key]}" style="width:{size}px;height:{size}px;object-fit:cover;border-radius:6px;vertical-align:middle">'


# SAFETY HELPERS

# Map page uses loose display thresholds (<=500 safe, <=5000 moderate)
# matching general Indian water quality categories.
def get_safety_map(fc):
    if fc <= 500:    return "Safe to drink",           "green",  "Safe"
    elif fc <= 5000: return "Not safe to drink",       "orange", "Moderate"
    else:            return "Dangerous avoid contact", "red",    "Unsafe"

# Prediction page uses CPCB Class A threshold (<=50 = safe).
def get_safety_predict(fc):
    if fc <= 50:    return "Safe (Class A)",   "#2e9e6e", True
    elif fc <= 500: return "Moderate risk",    "#d48f00", False
    else:           return "High risk unsafe", "#c94040", False

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    dataset_dir = "d:\mini_water_predict\Dataset"
    if not os.path.isdir(dataset_dir):
        return None
    candidates = [
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.endswith(".csv")
    ]
    for path in sorted(candidates):
        try:
            df = pd.read_csv(path)
            for col in ["State Name", "Type Water Body"]:
                if col not in df.columns:
                    continue
                df[col] = (df[col].astype(str)
                           .str.replace(r"[\n\r]", " ", regex=True)
                           .str.replace(r"\s+", " ", regex=True)
                           .str.strip().str.upper())
            raw_pairs = [
                ("Temperature",     "Min Temperature",           "Max Temperature"),
                ("DO",              "Min Dissolved Oxygen",      "Max Dissolved Oxygen"),
                ("pH",              "Min pH",                    "Max pH"),
                ("Conductivity",    "Min Conductivity",          "Max Conductivity"),
                ("BOD",             "Min BOD",                   "Max BOD"),
                ("Nitrate_Nitrite", "Min Nitrate N + Nitrite N", "Max Nitrate N + Nitrite N"),
                ("Fecal_Coliform",  "Min Fecal Coliform",        "Max Fecal Coliform"),
                ("Total_Coliform",  "Min Total Coliform",        "Max Total Coliform"),
            ]
            for avg_col, mn, mx in raw_pairs:
                for c in [mn, mx]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(
                            df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
                if mn in df.columns and mx in df.columns:
                    df[avg_col] = (df[mn] + df[mx]) / 2
            df = df.dropna(subset=["Fecal_Coliform", "State Name", "Type Water Body"])
            df = df[df["Fecal_Coliform"] >= 0]
            if len(df) > 0:
                return df
        except Exception:
            continue
    return None

@st.cache_resource
def load_artifacts():
    base = ARTIFACTS_DIR

    try:
        model = joblib.load(os.path.join(base, "best_model.pkl"))
        feature_cols = joblib.load(os.path.join(base, "feature_cols.pkl"))
        qt = joblib.load(os.path.join(base, "quantile_transformer.pkl"))

        sf_path = os.path.join(base, "state_freq_mapping.pkl")
        state_freq = joblib.load(sf_path) if os.path.exists(sf_path) else {}

        return model, feature_cols, qt, state_freq, None

    except Exception as e:
        return None, None, None, {}, str(e)

def load_reports():
    if os.path.exists(REPORTS_CSV):
        try:
            return pd.read_csv(REPORTS_CSV)
        except Exception:
            pass
    return pd.DataFrame(columns=[
        "timestamp", "reporter", "state", "water_body", "severity",
        "description", "lat", "lon", "photo_path"])

def save_report(record: dict):
    try:
        df = load_reports()
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        os.makedirs("data", exist_ok=True)
        df.to_csv(REPORTS_CSV, index=False)
        return True
    except Exception:
        return False

def get_session_reports_df():
    base = load_reports()
    if "session_reports" in st.session_state and st.session_state.session_reports:
        return pd.concat([base, pd.DataFrame(st.session_state.session_reports)], ignore_index=True)
    return base

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg-main:        #ddeeff;
  --bg-sidebar:     #c8e0f4;
  --bg-card:        #eef6ff;
  --bg-input:       #f5faff;
  --border-card:    #b8d8f5;
  --text-primary:   #0d3b4f;
  --text-secondary: #3d5a6b;
  --text-muted:     #5a7a99;
  --text-light:     #7a9aaa;
  --accent-blue:    #3fa8c8;
  --accent-dark:    #1a6e8a;
  --fact-bg-from:   #cce8ff;
  --fact-bg-to:     #b8d8f5;
  --conf-track:     #b8d8f5;
  --shadow:         rgba(13,59,79,0.10);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg-main:        #0e1e2e;
    --bg-sidebar:     #0a1624;
    --bg-card:        #132336;
    --bg-input:       #1a2d42;
    --border-card:    #1e3a54;
    --text-primary:   #e8f4fd;
    --text-secondary: #a8c8e0;
    --text-muted:     #7aa8c4;
    --text-light:     #5a88a4;
    --accent-blue:    #4db8d8;
    --accent-dark:    #3fa8c8;
    --fact-bg-from:   #0d2a40;
    --fact-bg-to:     #0a2035;
    --conf-track:     #1e3a54;
    --shadow:         rgba(0,0,0,0.40);
  }
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
[data-testid="stAppViewContainer"] > .main {
  background-color: var(--bg-main) !important;
  font-family: 'DM Sans', sans-serif;
  color: var(--text-primary) !important;
}
[data-testid="stSidebar"]   { background-color: var(--bg-sidebar) !important; }
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stForm"],
.stNumberInput input,
div[data-baseweb="input"] input,
div[data-baseweb="select"] {
  background-color: var(--bg-input) !important;
  color: var(--text-primary) !important;
  border-color: var(--border-card) !important;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] { color: var(--text-primary) !important; }
[data-testid="stDataFrame"]   { background-color: var(--bg-card) !important; }
[data-testid="stTabs"] button { color: var(--text-muted) !important; }
[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--accent-dark) !important;
  border-bottom-color: var(--accent-dark) !important;
}
hr { border-color: var(--border-card) !important; }

.page-title {
  font-family: 'Playfair Display', serif;
  font-size: 2.1rem; font-weight: 700;
  color: var(--text-primary); margin-bottom: 0; line-height: 1.2;
}
.page-sub {
  font-size: 0.88rem; color: var(--text-muted);
  margin-top: 4px; margin-bottom: 1.2rem;
}
.fact-card {
  background: linear-gradient(135deg, var(--fact-bg-from), var(--fact-bg-to));
  border-radius: 16px; padding: 24px 28px;
  border-left: 5px solid var(--accent-blue);
  margin-bottom: 16px; box-shadow: 0 4px 16px var(--shadow);
}
.fact-number {
  font-family: 'Playfair Display', serif;
  font-size: 2.8rem; font-weight: 700;
  color: var(--text-primary); line-height: 1;
}
.fact-text { font-size: 1.05rem; color: var(--accent-dark); margin-top: 8px; line-height: 1.5; }
.tip-card {
  background: var(--bg-card); border-radius: 12px;
  padding: 16px 18px; border: 1.5px solid var(--border-card);
  box-shadow: 0 2px 8px var(--shadow); margin-bottom: 10px;
}
.tip-header { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.tip-title  { font-weight: 600; color: var(--text-primary); font-size: 1rem; }
.tip-text   { font-size: 0.87rem; color: var(--text-secondary); line-height: 1.5; }
.result-box {
  border-radius: 14px; padding: 24px 28px; margin-bottom: 14px;
  border: 1.5px solid transparent; box-shadow: 0 4px 20px var(--shadow);
}
.result-safe   { background: #1a4a32; border-color: #2e9e6e; }
.result-mod    { background: #4a3a00; border-color: #d48f00; }
.result-unsafe { background: #4a1a1a; border-color: #c94040; }
@media (prefers-color-scheme: light) {
  .result-safe   { background: #d4f4e8; border-color: rgba(46,158,110,0.3); }
  .result-mod    { background: #fff3cd; border-color: rgba(212,143,0,0.3); }
  .result-unsafe { background: #fce4e4; border-color: rgba(201,64,64,0.3); }
}
.result-number {
  font-family: 'Playfair Display', serif; font-size: 3.2rem;
  font-weight: 700; line-height: 1; color: var(--text-primary); letter-spacing: -1px;
}
.result-unit  { font-size: 0.85rem; color: var(--text-light); margin-top: 4px; }
.result-badge { font-size: 1.05rem; font-weight: 600; margin-top: 12px; }
.conf-wrap { margin-bottom: 18px; }
.conf-label-row {
  display: flex; justify-content: space-between;
  font-size: 0.82rem; color: var(--text-secondary); margin-bottom: 6px;
}
.conf-track { background: var(--conf-track); border-radius: 8px; height: 10px; overflow: hidden; }
.conf-fill  { height: 100%; border-radius: 8px; background: linear-gradient(90deg, #7ecab8, var(--accent-blue)); }
.summary-grid {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 10px; margin: 14px 0;
}
.summary-chip {
  background: var(--bg-card); border: 1.5px solid var(--border-card);
  border-radius: 10px; padding: 12px 14px; box-shadow: 0 2px 8px var(--shadow);
}
.summary-chip-label {
  font-size: 0.72rem; font-weight: 600; color: var(--text-light);
  text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 4px;
}
.summary-chip-value {
  font-family: 'Playfair Display', serif;
  font-size: 1.3rem; font-weight: 600; color: var(--text-primary);
}
.empty-state {
  border: 2px dashed var(--border-card); border-radius: 14px;
  padding: 40px 24px; text-align: center;
}
.empty-state img.hero-gif {
  width: 140px; height: 140px; object-fit: cover;
  border-radius: 50%; margin-bottom: 18px;
  box-shadow: 0 4px 20px var(--shadow);
}
.empty-text { font-size: 1rem; font-weight: 500; color: var(--text-muted); }
.empty-sub  { font-size: 0.8rem; margin-top: 8px; color: var(--text-light); }
.nodata-banner {
  background: var(--bg-card); border: 1.5px solid var(--border-card);
  border-left: 5px solid var(--accent-blue); border-radius: 12px;
  padding: 16px 20px; margin-bottom: 16px;
  display: flex; align-items: center; gap: 14px;
}
.nodata-banner img { width: 48px; height: 48px; border-radius: 8px; object-fit: cover; }
.nodata-banner-text { flex: 1; }
.nodata-banner-title { font-weight: 600; color: var(--text-primary); font-size: 1rem; margin-bottom: 4px; }
.nodata-banner-sub   { font-size: 0.83rem; color: var(--text-muted); line-height: 1.5; }
.nodata-code {
  display: inline-block; background: var(--bg-input); border: 1px solid var(--border-card);
  border-radius: 6px; padding: 2px 10px;
  font-family: monospace; font-size: 0.82rem; color: var(--accent-dark);
}
.src-row {
  display: flex; align-items: flex-start; gap: 14px;
  background: var(--bg-card); border-radius: 12px;
  padding: 14px 16px; border: 1.5px solid var(--border-card);
  box-shadow: 0 2px 8px var(--shadow); margin-bottom: 10px;
}
.src-row img { width: 48px; height: 48px; border-radius: 8px; object-fit: cover; flex-shrink: 0; }
.src-body    { flex: 1; }
.src-name    { font-weight: 600; font-size: 1rem; color: var(--text-primary); margin-bottom: 4px; }
.src-desc    { font-size: 0.85rem; color: var(--text-secondary); line-height: 1.5; }
.src-badge   {
  display: inline-block; padding: 2px 12px; border-radius: 12px;
  font-size: 0.78rem; font-weight: 600; color: #fff; margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        f'<img src="{GIF["sidebar"]}" style="width:100%;border-radius:12px;margin-bottom:8px">',
        unsafe_allow_html=True,
    )
    st.markdown("### L'Eau Claire")
    st.caption("Water Quality Monitor · 2017–2022")
    st.divider()
    page = st.radio("Navigate", [
        "Water Quality Map",
        "Predict Contamination",
        "Forecast Comparison",
        "Learn & Tips",
        "Report Pollution",
    ], label_visibility="collapsed")
    st.divider()

df = load_data()
model, feature_cols, qt, state_freq, load_err = load_artifacts()
state_list = sorted(state_freq.keys()) if state_freq else sorted(STATE_COORDS.keys())

if "predict_bg" not in st.session_state:
    st.session_state.predict_bg = "default"

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: WATER QUALITY MAP
# ══════════════════════════════════════════════════════════════════════════════
if page == "Water Quality Map":
    st.session_state.predict_bg = "default"
    st.markdown('<p class="page-title">India Water Quality Map</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Drinking water safety across Indian states · Fecal Coliform levels 2017–2022</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.success("Safe  ≤ 500 MPN/100ml")
    c2.warning("Moderate  501–5,000 MPN/100ml")
    c3.error("Dangerous  > 5,000 MPN/100ml")
    st.divider()

    with st.sidebar:
        st.header("Filters")
        if df is not None:
            wb_options    = ["ALL"] + sorted(df["Type Water Body"].unique().tolist())
            selected_wb   = st.selectbox("Water Body Type", wb_options)
            year_options  = ["ALL"] + sorted(df["Year"].dropna().unique().astype(int).tolist())
            selected_year = st.selectbox("Year", year_options)
            st.divider()
            st.header("Safety Summary")
            state_agg  = df.groupby("State Name")["Fecal_Coliform"].median().reset_index()
            safe_n     = (state_agg["Fecal_Coliform"] <= 500).sum()
            moderate_n = ((state_agg["Fecal_Coliform"] > 500) & (state_agg["Fecal_Coliform"] <= 5000)).sum()
            unsafe_n   = (state_agg["Fecal_Coliform"] > 5000).sum()
            st.write(f"Safe states: **{safe_n}**")
            st.write(f"Moderate states: **{moderate_n}**")
            st.write(f"Unsafe states: **{unsafe_n}**")
        else:
            selected_wb, selected_year = "ALL", "ALL"
            st.info("Load dataset to enable filters.")

    if df is None:
        st.markdown(f"""
        <div class="nodata-banner">
          <img src="{GIF['no_data']}" alt="water">
          <div class="nodata-banner-text">
            <div class="nodata-banner-title">Dataset not found showing all states (no data)</div>
            <div class="nodata-banner-sub">
              Place CSV files inside the <span class="nodata-code">Dataset/</span> folder and restart.
              The map below shows all Indian states as grey markers until data is loaded.
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    m = folium.Map(location=[22.5, 82.0], zoom_start=5,
                   tiles="CartoDB positron", min_zoom=4, max_zoom=12)
    m.fit_bounds([[6.5, 68.0], [35.5, 97.5]])

    import hashlib

    if df is not None:
        plot_df = df.copy()
        if selected_wb   != "ALL": plot_df = plot_df[plot_df["Type Water Body"] == selected_wb]
        if selected_year != "ALL": plot_df = plot_df[plot_df["Year"] == int(selected_year)]
        agg = (plot_df.groupby(["State Name", "Type Water Body"])
               .agg(FC=("Fecal_Coliform", "median"), count=("Fecal_Coliform", "count"))
               .reset_index())

        if not agg.empty:
            for _, row in agg.iterrows():
                state, wb, fc = row["State Name"], row["Type Water Body"], row["FC"]
                if state not in STATE_COORDS: continue
                lat, lon = STATE_COORDS[state]
                h    = int(hashlib.md5(wb.encode()).hexdigest(), 16)
                lat += ((h % 100) - 50) / 800
                lon += ((h % 137) - 68) / 800
                label, color, status = get_safety_map(fc)
                bg = {"green": "#d4edda", "orange": "#fff3cd", "red": "#f8d7da"}[color]
                popup_html = f"""
                <div style="font-family:Arial,sans-serif;min-width:200px;padding:4px">
                  <b style="font-size:15px">📍 {state.title()}</b>
                  <hr style="margin:4px 0">
                  <table style="width:100%;font-size:13px">
                    <tr><td><b>Water Body</b></td><td>{wb.title()}</td></tr>
                    <tr><td><b>Fecal Coliform</b></td><td>{fc:,.0f} MPN/100ml</td></tr>
                    <tr><td><b>Records</b></td><td>{row['count']:,}</td></tr>
                  </table>
                  <div style="margin-top:8px;padding:6px;border-radius:6px;
                    background:{bg};font-weight:bold;font-size:13px;text-align:center">{label}</div>
                </div>"""
                folium.CircleMarker(
                    location=[lat, lon], radius=12, color="white", weight=1.5,
                    fill=True, fill_color=color, fill_opacity=0.85,
                    popup=folium.Popup(popup_html, max_width=260),
                    tooltip=folium.Tooltip(f"<b>{state.title()}</b> — {wb.title()}<br>{label}", sticky=True),
                ).add_to(m)
        else:
            for state, (lat, lon) in STATE_COORDS.items():
                folium.CircleMarker(
                    location=[lat, lon], radius=10, color="white", weight=1.5,
                    fill=True, fill_color="gray", fill_opacity=0.5,
                    tooltip=folium.Tooltip(f"<b>{state.title()}</b><br>No data for selected filters", sticky=True),
                ).add_to(m)
    else:
        for state, (lat, lon) in STATE_COORDS.items():
            popup_html = f"""
            <div style="font-family:Arial,sans-serif;min-width:180px;padding:4px">
              <b style="font-size:15px"> {state.title()}</b>
              <hr style="margin:4px 0">
              <div style="font-size:13px;color:#888">No data loaded yet.<br>
              Add CSV files to the Dataset/ folder.</div>
            </div>"""
            folium.CircleMarker(
                location=[lat, lon], radius=10, color="white", weight=1.5,
                fill=True, fill_color="gray", fill_opacity=0.55,
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=folium.Tooltip(f"<b>{state.title()}</b> — No data", sticky=True),
            ).add_to(m)

    reports_df = get_session_reports_df()
    if not reports_df.empty:
        for _, r in reports_df.iterrows():
            try:
                folium.Marker(
                    location=[float(r["lat"]), float(r["lon"])],
                    icon=folium.Icon(color="red", icon="warning-sign", prefix="glyphicon"),
                    tooltip=f"Report: {r['water_body']} — {r['severity']}",
                    popup=folium.Popup(
                        f"<b>Pollution Report</b><br><b>State:</b> {r['state']}<br>"
                        f"<b>Water Body:</b> {r['water_body']}<br><b>Severity:</b> {r['severity']}<br>"
                        f"<b>Details:</b> {r['description']}<br><b>Reported:</b> {r['timestamp']}",
                        max_width=250),
                ).add_to(m)
            except Exception:
                pass

    st_folium(m, width="100%", height=620, returned_objects=[])

    if df is not None:
        plot_df2 = df.copy()
        if selected_wb   != "ALL": plot_df2 = plot_df2[plot_df2["Type Water Body"] == selected_wb]
        if selected_year != "ALL": plot_df2 = plot_df2[plot_df2["Year"] == int(selected_year)]
        agg2 = (plot_df2.groupby(["State Name", "Type Water Body"])
                .agg(FC=("Fecal_Coliform", "median"), count=("Fecal_Coliform", "count"))
                .reset_index())
        if not agg2.empty:
            st.divider()
            st.subheader("Full Data Table")
            display = agg2.copy()
            display["Safety"]   = display["FC"].apply(lambda x: get_safety_map(x)[2])
            display["State"]    = display["State Name"].str.title()
            display["Water Body"] = display["Type Water Body"].str.title()
            display["Fecal Coliform (MPN/100ml)"] = display["FC"].round(0).astype(int)
            display = display[["State", "Water Body", "Fecal Coliform (MPN/100ml)", "count", "Safety"]]
            display.columns = ["State", "Water Body", "Fecal Coliform (MPN/100ml)", "Records", "Safety"]
            display = display.sort_values("Fecal Coliform (MPN/100ml)", ascending=False).reset_index(drop=True)
            st.dataframe(display, use_container_width=True, hide_index=True)

    st.caption("Source & Safe threshold: CPCB India & Indian Standards")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: PREDICT CONTAMINATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict Contamination":
    _bg       = {"safe": "#0d2a1a", "mod": "#2a2000", "unsafe": "#2a0d0d", "default": "var(--bg-main)"}
    _bg_light = {"safe": "#d6f5e3", "mod": "#fff8e1", "unsafe": "#fde8e8", "default": "#ddeeff"}
    _key = st.session_state.predict_bg
    st.markdown(f"""
    <style>
    @media (prefers-color-scheme: light) {{
      [data-testid="stAppViewContainer"] > .main,
      [data-testid="stAppViewContainer"],
      [data-testid="stApp"] {{
        background-color: {_bg_light.get(_key,"#ddeeff")} !important;
        transition: background-color 0.6s ease;
      }}
    }}
    @media (prefers-color-scheme: dark) {{
      [data-testid="stAppViewContainer"] > .main,
      [data-testid="stAppViewContainer"],
      [data-testid="stApp"] {{
        background-color: {_bg.get(_key,"#0e1e2e")} !important;
        transition: background-color 0.6s ease;
      }}
    }}
    </style>""", unsafe_allow_html=True)

    st.markdown('<p class="page-title">Predict Fecal Coliform</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Enter water parameters · Extra Tree model · Threshold: 50 MPN/100 mL (CPCB Class A)</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Model Info")
        st.metric("Algorithm",  "Extra Trees")
        st.metric("CV R²",      "~0.3")
        st.metric("Safe limit", "50 MPN/100mL")
        if load_err:
            st.error(f"Model not loaded:\n{load_err}")

    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        st.markdown("#### Water Parameters")
        c1, c2 = st.columns(2)
        with c1:
            temp    = st.number_input("Temperature (°C)",       0.0,  45.0,  28.0, 0.5)
            ph      = st.number_input("pH",                     0.0,  14.0,   7.2, 0.1)
            bod     = st.number_input("BOD (mg/L)",             0.0, 300.0,   3.0, 0.1)
        with c2:
            do      = st.number_input("Dissolved O₂ (mg/L)",   0.0,  20.0,   6.5, 0.1)
            cond    = st.number_input("Conductivity (µS/cm)",   0.0, 50000.0, 420.0, 10.0)
            nitrate = st.number_input("Nitrate/Nitrite (mg/L)", 0.0, 100.0,   1.2, 0.1)
        wbt         = st.selectbox("Water Body Type", ["LAKE", "POND", "TANK", "WETLAND"])
        predict_btn = st.button("Predict Fecal Coliform", type="primary", use_container_width=True)

    with col_right:
        st.markdown("#### Result")
        result_area = st.empty()

        if predict_btn:
            if model is None or qt is None:
                with result_area.container():
                    missing = []
                    if model is None: missing.append("model (best_model.pkl)")
                    if qt is None:    missing.append("quantile transformer (quantile_transformer.pkl)")
                    st.error(f"Artifacts not loaded: {', '.join(missing)}. Check sidebar for details.")
            else:
                try:
                    do_sqrt          = np.sqrt(max(do, 0))
                    conductivity_log = np.log1p(cond)
                    bod_log          = np.log1p(bod)
                    nitrate_log      = np.log1p(nitrate)
                    bod_temp_log     = np.log1p(bod * temp)
                    bod_cond_log     = np.log1p(bod * cond)
                    nitrate_temp     = nitrate * temp
                    is_monsoon       = 0
                    state_enc        = float(np.mean(list(state_freq.values()))) if state_freq else 0.03
                    all_wbt          = ["POND", "TANK", "WETLAND"]
                    wbt_cols         = {f"Water_Body_Type_{cat}": int(wbt == cat) for cat in all_wbt}
                    row = {
                        "Temperature": temp, "DO_sqrt": do_sqrt, "pH": ph,
                        "Conductivity_log": conductivity_log, "BOD_log": bod_log,
                        "Nitrate_Nitrite_log": nitrate_log, "BOD_Temp_log": bod_temp_log,
                        "BOD_Conductivity_log": bod_cond_log, "Nitrate_Temp": nitrate_temp,
                        "Is_Monsoon": is_monsoon, "State_freq": state_enc, **wbt_cols,
                    }
                    X_in = pd.DataFrame([row])
                    for col in feature_cols:
                        if col not in X_in.columns: X_in[col] = 0
                    X_in   = X_in[feature_cols]
                    y_qt   = model.predict(X_in)
                    y_pred = float(qt.inverse_transform(y_qt.reshape(-1, 1)).ravel().clip(0)[0])

                    # Confidence tree ensembles only; XGBoost does not expose .estimators_
                    confidence = None
                    try:
                        tree_preds = np.array([t.predict(X_in)[0] for t in model.estimators_])
                        confidence = float(max(0, min(100, 100 - np.std(tree_preds) * 40)))
                    except AttributeError:
                        pass

                    label, color, is_safe = get_safety_predict(y_pred)
                    box_cls = "result-safe" if is_safe else ("result-mod" if y_pred <= 500 else "result-unsafe")
                    st.session_state.predict_bg = "safe" if is_safe else ("mod" if y_pred <= 500 else "unsafe")

                    chips = [
                        ("Temperature",     f"{temp}°C"),
                        ("Dissolved O₂",    f"{do} mg/L"),
                        ("pH",              f"{ph}"),
                        ("BOD",             f"{bod} mg/L"),
                        ("Conductivity",    f"{cond:.0f}"),
                        ("Nitrate/Nitrite", f"{nitrate} mg/L"),
                    ]
                    chips_html = "".join(f"""
                      <div class="summary-chip">
                        <div class="summary-chip-label">{l}</div>
                        <div class="summary-chip-value">{v}</div>
                      </div>""" for l, v in chips)

                    with result_area.container():
                        st.markdown(f"""
                        <div class="result-box {box_cls}">
                          <div class="result-number">{y_pred:,.0f}</div>
                          <div class="result-unit">MPN / 100 mL</div>
                          <div class="result-badge">{label}</div>
                        </div>""", unsafe_allow_html=True)
                        if confidence is not None:
                            st.markdown(f"""
                            <div class="conf-wrap">
                              <div class="conf-label-row">
                                <span>Model Confidence</span><span><b>{confidence:.0f}%</b></span>
                              </div>
                              <div class="conf-track">
                                <div class="conf-fill" style="width:{confidence}%"></div>
                              </div>
                            </div>""", unsafe_allow_html=True)
                        else:
                            st.caption("Confidence interval not available for XGBoost.")
                        st.markdown(f'<div class="summary-grid">{chips_html}</div>', unsafe_allow_html=True)
                        st.markdown("---")
                        st.markdown("**Top feature importances**")
                        imp    = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
                        imp_df = imp.reset_index()
                        imp_df.columns = ["Feature", "Importance"]
                        imp_df["Feature"] = imp_df["Feature"].str.replace("_", " ")
                        st.bar_chart(imp_df.set_index("Feature"), height=240)

                except Exception as e:
                    with result_area.container():
                        st.error(f"Prediction error: {e}")
        else:
            with result_area.container():
                st.markdown(f"""
                <div class="empty-state">
                  <img src="{GIF['predict_idle']}" class="hero-gif" alt="water">
                  <div class="empty-text">Fill in the parameters on the left and click <b>Predict</b></div>
                  <div class="empty-sub">Safe threshold: 50 MPN/100 mL (CPCB Class A)</div>
                </div>""", unsafe_allow_html=True)
                if df is not None:
                    st.markdown("---")
                    st.markdown("**Historical median Fecal Coliform by Year**")
                    yearly = df.groupby("Year")["Fecal_Coliform"].median().reset_index()
                    yearly.columns = ["Year", "Median FC"]
                    st.line_chart(yearly.set_index("Year"), height=200)

    st.caption("Model: XGBoost · CPCB India 2017–2022")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: FORECAST COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Forecast Comparison":
    st.markdown('<p class="page-title">Predicted vs Real Year by Year</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Compare model predictions against real Fecal Coliform readings, 2018–2023 · 2023 has no ground truth yet</p>', unsafe_allow_html=True)

    FORECAST_CSV = "Dataset/forecast_all_years.csv"
    if not os.path.exists(FORECAST_CSV):
        st.error(f"Forecast file not found at {FORECAST_CSV}. Place forecast_all_years.csv in the data/ folder.")
    else:
        forecast_full = pd.read_csv(FORECAST_CSV)

        agg = forecast_full.groupby(["State_Name", "Year"]).agg(
            Predicted_FC=("Predicted_FC", "first"),
            Real_FC=("Real_FC", "median"),
            n_stations=("Real_FC", "count"),
        ).reset_index()
        agg["Safety"] = agg["Predicted_FC"].apply(lambda x: get_safety_map(x)[2])

        with st.sidebar:
            st.header("Comparison Filters")
            year_opts = sorted(agg["Year"].unique().tolist())
            selected_year = st.selectbox(
                "Year", year_opts,
                index=year_opts.index(2023) if 2023 in year_opts else len(year_opts) - 1,
                key="forecast_year_select",
            )

        year_df  = agg[agg["Year"] == selected_year].copy()
        has_real = year_df["Real_FC"].notna().any()

        col1, col2, col3 = st.columns(3)
        col1.metric("Safe States",     int((year_df["Safety"] == "Safe").sum()))
        col2.metric("Not Safe States", int((year_df["Safety"] != "Safe").sum()))
        if has_real:
            mae_year = (
                year_df["Predicted_FC"] - year_df["Real_FC"]
            ).abs().mean()

            col3.metric(
                f"{selected_year} Avg Error (MAE)",
                f"{mae_year:,.0f}"
            )
        else:
            col3.metric(
                f"{selected_year} Avg Error (MAE)",
                "N/A forecast only"
            )

        if not has_real:
            st.info(
                f"{selected_year} is a pure forecast year, "
            "no real world readings exist yet to compare against."
            )

        st.divider()

        st.subheader("Prediction Error Trend, All Years")
        trend = agg.dropna(subset=["Real_FC"]).groupby("Year").apply(
            lambda g: pd.Series({"MAE": (g["Predicted_FC"] - g["Real_FC"]).abs().mean()})
        ).reset_index()
        st.line_chart(trend.set_index("Year"), height=240)
        st.caption("Mean absolute error between predicted and real Fecal Coliform, by year. "
                   "2023 excluded no real data exists for it yet.")

        st.divider()

        st.subheader(f"Map {selected_year}")
        m_forecast = folium.Map(location=[22.5, 82.0], zoom_start=5,
                                tiles="CartoDB positron", min_zoom=4, max_zoom=12)
        m_forecast.fit_bounds([[6.5, 68.0], [35.5, 97.5]])

        for _, row in year_df.iterrows():
            state = row["State_Name"]
            if state not in STATE_COORDS:
                continue
            lat, lon = STATE_COORDS[state]
            label, color, status = get_safety_map(row["Predicted_FC"])
            bg = {"green": "#d4edda", "orange": "#fff3cd", "red": "#f8d7da"}[color]
            real_line = (f"<tr><td><b>Real FC (median)</b></td><td>{row['Real_FC']:,.0f}</td></tr>"
                         if pd.notna(row["Real_FC"]) else
                         "<tr><td colspan='2'><i>No real data yet</i></td></tr>")
            popup_html = f"""
            <div style="font-family:Arial,sans-serif;min-width:200px;padding:4px">
              <b style="font-size:15px">📍 {state.title()}</b>
              <hr style="margin:4px 0">
              <table style="width:100%;font-size:13px">
                <tr><td><b>Predicted FC</b></td><td>{row['Predicted_FC']:,.0f} MPN/100ml</td></tr>
                {real_line}
              </table>
              <div style="margin-top:8px;padding:6px;border-radius:6px;
                background:{bg};font-weight:bold;font-size:13px;text-align:center">{label}</div>
            </div>"""
            folium.CircleMarker(
                location=[lat, lon], radius=12, color="white", weight=1.5,
                fill=True, fill_color=color, fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=folium.Tooltip(f"<b>{state.title()}</b><br>{selected_year} — {label}", sticky=True),
            ).add_to(m_forecast)

        st_folium(m_forecast, width="100%", height=520, returned_objects=[])

        st.divider()

        if has_real:
            st.subheader(f"Predicted vs Real — {selected_year}")
            compare_df = year_df.dropna(subset=["Real_FC"])[["State_Name", "Predicted_FC", "Real_FC"]].copy()
            compare_df["State_Name"] = compare_df["State_Name"].str.title()
            compare_df = compare_df.sort_values("Real_FC", ascending=False).set_index("State_Name")
            st.bar_chart(compare_df, height=320)
        else:
            st.subheader(f"Predicted Levels — {selected_year}")
            chart_df = year_df[["State_Name", "Predicted_FC"]].copy()
            chart_df["State_Name"] = chart_df["State_Name"].str.title()
            chart_df = chart_df.sort_values("Predicted_FC", ascending=False).set_index("State_Name")
            st.bar_chart(chart_df, height=320)

        st.divider()

        st.subheader("Largest Misses" if has_real else "Highest Predicted States")
        worst = year_df.copy()
        if has_real:
            worst["Abs_Error"] = (worst["Predicted_FC"] - worst["Real_FC"]).abs()
            worst = worst.sort_values("Abs_Error", ascending=False).head(10)
            worst["State_Name"] = worst["State_Name"].str.title()
            show = worst[["State_Name", "Predicted_FC", "Real_FC", "Abs_Error", "Safety"]].rename(columns={
                "State_Name": "State", "Predicted_FC": "Predicted FC",
                "Real_FC": "Real FC (median)", "Abs_Error": "Abs Error",
            })
        else:
            worst = worst.sort_values("Predicted_FC", ascending=False).head(10)
            worst["State_Name"] = worst["State_Name"].str.title()
            show = worst[["State_Name", "Predicted_FC", "Safety"]].rename(columns={
                "State_Name": "State", "Predicted_FC": "Predicted FC",
            })
        st.dataframe(show, use_container_width=True, hide_index=True)

    st.caption("Source: forecast_all_years.csv · Real_FC aggregated as per-state median across monitoring stations.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: LEARN & TIPS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Learn & Tips":
    st.markdown('<p class="page-title">Learn & Tips</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Water facts, safety tips, which water to drink, and how to save water</p>', unsafe_allow_html=True)

    FACTS = [
        ("3.4M",       "people die each year from water-related diseases, one of the leading causes of death globally."),
        ("1 in 3",     "people globally do not have access to safe drinking water at home."),
        ("80%",        "of diseases in developing countries are linked to unsafe water and poor sanitation."),
        ("50 L/day",   "is the minimum recommended by the UN for basic human needs: drinking, cooking, and hygiene."),
        ("~Same",      "amount of water exists on Earth today as millions of years ago, water is constantly recycled."),
        ("1.386B km³", "of water is held on Earth in total across all oceans, ice, rivers, and groundwater."),
        ("2.5%",       "of all Earth's water is freshwater, only 0.3% of that is surface water in rivers and lakes."),
        ("0.014%",     "of all water on Earth is both fresh and easily accessible for direct human use."),
        ("68%",        "of all freshwater is locked in ice sheets and glaciers - the planet's largest freshwater store."),
        ("30%",        "of freshwater is stored as groundwater in aquifers deep below the surface."),
        ("Majority",   "of ice-bound freshwater is held by Antarctica and Greenland, with the rest in glaciers."),
        ("2.1%",       "of all Earth's water is frozen in glaciers and ice caps."),
    ]

    st.markdown("### Water Fact of the Day")
    if "fact_idx" not in st.session_state:
        st.session_state.fact_idx = random.randint(0, len(FACTS) - 1)
    num, text = FACTS[st.session_state.fact_idx]
    st.markdown(f"""
    <div class="fact-card">
      <div class="fact-number">{num}</div>
      <div class="fact-text">{text}</div>
    </div>""", unsafe_allow_html=True)
    if st.button("Next Fact"):
        st.session_state.fact_idx = random.randint(0, len(FACTS) - 1)
        st.rerun()

    st.divider()
    tab1, tab2, tab3 = st.tabs(["Which Water to Drink?", "Safety Tips", "Water Saving"])

    with tab1:
        st.markdown("### Which Water Source is Safest?")
        col_gif, col_txt = st.columns([1, 2])
        with col_gif:
            st.markdown(
                f'<img src="{GIF["water_cycle"]}" style="width:100%;border-radius:12px">',
                unsafe_allow_html=True)
            st.caption("Earth's water cycle")
        with col_txt:
            st.markdown("""
            Water moves continuously through the environment - evaporating from oceans,
            falling as rain, filtering through rock, and flowing into rivers and lakes.
            Contamination at any stage affects drinking water quality.
            Fecal Coliform bacteria are the primary indicator of unsafe water
            in India's CPCB monitoring programme.
            """)
        st.markdown("---")

        water_types = [
            ("src_mountain", "Mountain Spring Water",  "Excellent", "#2e9e6e",
             "Naturally filtered through rock. Very low contamination risk. Best choice if available."),
            ("src_tap",      "Municipal Tap Water",    "Good",      "#3fa8c8",
             "Treated and chlorinated. Safe in most Indian cities. Always boil if unsure."),
            ("src_ro",       "Filtered / RO Water",    "Good",      "#3fa8c8",
             "RO removes dissolved solids, bacteria, and viruses. Reliable for daily use."),
            ("src_bottle",   "Bottled Water",          "Good",      "#7ecab8",
             "Generally safe - check BIS certification (IS 14543). Avoid if seal is broken."),
            ("src_river",    "River Water (untreated)", "Dangerous", "#c94040",
             "High risk of Fecal Coliform, BOD, and heavy metals. Never drink without treatment."),
            ("src_pond",     "Pond / Lake (untreated)", "Dangerous", "#c94040",
             "Stagnant water breeds bacteria. Fecal Coliform often exceeds 5,000 MPN/100ml."),
            ("src_ground",   "Groundwater / Borewells", "Moderate",  "#d48f00",
             "Risk of arsenic, fluoride, and nitrate contamination. Always test before drinking."),
            ("src_rain",     "Rainwater (harvested)",   "Moderate",  "#d48f00",
             "Clean at source but contaminated by rooftops. Filter and disinfect before use."),
        ]
        for gif_key, name, safety, color, desc in water_types:
            st.markdown(f"""
            <div class="src-row" style="border-left:4px solid {color}">
              <img src="{GIF[gif_key]}" alt="{name}">
              <div class="src-body">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
                  <span class="src-name">{name}</span>
                  <span class="src-badge" style="background:{color}">{safety}</span>
                </div>
                <div class="src-desc">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("### Water Safety Tips")
        tips = [
            ("tip_boil",     "Always Boil Suspicious Water",
             "Boil for at least 1 minute to kill bacteria, viruses, and parasites. At high altitudes, boil for 3 minutes."),
            ("tip_test",     "Test Your Water Annually",
             "Home water testing kits detect pH, chlorine, nitrates, and bacteria. Test borewell water every 6 months."),
            ("tip_tank",     "Clean Your Storage Tanks",
             "Water tanks breed bacteria. Clean and disinfect overhead tanks every 3–6 months."),
            ("tip_check",    "Check for Contamination Signs",
             "Discolored water, unusual smell, or strange taste are red flags. Don't drink - report to local authorities."),
            ("tip_store",    "Proper Storage",
             "Store water in covered, food-grade containers. Avoid plastic containers exposed to sunlight."),
            ("tip_tablet",   "Use Purification Tablets",
             "When travelling or in emergencies, chlorine/iodine tablets are effective and inexpensive."),
            ("tip_monsoon",  "Monsoon Extra Caution",
             "Fecal Coliform spikes June–September. Switch to RO or boiled water during monsoon season."),
            ("tip_symptoms", "Know the Symptoms",
             "Diarrhoea, vomiting, cramps, and fever after drinking water = possible contamination. Seek help immediately."),
        ]
        col_a, col_b = st.columns(2)
        for i, (gif_key, title, text) in enumerate(tips):
            with (col_a if i % 2 == 0 else col_b):
                st.markdown(f"""
                <div class="tip-card">
                  <div class="tip-header">
                    <img src="{GIF[gif_key]}" style="width:38px;height:38px;border-radius:8px;object-fit:cover">
                    <span class="tip-title">{title}</span>
                  </div>
                  <div class="tip-text">{text}</div>
                </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown("### Water Conservation Tips")
        st.markdown("#### At Home")
        home_tips = [
            ("tip_monsoon", "Shorter Showers",    "Cutting shower time by 2 minutes saves ~30 litres per shower."),
            ("tip_check",   "Fix Leaks Fast",      "A dripping tap wastes up to 20,000 litres per year."),
            ("src_rain",    "Water Plants Wisely", "Water early morning or evening to reduce evaporation by 50%."),
            ("tip_tank",    "Full Loads Only",     "Run washing machines only when full — saves 30–50 litres per cycle."),
            ("tip_boil",    "Dual-Flush Toilets",  "Switching to dual-flush saves up to 67% of toilet water usage."),
        ]
        for gif_key, title, text in home_tips:
            st.markdown(f"""
            <div class="tip-card" style="border-left:4px solid var(--accent-blue)">
              <div class="tip-header">
                <img src="{GIF[gif_key]}" style="width:38px;height:38px;border-radius:8px;object-fit:cover">
                <span class="tip-title">{title}</span>
              </div>
              <div class="tip-text">{text}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### Most Contaminated States")
        if df is not None:
            state_avg = (df.groupby("State Name")["Fecal_Coliform"]
                         .median().sort_values(ascending=False).head(10).reset_index())
            state_avg.columns = ["State", "Median FC (MPN/100ml)"]
            state_avg["State"] = state_avg["State"].str.title()
            st.markdown("**Top 10 most contaminated states (median Fecal Coliform):**")
            st.bar_chart(state_avg.set_index("State"), height=280)
            st.caption("Higher contamination = more urgent need for water safety measures.")
        else:
            st.info("Dataset not loaded — chart unavailable.")

    st.caption("Sources: WHO, UNICEF, CPCB India, UN Water")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: REPORT POLLUTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Report Pollution":
    st.markdown(
        f'<img src="{GIF["report_hero"]}" style="width:100%;max-height:200px;object-fit:cover;border-radius:14px;margin-bottom:16px">',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="page-title">Report Pollution</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Help your community - report polluted water bodies. Reports appear on the map.</p>', unsafe_allow_html=True)

    is_read_only = not os.access(".", os.W_OK)
    if is_read_only:
        st.warning("Running in read-only mode. Reports visible this session only.")

    col_form, col_map = st.columns([1, 1.2], gap="large")

    with col_form:
        st.markdown("#### Submit a Report")
        with st.form("report_form", clear_on_submit=True):
            reporter = st.text_input("Your Name (optional)", placeholder="Anonymous")
            state_r  = st.selectbox("State", sorted(STATE_COORDS.keys()),
                                    format_func=lambda x: x.title())
            wb_r     = st.selectbox("Water Body Type",
                                    ["River", "Lake", "Pond", "Tank", "Wetland", "Canal", "Other"])
            severity = st.select_slider("Severity", options=[
                "Low — minor discoloration",
                "Medium — bad smell / foam",
                "High — dead fish / visible waste",
                "Critical — chemical spill / industrial discharge",
            ])
            desc  = st.text_area("Describe what you observed",
                placeholder="e.g. Dark oily water near the river bank, strong chemical smell...")
            photo = st.file_uploader("Upload a photo (optional)",
                                     type=["jpg", "jpeg", "png", "webp", "gif"])
            submitted = st.form_submit_button("Submit Report", type="primary", use_container_width=True)

        if submitted:
            if not desc.strip():
                st.error("Please describe what you observed.")
            else:
                lat, lon = STATE_COORDS.get(state_r, (22.5, 82.0))
                lat += random.uniform(-0.3, 0.3)
                lon += random.uniform(-0.3, 0.3)
                photo_path = ""
                if photo:
                    try:
                        fname = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{photo.name}"
                        fpath = os.path.join(UPLOADS_DIR, fname)
                        Image.open(photo).save(fpath)
                        photo_path = fpath
                    except Exception:
                        pass
                record = {
                    "timestamp":   datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "reporter":    reporter.strip() or "Anonymous",
                    "state":       state_r,
                    "water_body":  wb_r,
                    "severity":    severity.split("—")[0].strip(),
                    "description": desc.strip(),
                    "lat":         round(lat, 4),
                    "lon":         round(lon, 4),
                    "photo_path":  photo_path,
                }
                saved = save_report(record)
                if saved:
                    st.success("Report submitted! It will appear on the map.")
                else:
                    if "session_reports" not in st.session_state:
                        st.session_state.session_reports = []
                    st.session_state.session_reports.append(record)
                    st.success("Report submitted for this session.")
                if photo:
                    st.image(photo, caption="Uploaded photo", use_container_width=True)

        reports_df = get_session_reports_df()
        if not reports_df.empty:
            st.divider()
            st.markdown("#### Report Statistics")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Reports",  len(reports_df))
            c2.metric("States Covered", reports_df["state"].nunique())
            high = (reports_df["severity"].str.contains("High|Critical", case=False, na=False)).sum()
            c3.metric("High/Critical",  int(high))

    with col_map:
        st.markdown("#### Pollution Reports Map")
        reports_df = get_session_reports_df()

        m3 = folium.Map(location=[22.5, 82.0], zoom_start=5,
                        tiles="CartoDB positron", min_zoom=4, max_zoom=12)
        m3.fit_bounds([[6.5, 68.0], [35.5, 97.5]])

        SEVERITY_COLORS = {"Low": "green", "Medium": "orange", "High": "red", "Critical": "darkred"}

        if not reports_df.empty:
            for _, r in reports_df.iterrows():
                try:
                    sev   = r["severity"].split("—")[0].strip() if "—" in str(r["severity"]) else str(r["severity"])
                    color = SEVERITY_COLORS.get(sev, "red")
                    folium.Marker(
                        location=[float(r["lat"]), float(r["lon"])],
                        icon=folium.Icon(color=color, icon="warning-sign", prefix="glyphicon"),
                        tooltip=f"{r['water_body']} — {sev} severity",
                        popup=folium.Popup(
                            f"<div style='font-family:Arial;min-width:180px'>"
                            f"<b>{sev} Severity</b><br>"
                            f"<b>State:</b> {str(r['state']).title()}<br>"
                            f"<b>Water body:</b> {r['water_body']}<br>"
                            f"<b>Reported by:</b> {r['reporter']}<br>"
                            f"<b>When:</b> {r['timestamp']}<br><br>"
                            f"<i>{r['description']}</i></div>",
                            max_width=260),
                    ).add_to(m3)
                except Exception:
                    pass
            st.caption(f"Showing {len(reports_df)} report(s). Click markers for details.")
        else:
            st.caption("No reports yet. Be the first to report a pollution issue!")

        st_folium(m3, width="100%", height=520, returned_objects=[])

        if not reports_df.empty:
            st.markdown("#### Recent Reports")
            show = reports_df[["timestamp", "state", "water_body", "severity", "description", "reporter"]].copy()
            show.columns = ["Time", "State", "Water Body", "Severity", "Description", "Reporter"]
            show["State"] = show["State"].str.title()
            show = show.sort_values("Time", ascending=False).head(10).reset_index(drop=True)
            st.dataframe(show, use_container_width=True, hide_index=True)

    st.caption("Reports saved locally in data/reports.csv · Photos in uploads/")