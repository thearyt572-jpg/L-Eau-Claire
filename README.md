# 💧 AquaSense India — Water Quality App

Two-page Streamlit app: interactive map + ML prediction.

---

## Folder Structure

```
water_app/
├── app.py                         ← Single Streamlit app (both pages)
├── dataset 2017-2022.csv          ← Raw data  ← YOU PROVIDE
├── model/
│   └── artifacts/
│       ├── finalbest_model_tuned.pkl   ← from your ML notebook
│       ├── finalfeature_cols.pkl       ← from your ML notebook
│       ├── qt.pkl                      ← from your ML notebook
│       └── state_freq_mapping.pkl      ← generate below
├── requirements.txt
└── README.md
```

---

## Step 1 — Save state frequency map (run once in your ML notebook)
```python
import joblib, os
os.makedirs("model/artifacts", exist_ok=True)

state_freq = df['State_Name'].value_counts(normalize=True).to_dict()
joblib.dump(state_freq, 'model/artifacts/state_freq_mapping.pkl')
print("Saved!")
```

## Step 2 — Copy your pkl files into model/artifacts/
```
finalbest_model_tuned.pkl  →  model/artifacts/finalbest_model_tuned.pkl
finalfeature_cols.pkl      →  model/artifacts/finalfeature_cols.pkl
qt.pkl                     →  model/artifacts/qt.pkl
```

## Step 3 — Install & run
```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

---

## Pages

| Page | What it does |
|------|-------------|
| 🗺️ Water Quality Map | Interactive map — colored dots per state/water body, filters by year & type |
| 🔮 Predict Contamination | Enter 6 parameters + state + water body → predicted Fecal Coliform + safe/unsafe badge + confidence + feature importance + mini map |

---

## Safety Thresholds

| Page | Threshold | Standard |
|------|-----------|----------|
| Map  | Safe ≤ 500 MPN/100ml | WHO / CPCB general |
| Predict | Safe ≤ 50 MPN/100ml | CPCB Class A drinking water |
