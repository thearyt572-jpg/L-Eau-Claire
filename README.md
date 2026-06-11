# 💧 L'Eau Claire Water Quality Dashboard

An interactive Streamlit web app for exploring, predicting, and reporting water quality across Indian states (2017–2022), built on CPCB data with an Extra Trees machine learning model.

---

## Folder Structure

```text
mini_water_predict/
├── app.py
├── requirements.txt
├── README.md
├── dataset 2017-2022.csv
├── data/
│   └── reports.csv
├── uploads/
└── model/
    └── artifacts/
        ├── finalbest_model_tuned.pkl
        ├── finalfeature_cols.pkl
        ├── qt.pkl
        └── state_freq_mapping.pkl
```

---

## Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Place Model Files

Copy the following files into `model/artifacts/`:

```text
finalbest_model_tuned.pkl
finalfeature_cols.pkl
qt.pkl
state_freq_mapping.pkl
```

### Generate State Frequency Mapping

Run once before encoding `State Name`:

```python
import joblib
import os

os.makedirs("model/artifacts", exist_ok=True)

state_freq = df["State Name"].value_counts(normalize=True).to_dict()
joblib.dump(state_freq, "model/artifacts/state_freq_mapping.pkl")
```

### Run the Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## Pages

### 🗺️ Water Quality Map

* Interactive India map
* Color-coded contamination markers
* Filter by year and water body type
* Displays community pollution reports

### 🔮 Predict Contamination

* Predicts Fecal Coliform levels
* Water safety classification
* Confidence score visualization
* Feature importance analysis

### 💡 Learn & Tips

* Water quality facts
* Safe drinking water guidance
* Water conservation recommendations
* State-wise contamination insights

### 🚨 Report Pollution

* Submit pollution incidents
* Optional photo upload
* Stores reports locally
* Reports appear on the map

---

## Model Details

| Item             | Detail                      |
| ---------------- | --------------------------- |
| Algorithm        | Extra Trees Regressor       |
| Target           | Fecal Coliform (MPN/100 mL) |
| Target Transform | QuantileTransformer         |
| CV Strategy      | 5-Fold Cross Validation     |
| CV R²            | Approximately 0.30          |
| Dataset          | CPCB India 2017–2022        |

### Features

```text
Temperature
DO_sqrt
pH
Conductivity_log
BOD_log
Nitrate_Nitrite_log
BOD_Temp_log
BOD_Conductivity_log
Nitrate_Temp
Is_Monsoon
State_freq
Water_Body_Type_POND
Water_Body_Type_TANK
Water_Body_Type_WETLAND
```

---

## Safety Thresholds

| Module     | Threshold             |
| ---------- | --------------------- |
| Map        | Safe ≤ 500 MPN/100 mL |
| Prediction | Safe ≤ 50 MPN/100 mL  |

---

## Tech Stack

* Streamlit
* Folium
* streamlit-folium
* scikit-learn
* joblib
* pandas
* numpy
* Pillow

---

## Data Source

Central Pollution Control Board (CPCB), India

Water quality monitoring data (2017–2022) including:

* Temperature
* Dissolved Oxygen (DO)
* pH
* Conductivity
* Biological Oxygen Demand (BOD)
* Nitrate/Nitrite
* Fecal Coliform
* Total Coliform
