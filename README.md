# Predictive Maintenance & Visual Inspection

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-2.x-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-0091BD?style=flat-square)](https://xgboost.ai/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.x-40B0A6?style=flat-square)](https://lightgbm.readthedocs.io/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-22B8CF?style=flat-square)](https://docs.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agents-1C3C3C?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)

Fusing **sensor telemetry and computer vision into a single maintenance decision** for
telecom infrastructure — failure-risk prediction, defect detection, and multi-agent
prioritization of work orders, all surfaced in a Streamlit dashboard.

## How it works

- **Numeric pipeline** — an XGBoost + LightGBM ensemble (GroupKFold cross-validation)
  predicts 14-day failure risk from sensor telemetry (13,500 readings · 150 sites)
- **Vision pipeline** — a fine-tuned YOLOv8 model detects defects in site imagery, with
  explicit overfitting detection and automatic model escalation (n → m)
- **Fusion** — a LangGraph multi-agent system fuses both signals into prioritized,
  cost-aware work orders
- **Dashboard** — a Streamlit app visualizes risk scores, predictions, and work orders

## Results

| Stage | Metric | Achieved | Target |
|---|---|---|---|
| Numeric | Site-level failure recall | **18 / 18 = 100%** | ≥ 80% ✅ |
| Numeric | OOF ROC-AUC / PR-AUC | **0.9996 / 0.9674** | — |
| Numeric | Failure recall (best threshold) | 0.996 | — |
| Fusion | HIGH-priority work-order precision | **100%** | ≥ 75% ✅ |
| Fusion | Downtime prevented / action cost | 738 hours / $143.5k | — |

## Repository structure

```
predictive-maintenance-inspection/
├── Predictive_Maintenance_Inspection.ipynb   # end-to-end pipeline (19 cells)
├── streamlit_app.py                          # operations dashboard
├── streamlit_data/                           # model outputs & results consumed by the app
└── figures/                                  # charts from the runs
    ├── 01_sensor_telemetry.png               ├── 05_yolov8m_confusion_matrix.png
    ├── 02_top_risk_sites.png                 ├── 06_yolov8m_pr_curve.png
    ├── 03_training_curves.png                └── 07_yolov8m_val_predictions.jpg
    └── 04_yolov8m_training_results.png
```

## Getting started

```bash
# notebook (GPU recommended for the vision training section)
# open Predictive_Maintenance_Inspection.ipynb in Google Colab and run top to bottom —
# data downloads automatically from Google Drive on first run.

# dashboard
pip install -r requirements.txt   # streamlit, pandas, plotly, altair
streamlit run streamlit_app.py
```

**Notes**
- The vision section needs a Roboflow API key — set `ROBOFLOW_API_KEY` as an environment
  variable (or enter it when prompted); the notebook downloads the defect dataset via
  the Roboflow API.
- The notebook was trained on a Colab GPU; the numeric and fusion sections run on CPU fine.
