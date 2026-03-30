# ❤️ CardioSense — Heart Disease Risk Predictor

> A machine learning web application that predicts heart disease risk using clinical health data.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🎯 Objective

Build a binary classification model to predict whether a person is at risk of heart disease based on their health data using the **Heart Disease UCI Dataset**.

---

<img width="1815" height="847" alt="h" src="https://github.com/user-attachments/assets/0fc450df-d593-4571-b1f2-28dd73758e0e" />
<img width="1866" height="920" alt="predict" src="https://github.com/user-attachments/assets/2c5a2e7a-ff36-4ed6-9128-251855973433" />


## 📊 Dataset

- **Source:** Heart Disease UCI Dataset (Kaggle / UCI Repository)
- **Records:** 303 patients
- **Features:** 13 clinical features
- **Target:** Binary (0 = No Disease, 1 = Disease)

---

## 🧠 Models

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~85% | ~91% |
| Decision Tree | ~82% | ~86% |

---


<img width="1872" height="918" alt="performance3" src="https://github.com/user-attachments/assets/a1ca59e1-a3f7-46d6-8061-154788e09a75" />

<img width="1870" height="919" alt="about5" src="https://github.com/user-attachments/assets/b7300dc0-b2e2-4dba-8b5e-14d4e4a7d253" />

## 🗂️ Project Structure

```
heart-disease-predictor/
├── backend/
│   ├── model.py          # ML training, EDA, prediction
│   ├── app.py            # FastAPI REST API
│   └── requirements.txt
├── frontend/
│   └── index.html        # Complete web UI
├── data/
│   └── heart.csv         # Dataset (download from Kaggle)
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/heart-disease-predictor
cd heart-disease-predictor
```

### 2. Install dependencies
```bash
pip install -r backend/requirements.txt
```

### 3. Download dataset
- Go to [Kaggle Heart Disease UCI](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)
- Download `heart.csv` → save to `data/heart.csv`

### 4. Train the model
```bash
cd backend
python model.py
```

### 5. Start API server
```bash
uvicorn app:app --reload
# API runs at http://localhost:8000
```

### 6. Open the web UI
Open `frontend/index.html` in your browser.

---

## 📐 Features Covered

- ✅ Data cleaning (missing values, type conversion)
- ✅ Exploratory Data Analysis (EDA)
- ✅ Logistic Regression classification
- ✅ Decision Tree classification
- ✅ Accuracy evaluation
- ✅ ROC curve + AUC score
- ✅ Confusion matrix
- ✅ Feature importance analysis
- ✅ Interactive web UI with live prediction

---

## 📸 Web UI Pages

| Page | Description |
|---|---|
| **Predict** | Enter patient data → get instant risk prediction |
| **Metrics** | ROC curve, confusion matrix, feature importance |
| **EDA** | Dataset statistics and visualizations |
| **About** | Project documentation and setup guide |

---

## 📦 Tech Stack

- **ML:** Python, scikit-learn, pandas, numpy
- **API:** FastAPI, uvicorn
- **Frontend:** HTML, CSS, JavaScript, Chart.js
- **Deployment:** GitHub Pages (frontend) / any Python host (backend)

---

## ⚠️ Disclaimer

This tool is for educational purposes only. Not a substitute for professional medical advice.

---

## 📄 License

MIT License — free to use and modify.
