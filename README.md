# Chronic Kidney Disease (CKD) Ensemble Risk Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-blue)](https://xgboost.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An ensemble predictive healthcare machine learning system designed to detect and stratify **Chronic Kidney Disease (CKD)** from routine clinical urinalysis and blood biomarker profiles. Built using a soft-voting ensemble of **Random Forest** and **XGBoost (Extreme Gradient Boosting)** classifiers with automated feature selection.

---

## 🏛️ Ensemble Architecture

```mermaid
flowchart TD
    subgraph RawData ["1. Clinical Input"]
        RawCSV["Patient Urinalysis & Serum Blood Panel"]
    end

    subgraph DataEngineering ["2. Feature Pipeline"]
        Impute["Missing Value KNN / Median Imputation"]
        Enc["Label Encoding of Categorical Urinalysis Variables"]
        Scale["StandardScaler Normalization"]
        Select["Variance & Mutual Information Feature Selection"]
    end

    subgraph EnsembleCore ["3. Hybrid Ensemble Model (VotingClassifier)"]
        direction LR
        RF["Random Forest Estimators (Bagging)"]
        XGB["XGBoost Estimators (Gradient Boosting)"]
        Vote{"Soft Probability Voting Engine"}
        RF --> Vote
        XGB --> Vote
    end

    subgraph DiagnosticOutput ["4. Inference & Clinical Dashboard"]
        Prob["CKD Probability Score (%)"]
        Class["Stage Stratification: CKD vs Normal"]
        UI["Streamlit Interactive Diagnostic Dashboard"]
    end

    RawData --> DataEngineering
    DataEngineering --> EnsembleCore
    Vote --> Prob
    Vote --> Class
    Prob --> UI
    Class --> UI
```

---

## 🧪 Biomarkers & Selected Clinical Indicators

The model isolates the highest-impact physiological biomarkers from the UCI Machine Learning Chronic Kidney Disease dataset:

| Clinical Feature | Marker | Normal Reference Range | Diagnostic Relevance |
| :--- | :--- | :--- | :--- |
| **Serum Creatinine (`sc`)** | Blood | $0.6 - 1.2 \text{ mg/dL}$ | Primary indicator of glomerular filtration efficiency |
| **Blood Urea (`bu`)** | Blood | $10 - 20 \text{ mg/dL}$ | Clearance of nitrogenous waste products |
| **Hemoglobin (`hemo`)** | Blood | $13.5 - 17.5 \text{ g/dL}$ | Renal erythropoietin deficiency & anemia |
| **Specific Gravity (`sg`)** | Urine | $1.010 - 1.025$ | Urine concentration and tubular reabsorption capacity |
| **Albumin (`al`)** | Urine | $0 \text{ (Negative)}$ | Proteinuria / microalbuminuria glomerular barrier breakdown |
| **Hypertension (`htn`)** | Vitals | Normal ($< 120/80$) | Bidirectional etiology: renal vascular damage |
| **Diabetes Mellitus (`dm`)** | Metabolic | Normal fasting glucose | Leading cause of diabetic nephropathy |

---

## 📂 Repository Structure

```text
kidney-disease-prediction-ml/
└── kidney prediction/
    ├── ckd.csv                 # Cleaned clinical pathology dataset
    ├── ensemble_model.pkl      # Trained VotingClassifier (RandomForest + XGBoost)
    ├── label_encoders.pkl      # Categorical encoders for urinalysis flags
    ├── scaler.pkl              # StandardScaler normalization artifact
    ├── selected_features.pkl   # Serialized optimal feature subset
    ├── model.py                # Streamlit clinical dashboard & inference interface
    └── train.py                # End-to-end training, validation & serialization pipeline
```

---

## 🚀 Execution & Setup

### 1. Installation
```bash
git clone https://github.com/ak-bharadwaj/kidney-disease-prediction-ml.git
cd "kidney-disease-prediction-ml/kidney prediction"

python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install streamlit pandas numpy scikit-learn xgboost joblib matplotlib
```

### 2. Retraining & Evaluation (Optional)
```bash
python train.py
```
Outputs model accuracy, ROC-AUC curves, and classification metrics.

### 3. Launch Clinical UI
```bash
streamlit run model.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.
