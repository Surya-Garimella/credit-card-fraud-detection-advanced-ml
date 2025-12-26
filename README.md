# 💳 Credit Card Fraud Detection using Advanced Machine Learning

## 📌 Project Overview
This project implements a **production-style credit card fraud detection system** that tackles real-world machine learning challenges such as **extreme class imbalance, proper evaluation metrics, decision threshold tuning, explainable AI, and anomaly detection**.

The goal is to **maximize fraud detection (recall)** while keeping **false positives under control**, similar to real fintech fraud systems.

---

## 🚀 Key Features
- Handles **highly imbalanced data** (fraud < 1%)
- Uses **Precision, Recall, F1-score, ROC-AUC** instead of accuracy
- Applies **SMOTE** and **class-weighted learning**
- Performs **decision threshold tuning** for business optimization
- Explains predictions using **SHAP**
- Includes **unsupervised anomaly detection** using Isolation Forest
- Modular, production-style **Python codebase**

---

## 🧠 Models Used

### 🔹 Supervised Learning
- **Random Forest Classifier**
  - Class-weighted training
  - SMOTE oversampling
  - Threshold tuning

### 🔹 Unsupervised Learning
- **Isolation Forest**
  - Detects anomalous transactions without labels
  - Acts as a backup fraud detection mechanism

---

## ⚙️ Tech Stack
- Python
- NumPy
- Pandas
- Scikit-learn
- Imbalanced-learn (SMOTE)
- SHAP
- Matplotlib

---

## 📊 Evaluation Strategy
Due to extreme class imbalance, **accuracy is avoided**.

Models are evaluated using:
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

---

## 🔍 Explainability
- Used **SHAP summary plots** to interpret why transactions are flagged as fraud.
- Identified top transaction features influencing fraud predictions.
- Ensured model transparency for business and compliance use cases.

---

## 🚨 Anomaly Detection
- Implemented **Isolation Forest** as an unsupervised model.
- Useful when fraud labels are delayed or unavailable.
- Trades precision for broader anomaly coverage.



## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py
