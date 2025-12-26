# Credit Card Fraud Detection using Advanced Machine Learning

**Project Overview**

This project implements a production-style credit card fraud detection system that addresses real-world machine learning challenges such as extreme class imbalance, robust evaluation strategies, decision threshold optimization, model explainability, and anomaly detection.

The primary objective is to maximize fraud detection (recall) while keeping false positives at an acceptable level, reflecting how fraud detection systems are designed in real fintech environments.

---

**Key Features**

- Handles highly imbalanced transaction data (fraud cases < 1%)
- Uses Precision, Recall, F1-score, and ROC-AUC instead of accuracy
- Applies SMOTE and class-weighted learning for imbalance handling
- Performs decision threshold tuning to optimize business trade-offs
- Explains model predictions using SHAP
- Includes unsupervised anomaly detection using Isolation Forest
- Modular, production-style Python codebase

---

**Models Used**

**Supervised Learning**
- Random Forest Classifier
  - Class-weighted training
  - SMOTE oversampling
  - Decision threshold tuning

**Unsupervised Learning**
- Isolation Forest
  - Detects anomalous transactions without using labels
  - Acts as a backup fraud detection mechanism when labels are unavailable

---

**Tech Stack**

- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- SHAP  
- Matplotlib  

---

**Evaluation Strategy**

Due to the extreme class imbalance, accuracy is avoided as a primary metric.

Models are evaluated using:
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

---

**Explainability**

- SHAP summary plots are used to explain why transactions are classified as fraudulent.
- Key transaction features influencing fraud predictions are identified.
- Model transparency is ensured for business and compliance requirements.

---

**Anomaly Detection**

- An Isolation Forest model is implemented for unsupervised anomaly detection.
- Useful in scenarios where fraud labels are delayed or unavailable.
- Trades precision for broader anomaly coverage.

---

**How to Run**

```bash
pip install -r requirements.txt
python main.py
