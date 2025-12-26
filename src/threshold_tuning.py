import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def tune_threshold(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Threshold | Precision | Recall | F1-score")
    print("-------------------------------------------")

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        y_pred = (y_prob >= threshold).astype(int)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"{threshold:.1f}      | {precision:.2f}     | {recall:.2f}  | {f1:.2f}")
