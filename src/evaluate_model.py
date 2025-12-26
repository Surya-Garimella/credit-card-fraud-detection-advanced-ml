from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_prob)
    print("ROC-AUC Score:", roc_auc)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
