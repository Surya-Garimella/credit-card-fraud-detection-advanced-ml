from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

def train_isolation_forest(X_train, X_test, y_test):
    """
    Trains Isolation Forest and evaluates anomaly detection
    """
    iso_model = IsolationForest(
        n_estimators=100,
        contamination=0.0017,  # approx fraud ratio
        random_state=42
    )

    iso_model.fit(X_train)

    # Isolation Forest output: -1 = anomaly, 1 = normal
    y_pred = iso_model.predict(X_test)
    y_pred = [1 if x == -1 else 0 for x in y_pred]

    print("\nIsolation Forest Evaluation")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
