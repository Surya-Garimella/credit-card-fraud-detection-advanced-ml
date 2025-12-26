from src.data_loading import load_data
from src.preprocessing import split_data, scale_features
from src.train_model import train_random_forest
from src.evaluate_model import evaluate_model
from src.imbalance import apply_smote
from src.threshold_tuning import tune_threshold
from src.explain_model import explain_with_shap
from src.anomaly_model import train_isolation_forest

DATA_PATH = "data/raw/creditcard.csv"

def main():
    # 1. Load dataset
    df = load_data(DATA_PATH)
    print("Dataset Loaded:", df.shape)

    # 2. Train-test split
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. Feature scaling
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # 4. Handle imbalance using SMOTE
    print("\nApplying SMOTE...")
    X_train_smote, y_train_smote = apply_smote(X_train_scaled, y_train)

    # 5. Train supervised model
    model = train_random_forest(X_train_smote, y_train_smote)
    print("Random Forest trained with SMOTE")

    # 6. Default threshold evaluation
    print("\nDefault Threshold (0.5) Evaluation")
    evaluate_model(model, X_test_scaled, y_test)

    # 7. Threshold tuning
    print("\nThreshold Tuning Results")
    tune_threshold(model, X_test_scaled, y_test)

    # 8. SHAP explainability (LIMITED samples – VERY IMPORTANT)
    print("\nRunning SHAP Explainability...")
    X_test_shap = X_test_scaled[:1000]  # limit samples for speed
    explain_with_shap(model, X_test_shap)

    # 9. Isolation Forest (unsupervised anomaly detection)
    print("\nTraining Isolation Forest (Anomaly Detection)")
    train_isolation_forest(X_train_scaled, X_test_scaled, y_test)


if __name__ == "__main__":
    main()
