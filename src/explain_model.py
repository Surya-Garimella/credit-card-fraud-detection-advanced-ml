import shap
import matplotlib.pyplot as plt
import numpy as np

def explain_with_shap(model, X_test):
    """
    Robust SHAP explainability for Random Forest
    Works across SHAP versions
    """

    print("\nGenerating SHAP summary plot (sampled data)...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        # Binary classification → take fraud class
        shap_vals = shap_values[1]
    else:
        # Newer SHAP versions may return 3D array
        shap_vals = shap_values[:, :, 1]

    shap.summary_plot(
        shap_vals,
        X_test,
        show=False
    )

    plt.tight_layout()
    plt.show()
