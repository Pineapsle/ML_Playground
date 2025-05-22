import matplotlib.pyplot as plt
import numpy as np

def plot_logistic_combined(y_test, predictions, X, model, feature_index=0, 
                           title1="Logistic Regression Predictions vs Actual",
                           title2="Sigmoid Curve of Logistic Regression"):

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # 2 rows, 1 col

    # First plot: Actual vs Predicted Classes scatter
    axs[0].scatter(range(len(y_test)), y_test, color='green', label='Actual Class', alpha=0.6)
    axs[0].scatter(range(len(predictions)), predictions, color='blue', label='Predicted Class', alpha=0.6)
    axs[0].set_title(title1)
    axs[0].set_xlabel('Sample index')
    axs[0].set_ylabel('Class')
    axs[0].set_yticks([0, 1])
    axs[0].legend()
    axs[0].grid(True)

    # Second plot: Sigmoid curve of predicted probabilities vs one feature
    x_feature = X[:, feature_index]
    x_values = np.linspace(x_feature.min(), x_feature.max(), 300).reshape(-1, 1)

    if X.shape[1] > 1:
        mean_features = np.mean(X, axis=0)
        X_input = np.tile(mean_features, (x_values.shape[0], 1))
        X_input[:, feature_index] = x_values.flatten()
    else:
        X_input = x_values

    y_proba = model.predict_proba(X_input)

    axs[1].plot(x_values, y_proba, color='red', label='Predicted Probability')
    axs[1].set_title(title2)
    axs[1].set_xlabel(f'Feature {feature_index} values')
    axs[1].set_ylabel('Predicted Probability')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()