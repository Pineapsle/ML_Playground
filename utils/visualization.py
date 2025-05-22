import matplotlib.pyplot as plt
import numpy as np

def plot_linear_regression(y_test, predictions, title="Linear Regression Predictions vs Actual"):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, color='blue', label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(title)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid()
    plt.show()

def plot_logistic_regression(y_test, predictions, X, model, feature_index=0,
                             title1="Logistic Regression Predictions vs Actual",
                             title2="Sigmoid Curve of Logistic Regression"):

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # --- First plot ---
    proba_test = model.predict_proba(X)
    predicted_probs = proba_test[:, 1] if proba_test.ndim == 2 else proba_test
    sorted_idx = np.argsort(predicted_probs)

    def jitter(arr, amount=0.05):
        return arr + np.random.uniform(-amount, amount, size=arr.shape)

    axs[0].scatter(range(len(y_test)), jitter(y_test[sorted_idx]), color='green', label='Actual Class', alpha=0.6)
    axs[0].scatter(range(len(predictions)), jitter(predictions[sorted_idx]), color='blue', label='Predicted Class', alpha=0.6)
    axs[0].set_title(title1)
    axs[0].set_xlabel('Samples sorted by predicted probability')
    axs[0].set_ylabel('Class')
    axs[0].set_yticks([0, 1])
    axs[0].legend()
    axs[0].grid(True)

    # Plot sigmoid curve for the first graph
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    x_sig = np.linspace(-10, 10, 200)
    axs[0].plot(np.linspace(0, len(y_test)-1, 200), sigmoid(x_sig), color='orange', label='Sigmoid Curve', alpha=0.7)
    axs[0].legend()

    # --- Second plot ---
    x_feature = X[:, feature_index]
    x_values = np.linspace(x_feature.min(), x_feature.max(), 300).reshape(-1, 1)

    if X.shape[1] > 1:
        mean_features = np.mean(X, axis=0)
        X_input = np.tile(mean_features, (x_values.shape[0], 1))
        X_input[:, feature_index] = x_values.flatten()
    else:
        X_input = x_values

    proba_output = model.predict_proba(X_input)
    y_proba = proba_output[:, 1] if proba_output.ndim == 2 else proba_output

    axs[1].scatter(x_feature, predicted_probs, color='blue', alpha=0.5, label='Predicted Probabilities')
    axs[1].set_title(title2)
    axs[1].set_xlabel(f'Feature {feature_index} values')
    axs[1].set_ylabel('Predicted Probability')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
