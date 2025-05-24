# Main script to run models and experiments

import numpy as np
import pandas as pd
from math import sqrt
from models.linear_regression import LinearRegression
from models.logistic_regression import LogisticRegression
from models.decision_tree import DecisionTreeClassifier, DecisionTreeNode
from models.neural_network import NeuralNetwork
from utils.preprocess import load_data, scale_features, train_test_split
from utils.metrics import mean_squared_error, accuracy, r2_score, confusion_matrix, precision_recall_f1
from utils.metrics import evaluate_classification_model, binarize_target
from utils.visualization import plot_linear_regression, plot_logistic_regression


# Linear regression model 
def get_linear_regression_model(X_train, X_test, y_train, y_test):
    # Initialize and train the Linear Regression model
    model1 = LinearRegression(learning_rate=0.01, n_iters=5500)
    model1.fit(X_train, y_train)

    # Make predictions and evaluate
    predictions = model1.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    # Print the results
    print(" \n------ Linear Regression Model ------ ")
    print(f"Mean Squared Error: {mse:.5f}")
    print(f"Root Mean Squared Error: {sqrt(mse):.5f}")
    print(f"R2 Score: {r2_score(y_test, predictions):.5f}")
    print(f"R Score: {sqrt(r2_score(y_test, predictions)):.5f}")

    return predictions


def get_logistic_regression_model(X_train, X_test, y_train, y_test):
    # Convert target to binary: 1 if above median price, else 0
    y_train_binary, y_test_binary = binarize_target(y_train, y_test)

    # Train the Logistic Regression model
    model2 = LogisticRegression(learning_rate=0.01, n_iters=5500)
    model2.fit(X_train, y_train_binary)

    # Make predictions and evaluate
    predictions = model2.predict(X_test)

    # Print the results
    evaluate_classification_model(y_test_binary, predictions, model_name="Logistic Regression")

    return predictions, y_test_binary, model2


def get_decision_tree_model(X_train, X_test, y_train, y_test):
    # Convert target to binary: 1 if above median price, else 0
    y_train_binary, y_test_binary = binarize_target(y_train, y_test)
    y_train_binary = y_train_binary.reshape(-1, 1)
    y_test_binary = y_test_binary.reshape(-1, 1)
     
    # Train the Decision Tree model
    model3 = DecisionTreeClassifier(max_depth=5, min_samples_split=10)
    model3.fit(X_train, y_train_binary)

    # Make predictions and evaluate
    predictions = model3.predict(X_test)

    # Print the results
    evaluate_classification_model(y_test_binary, predictions, model_name="Decision Tree")

    return predictions, y_test_binary, model3

def get_neural_network_model(X_train, X_test, y_train, y_test):
    # Convert target to binary: 1 if above median price, else 0
    y_train_binary, y_test_binary = binarize_target(y_train, y_test)

    input_size = X_train.shape[1]
    hidden_size = 10
    output_size = 1

    model4 = NeuralNetwork(input_size, hidden_size, output_size)
    model4.train(X_train, y_train_binary, iterations=1000, learning_rate=0.01)

    predictions = model4.predict(X_test)

    # Print the results
    evaluate_classification_model(y_test_binary, predictions, model_name="Neural Network")

    return predictions.flatten(), y_test_binary.flatten(), model4


def main():
    # Load and preprocess data
    df = load_data('D:/Projects/ML_Lab/data/processed/train_cleaned.csv')

    # Drop rows with missing values 
    df.dropna(inplace=True)

    # Split into features and target
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"].values

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number]).values

    # Scale features
    X = scale_features(X)

    # Train-test split the model using my own function
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get the models and run the experiments 
    preds_lin_reg = get_linear_regression_model(X_train, X_test, y_train, y_test)
    preds_log_reg, y_test_binary, model_log_reg = get_logistic_regression_model(X_train, X_test, y_train, y_test)
    preds_decision_tree, y_test_binary, model_decision_tree = get_decision_tree_model(X_train, X_test, y_train, y_test)
    preds_nn, y_test_binary_nn, model_nn = get_neural_network_model(X_train, X_test, y_train, y_test)

    # Graph the results
    plot_linear_regression(y_test, preds_lin_reg, title="Linear Regression Predictions vs Actual")
    plot_logistic_regression(y_test_binary, preds_log_reg, X_test, model_log_reg, feature_index=0, title1="Logistic Regression Predictions vs Actual with Sigmoid Curve", title2="Predicted Probabilities vs Feature 0 Values")


    print("\n ------ ALL MODELS SUCCEEDED ------ ")
    print(" \n------ END OF THE PROGRAM ------ ")

if __name__ == "__main__":
    main()


    # Maybe try to separate the print statements from the model classes
    # and put them in the main function