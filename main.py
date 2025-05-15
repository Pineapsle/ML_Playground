# Main script to run models and experiments

import numpy as np
import pandas as pd
from math import sqrt
from models.linear_regression import LinearRegression
from utils.preprocess import load_data, scale_features, train_test_split
from utils.metrics import mean_squared_error, accuracy, r2_score

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

    # Initialize and train the model
    model1 = LinearRegression(learning_rate=0.01, n_iters=5500)
    model1.fit(X_train, y_train)

    # Make predictions and evaluate
    predictions = model1.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    # Print the results
    print(f"Mean Squared Error: {mse:.5f}")
    print(f"Root Mean Squared Error: {sqrt(mse):.5f}")
    print(f"R2 Score: {r2_score(y_test, predictions):.5f}")
    print(f"R Score: {sqrt(r2_score(y_test, predictions)):.5f}")

if __name__ == "__main__":
    main()