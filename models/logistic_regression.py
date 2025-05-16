# Logistic regression model from scratch

import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
        # Sigmoid function Forumla
        #          1
        # σ(x)= ________
        #       1+e^(-x)

    def fit(self, X, y):
        # Implement fitting algorithm
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))      # Gradient with respect to weights
            db = (1 / n_samples) * np.sum(y_predicted - y)             # Gradient with respect to bias

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        # Return predictions
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_class)

    def predict_proba(self, X):
        # Return probabilities
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted_proba = self.sigmoid(linear_model)
        return y_predicted_proba