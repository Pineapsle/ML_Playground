# Neural Network model from scratch

import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    # Activation function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, a):
        return a * (1 - a)

    # Forward propagation/pass 
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1      # Linear Combination
        self.A1 = self.sigmoid(self.Z1)             # Activation 
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    # Loss Computation
    def compute_loss(self, Y, A2):
        m = Y.shape[0]
        loss = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
        return loss             

    # Backward propagation
    def backward(self, X, Y, A2):
        assert Y.shape == A2.shape, f"Shape mismatch: Y {Y.shape}, A2 {A2.shape}"
        m = X.shape[0]  # number of examples

        dZ2 = A2 - Y                                # shape (m, 1)
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)      # shape (hidden_size, 1)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)  # shape (1, 1)

        dA1 = np.dot(dZ2, self.W2.T)                # shape (m, hidden_size)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)            # shape (input_size, hidden_size)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        # Update weights and biases
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        return dW1, db1, dW2, db2

    # Training the model
    def train(self, X, Y, iterations=1000, learning_rate=0.01):
        self.learning_rate = learning_rate
        Y = Y.reshape(-1, 1)  # Ensure Y is a column vector
        for i in range(iterations):
            A2 = self.forward(X)
            loss = self.compute_loss(Y, A2)
            self.backward(X, Y, A2)
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}")

    # Making Predictions
    def predict(self, X):
        A2 = self.forward(X)
        predictions = (A2 > 0.5).astype(int)
        return predictions
