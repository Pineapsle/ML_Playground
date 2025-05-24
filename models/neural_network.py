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

    # Loss Computation (Measures the difference between predicted and actual values)
    # Binary Cross-Entropy Loss (Classification Tasks)

    def compute_loss(self, Y, A2):
        m = Y.shape[0]
        loss = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
        return loss             

    # Backward propagation

    def backward(self, X, Y, A2):
        m = X.shape[0]
        dZ2 = A2 - Y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    # Training the model

    def train(self, X, Y, iterations=1000, learning_rate=0.01):
        self.learning_rate = learning_rate
        for i in range(iterations):
            A2 = self.forward(X)
            loss = self.compute_loss(Y, A2)
            self.backward(X, Y, A2)
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss}")
    
    # Making Predictions

    def predict(self, X):
        A2 = self.forward(X)
        predictions = (A2 > 0.5).astype(int)
        return predictions                              # Applies threshold of 0.5 to classify