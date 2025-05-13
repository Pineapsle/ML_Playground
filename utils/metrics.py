# Functions for calculating accuracy, loss, etc. 

import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Calculate Mean Squared Error (MSE) between true and predicted values.
    
    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.
    
    Returns:
    float: Mean Squared Error.
    """
    return np.mean((y_true - y_pred) ** 2)

def accuracy(y_true, y_pred):
    """
    Calculate accuracy of predictions.
    
    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.
    
    Returns:
    float: Accuracy score.
    """
    return np.mean(y_true == y_pred)