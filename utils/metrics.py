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
    y_pred = np.round(y_pred)
    # Round predictions to nearest integer for classification tasks
    return np.mean(y_true == y_pred)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
