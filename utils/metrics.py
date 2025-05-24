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
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

def confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix for binary classification.
    
    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.
    
    Returns:
    tuple: True Positives, True Negatives, False Positives, False Negatives
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp],
                     [fn, tp]])

def precision_recall_f1(y_true, y_pred):
    """
    Calculate precision, recall, and F1 score for binary classification.
    
    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.
    
    Returns:
    tuple: Precision, Recall, F1 Score
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return precision, recall, f1

def evaluate_classification_model(y_true, y_pred, model_name="Model"):
    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    acc = accuracy(y_true, y_pred)

    print(f" \n------ {model_name} Evaluation ------ ")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision: {precision:.5f}")
    print(f"Recall: {recall:.5f}")
    print(f"F1 Score: {f1:.5f}")
    print(f"Accuracy: {acc:.5f}")

def binarize_target(y_train, y_test):
    # Convert target to binary: 1 if above median price, else 0
    median_price = np.median(np.concatenate((y_train, y_test)))
    return (y_train > median_price).astype(int), (y_test > median_price).astype(int)