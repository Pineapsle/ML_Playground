# Decision Tree Model from Scratch

import numpy as np


class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict(inputs, self.root) for inputs in X])

    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        impurity = 1.0
        for count in counts:
            prob = count / len(y)
            impurity -= prob ** 2
        return impurity

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_idx, best_thr = None, None
        n_samples, n_features = X.shape
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = X[:, feature_idx] > threshold
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])
                gini = (len(y[left_mask]) * gini_left + len(y[right_mask]) * gini_right) / n_samples
                if gini < best_gini:
                    best_gini = gini
                    best_idx = feature_idx
                    best_thr = threshold
        return best_idx, best_thr

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        if (self.max_depth is not None and depth >= self.max_depth) or num_labels == 1 or num_samples < self.min_samples_split:
            leaf_value = self._majority_vote(y)
            return DecisionTreeNode(value=leaf_value)

        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None:
            leaf_value = self._majority_vote(y)
            return DecisionTreeNode(value=leaf_value)

        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return DecisionTreeNode(feature_index=feature_idx, threshold=threshold, left=left, right=right)

    def _majority_vote(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _predict(self, inputs, node):
        if node.value is not None:
            return node.value
        if inputs[node.feature_index] <= node.threshold:
            return self._predict(inputs, node.left)
        else:
            return self._predict(inputs, node.right)