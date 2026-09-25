import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class LinearL1NormMRC:
    """Single-Pass L1-Norm Pseudo-Inverse Target Sequencer for Linear Domains."""
    
    def __init__(self):
        self.proposed_order = []
        self.chain_models = []

    def _iqr_scale(self, X):
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        iqr = q75 - q25
        iqr[iqr == 0] = 1.0
        return (X - np.median(X, axis=0)) / iqr

    def fit(self, X, Y):
        X_s = self._iqr_scale(X)
        Y_s = self._iqr_scale(Y)

        # Joint SVD Pseudo-Inverse Matrix Decomposition
        X_pinv = np.linalg.pinv(X_s)
        W = X_pinv @ Y_s

        # L1-Norm Weight Magnitude Summation
        S_linear = np.sum(np.abs(W), axis=0)
        
        # Global Sequence Ordering (Descending)
        self.proposed_order = np.argsort(S_linear)[::-1].tolist()

        # Train Regressor Chain
        self.chain_models = []
        current_X = X.copy()
        
        for t in self.proposed_order:
            model = LinearRegression()
            model.fit(current_X, Y[:, t])
            self.chain_models.append((model, t))
            current_X = np.hstack([current_X, Y[:, t : t + 1]])

        return self

    def predict(self, X):
        N = X.shape[0]
        K = len(self.proposed_order)
        test_preds = np.zeros((N, K), dtype=np.float64)

        current_X = X.copy()
        for model, t in self.chain_models:
            pred = model.predict(current_X)
            test_preds[:, t] = pred
            current_X = np.hstack([current_X, pred.reshape(-1, 1)])

        return test_preds


class LinearPredictionEngine:
    def __init__(self, dataset_path, target_cols):
        self.dataset_path = dataset_path
        self.target_cols = target_cols
        self.feature_cols = []
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = LinearL1NormMRC()
        self.is_fitted = False

    def train(self):
        df = pd.read_csv(self.dataset_path).dropna()
        self.feature_cols = [c for c in df.columns if c not in self.target_cols]
        
        X = df[self.feature_cols].values
        Y = df[self.target_cols].values

        X_scaled = self.scaler_x.fit_transform(X)
        Y_scaled = self.scaler_y.fit_transform(Y)

        self.model.fit(X_scaled, Y_scaled)
        self.is_fitted = True
        
        ordered_targets = [self.target_cols[i] for i in self.model.proposed_order]
        print(f"Linear Engine Trained! Target Sequence: {ordered_targets}")

    def predict(self, input_list):
        if not self.is_fitted:
            raise RuntimeError("মডেলটি আগে ট্রেইন করতে হবে।")
            
        X_raw = np.array(input_list).reshape(1, -1)
        if X_raw.shape[1] != len(self.feature_cols):
            raise ValueError(f"{len(self.feature_cols)} টি ফিচার ইনপুট দিতে হবে।")

        X_scaled = self.scaler_x.transform(X_raw)
        Y_scaled_pred = self.model.predict(X_scaled)
        Y_pred = self.scaler_y.inverse_transform(Y_scaled_pred)

        return dict(zip(self.target_cols, Y_pred[0]))

# example:
# engine = LinearPredictionEngine("dataset.csv", target_cols=["T1", "T2", "T3"])
# engine.train()
# result = engine.predict([1.5, 2.3, 4.1, 0.9])
# print(result)
