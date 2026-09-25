import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

class NonLinearKernelSVDMRC:
    """Regularized Dynamic Kernel-SVD Sequence Optimizer for Non-Linear Domains."""
    
    def __init__(self, degree=2, alpha=1.0, gamma=0.01, random_state=42):
        self.degree = degree
        self.alpha = alpha
        self.gamma = gamma
        self.random_state = random_state
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
        K = Y_s.shape[1]
        
        # Polynomial Kernel Expansion
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        X_phi = poly.fit_transform(X_s)

        remaining = list(range(K))
        self.proposed_order = []
        
        # Initial Augmented Matrix starts with Kernel features
        current_X_aug = X_phi.copy()

        for step in range(K):
            XtX = current_X_aug.T @ current_X_aug
            lambda_I = self.alpha * np.eye(XtX.shape[0])
            reg_pinv = np.linalg.pinv(XtX + lambda_I, rcond=1e-12)
            projection_mat = reg_pinv @ current_X_aug.T

            best_t = None
            best_res = float("inf")

            for t in remaining:
                y_t = Y_s[:, t]
                w_t = projection_mat @ y_t
                y_hat = current_X_aug @ w_t
                base_res = np.mean((y_t - y_hat) ** 2)

                # Condition-Number Penalty
                test_matrix = np.hstack([current_X_aug, y_t.reshape(-1, 1)])
                cond_val = np.linalg.cond(test_matrix)
                penalty_factor = 1.0 + (self.gamma * np.log1p(cond_val))
                penalized_res = base_res * penalty_factor

                if penalized_res < best_res:
                    best_res = penalized_res
                    best_t = t

            self.proposed_order.append(best_t)
            remaining.remove(best_t)
            current_X_aug = np.hstack([current_X_aug, Y_s[:, best_t : best_t + 1]])

        # Train Random Forest Regressor Chain
        self.chain_models = []
        current_X = X.copy()
        
        for t in self.proposed_order:
            model = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=self.random_state)
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


class NonLinearPredictionEngine:
    def __init__(self, dataset_path, target_cols):
        self.dataset_path = dataset_path
        self.target_cols = target_cols
        self.feature_cols = []
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = NonLinearKernelSVDMRC(degree=2, alpha=1.0, gamma=0.01)
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
        print(f"Non-Linear Engine Trained! Target Sequence: {ordered_targets}")

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
# engine = NonLinearPredictionEngine("dataset.csv", target_cols=["T1", "T2", "T3"])
# engine.train()
# result = engine.predict([1.5, 2.3, 4.1, 0.9])
# print(result)
