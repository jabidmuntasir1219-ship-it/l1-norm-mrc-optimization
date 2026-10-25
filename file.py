"""
Title: Dependency-Based Target Sequence Optimization in Multi-Target Regressor Chains
Author: MD Jabid Muntasir
Description: Implementation of Single-Pass L1-Norm Linear Sequence Optimization 
             and Regularized Dynamic Kernel-SVD Non-Linear Sequence Optimization.
"""

import numpy as np

class LinearMRCSeqOptimizer:
    """
    Algorithm 1: Single-Pass L1-Norm Pseudo-Inverse Sequence Optimization (Linear Regime)
    """
    def __init__(self, epsilon: float = 1e-12):
        self.epsilon = epsilon

    def fit_order(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Computes the optimal target chain sequence for linear multi-target regression.
        
        Parameters:
        -----------
        X : np.ndarray of shape (N, M)
            Training feature matrix.
        Y : np.ndarray of shape (N, K)
            Training target matrix.
            
        Returns:
        --------
        np.ndarray of shape (K,)
            Optimal target permutation indices (pi*).
        """
        # Step 1 & 2: Robust Interquartile Range (IQR) Standardization
        median_X = np.median(X, axis=0)
        iqr_X = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        X_scaled = (X - median_X) / np.maximum(iqr_X, self.epsilon)

        median_Y = np.median(Y, axis=0)
        iqr_Y = np.percentile(Y, 75, axis=0) - np.percentile(Y, 25, axis=0)
        Y_scaled = (Y - median_Y) / np.maximum(iqr_Y, self.epsilon)

        # Step 3: Compact Singular Value Decomposition (SVD) of X_scaled
        U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)

        # Step 4: Moore-Penrose Pseudo-Inverse with stability floor epsilon
        sigma_inv = np.array([1.0 / sigma if sigma > self.epsilon else 0.0 for sigma in s])
        X_pinv = Vt.T @ np.diag(sigma_inv) @ U.T

        # Step 5: Joint multi-target weight matrix W
        W = X_pinv @ Y_scaled  # shape: (M, K)

        # Step 6 & 7: Linear Dependency Score (L1-norm magnitude sum across features)
        scores = np.sum(np.abs(W), axis=0)  # shape: (K,)

        # Step 8: Sort scores in descending order to get optimal sequence pi*
        optimal_order = np.argsort(scores)[::-1]
        return optimal_order


class NonLinearKernelSVDSeqOptimizer:
    """
    Algorithm 2: Regularized Dynamic Kernel-SVD Sequence Optimization (Non-Linear Regime)
    """
    def __init__(self, degree: int = 2, lambda_reg: float = 1.0, gamma: float = 0.5, epsilon: float = 1e-12):
        self.degree = degree
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.epsilon = epsilon

    def _expand_polynomial(self, X: np.ndarray) -> np.ndarray:
        """
        Constructs explicit degree-2 polynomial kernel feature mapping Phi(X)
        matching Eq. (19) in the manuscript.
        """
        N, M = X.shape
        features = [np.ones((N, 1))]  # Intercept term
        
        # Linear terms: sqrt(2) * x_i
        features.append(np.sqrt(2) * X)
        
        # Quadratic terms: x_i^2
        features.append(X ** 2)
        
        # Cross-terms: sqrt(2) * x_i * x_j for i < j
        cross_terms = []
        for i in range(M):
            for j in range(i + 1, M):
                cross_terms.append(np.sqrt(2) * X[:, i:i+1] * X[:, j:j+1])
        if cross_terms:
            features.append(np.hstack(cross_terms))
            
        return np.hstack(features)

    def fit_order(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Computes the optimal stabilized non-linear target chain sequence.
        
        Parameters:
        -----------
        X : np.ndarray of shape (N, M)
            Training feature matrix.
        Y : np.ndarray of shape (N, K)
            Training target matrix.
            
        Returns:
        --------
        np.ndarray of shape (K,)
            Optimal stabilized target permutation indices (pi*).
        """
        N, M = X.shape
        K = Y.shape[1]

        # Step 1: Standardize X and Y using training IQR statistics
        median_X = np.median(X, axis=0)
        iqr_X = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        X_scaled = (X - median_X) / np.maximum(iqr_X, self.epsilon)

        median_Y = np.median(Y, axis=0)
        iqr_Y = np.percentile(Y, 75, axis=0) - np.percentile(Y, 25, axis=0)
        Y_scaled = (Y - median_Y) / np.maximum(iqr_Y, self.epsilon)

        # Step 2: Construct explicit degree-2 polynomial kernel expansion Phi(X_scaled)
        Phi_X = self._expand_polynomial(X_scaled)

        assigned_seq = []
        unassigned_set = list(range(K))

        # Step 3 - 13: Sequential chain link construction
        for k in range(K):
            # Form candidate-independent design D_S = [Phi(X_scaled), Y_S]
            if len(assigned_seq) == 0:
                D_S = Phi_X
            else:
                Y_S = Y_scaled[:, assigned_seq]
                D_S = np.hstack([Phi_X, Y_S])

            best_t = None
            min_score = float('inf')

            for t in unassigned_set:
                y_t = Y_scaled[:, t]
                
                # Step 6: Compute closed-form Tikhonov-regularized (Ridge) residual R_base(t)
                DtD = D_S.T @ D_S
                q = DtD.shape[0]
                inv_term = np.linalg.inv(DtD + self.lambda_reg * np.eye(q))
                w_ridge = inv_term @ (D_S.T @ y_t)
                residual = y_t - D_S @ w_ridge
                R_base = np.sum(residual ** 2) / N

                # Step 7 & 8: Form candidate-augmented system [D_S, y_t] and extract condition number kappa
                X_aug_t = np.hstack([D_S, y_t[:, np.newaxis]])
                singular_values = np.linalg.svd(X_aug_t, compute_uv=False)
                sigma_max = singular_values[0]
                sigma_min = singular_values[-1]
                kappa = sigma_max / max(sigma_min, self.epsilon)

                # Step 9: Evaluate penalized non-linear dependency risk score S_t^nonlinear
                s_nonlinear = R_base * (1.0 + self.gamma * np.log(1.0 + kappa))

                if s_nonlinear < min_score:
                    min_score = s_nonlinear
                    best_t = t

            assigned_seq.append(best_t)
            unassigned_set.remove(best_t)

        return np.array(assigned_seq)


# ==========================================
# Example Usage / Verification Script
# ==========================================
if __name__ == "__main__":
    print("Running verification test for MRC Sequence Optimizers...")
    
    # Generate dummy synthetic data
    np.random.seed(42)
    N_samples = 500
    M_features = 4
    K_targets = 3
    
    X_train = np.random.randn(N_samples, M_features)
    Y_train = np.random.randn(N_samples, K_targets) + 0.5 * X_train[:, 0:1]

    # Test Linear Optimizer
    linear_opt = LinearMRCSeqOptimizer()
    linear_order = linear_opt.fit_order(X_train, Y_train)
    print(f"Optimal Linear Chain Order ($\pi^*$): {linear_order}")

    # Test Non-Linear Kernel-SVD Optimizer
    nonlinear_opt = NonLinearKernelSVDSeqOptimizer(lambda_reg=1.0, gamma=0.5)
    nonlinear_order = nonlinear_opt.fit_order(X_train, Y_train)
    print(f"Optimal Non-Linear Stabilized Chain Order ($\pi^*$): {nonlinear_order}")
    print("Verification completed successfully!")
