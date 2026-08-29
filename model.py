"""
End-to-End Multi-Target Regressor Chain (MRC) Prediction Pipeline
Algorithmic Core: O(1) Matrix Pseudoinverse Target Ordering (By MD Jabid Muntasir)

Features:
- Dynamic Dataset Loading & Preprocessing
- Robust Scaling (IQR Standard)
- O(1) Single-Pass Target Sequence Optimization
- Interactive User Input & Real-Time Multi-Target Prediction
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import RobustScaler


class FastMRCPipeline:
    def __init__(self):
        self.scaler_X = RobustScaler()
        self.scaler_Y = RobustScaler()
        self.feature_names = []
        self.target_names = []
        self.optimal_order = []
        self.chain_model = None
        self.is_trained = False

    def compute_pseudoinverse_order(self, X_scaled: np.ndarray, Y_scaled: np.ndarray) -> list:
        """
        YOUR CORE UNTOUCHED LOGIC:
        Computes optimal target sequence in O(1) pass using SVD Pseudo-Inverse
        and L1-Norm Weight Magnitude Summation.
        """
        # 1. Single-pass Moore-Penrose Pseudo-inverse calculation
        X_pinv = np.linalg.pinv(X_scaled)

        # 2. Joint Weight Matrix Calculation (W = X_pinv @ Y)
        W = X_pinv @ Y_scaled

        # 3. L1-norm Weight Summation per target column
        weight_magnitudes = np.sum(np.abs(W), axis=0)

        # 4. Descending order sequence permutation
        proposed_order = list(np.argsort(weight_magnitudes)[::-1])
        return proposed_order

    def fit_from_csv(self, file_path: str, target_columns: list):
        """
        Loads CSV dataset, separates features and targets, applies Robust Scaling,
        determines sequence order using your O(1) logic, and fits the Regressor Chain.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Error: Dataset file not found at path: {file_path}")

        print(f"\n[1/4] Loading dataset from: {file_path}")
        df = pd.read_csv(file_path)

        # Validate target columns
        missing_targets = [col for col in target_columns if col not in df.columns]
        if missing_targets:
            raise ValueError(f"Target columns missing in CSV: {missing_targets}")

        self.target_names = target_columns
        self.feature_names = [col for col in df.columns if col not in target_columns]

        X_raw = df[self.feature_names].values
        Y_raw = df[self.target_names].values

        print(f"[2/4] Dataset Specs: {X_raw.shape[0]} samples | {len(self.feature_names)} features | {len(self.target_names)} targets")

        # Apply Robust Scaling (IQR Standard)
        X_scaled = self.scaler_X.fit_transform(X_raw)
        Y_scaled = self.scaler_Y.fit_transform(Y_raw)

        # Compute optimal sequence order using your exact O(1) logic
        print("[3/4] Optimizing sequence order using O(1) Pseudoinverse L1-Norm method...")
        self.optimal_order = self.compute_pseudoinverse_order(X_scaled, Y_scaled)

        ordered_target_names = [self.target_names[i] for i in self.optimal_order]
        print(f"      Calculated Optimal Sequence Order : {self.optimal_order}")
        print(f"      Target Dependency Chain Sequence  : {' -> '.join(ordered_target_names)}")

        # Train Regressor Chain
        print("[4/4] Fitting Regressor Chain with calculated sequence...")
        self.chain_model = RegressorChain(LinearRegression(), order=self.optimal_order)
        self.chain_model.fit(X_scaled, Y_scaled)
        
        self.is_trained = True
        print("\n--> Pipeline successfully trained and ready for user predictions!\n")

    def predict_single_input(self, user_features: list) -> dict:
        """
        Takes raw user input features, normalizes using fitted RobustScaler,
        runs prediction through the optimized Regressor Chain, and inverse-transforms targets.
        """
        if not self.is_trained:
            raise RuntimeError("Pipeline must be trained before predicting.")

        if len(user_features) != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, but got {len(user_features)}.")

        # Reshape and scale input features
        X_input = np.array(user_features).reshape(1, -1)
        X_scaled = self.scaler_X.transform(X_input)

        # Predict using Regressor Chain
        Y_scaled_pred = self.chain_model.predict(X_scaled)

        # Inverse transform to get original target scale
        Y_pred = self.scaler_Y.inverse_transform(Y_scaled_pred)[0]

        # Format output as a dictionary mapping target names to predicted values
        return {target_name: float(pred_val) for target_name, pred_val in zip(self.target_names, Y_pred)}


# =====================================================================
# INTERACTIVE CLI INTERFACE FOR USER
# =====================================================================
if __name__ == "__main__":
    pipeline = FastMRCPipeline()

    print("=" * 70)
    print(" Fast-MRC: Single-Pass Pseudoinverse Target Ordering Prediction CLI")
    print(" Core Mathematical Engine: O(1) Matrix Pseudoinverse (MD Jabid Muntasir)")
    print("=" * 70)

    # Step 1: Get Dataset Path
    csv_path = input("Enter path to your dataset CSV file (e.g., data.csv): ").strip()
    
    if not csv_path:
        # Fallback dummy sample creator if user enters nothing for testing
        print("\n[Notice] No file path entered. Generating a sample dataset 'sample_dataset.csv' for demonstration...")
        dummy_df = pd.DataFrame(
            np.random.randn(500, 7),
            columns=['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Target_A', 'Target_B', 'Target_C']
        )
        csv_path = "sample_dataset.csv"
        dummy_df.to_csv(csv_path, index=False)
        targets_input = "Target_A, Target_B, Target_C"
    else:
        targets_input = input("Enter target column names (comma-separated, e.g., Target1, Target2): ").strip()

    target_list = [t.strip() for t in targets_input.split(",") if t.strip()]

    # Step 2: Process Dataset and Train Pipeline
    try:
        pipeline.fit_from_csv(csv_path, target_list)
    except Exception as e:
        print(f"\nTraining Failed: {e}")
        sys.exit(1)

    # Step 3: Interactive Prediction Loop
    print("-" * 70)
    print("INTERACTIVE PREDICTION MODE")
    print("Provide feature values below to get real-time multi-target predictions.")
    print("-" * 70)

    while True:
        print(f"\nPlease enter values for the following {len(pipeline.feature_names)} features:")
        user_inputs = []
        
        try:
            for feat in pipeline.feature_names:
                val = float(input(f"  -> {feat}: "))
                user_inputs.append(val)
        except ValueError:
            print("[Error] Invalid numeric input. Resetting entry...")
            continue
        except (KeyboardInterrupt, EOFError):
            print("\nExiting prediction pipeline.")
            break

        # Predict
        results = pipeline.predict_single_input(user_inputs)

        print("\n" + "=" * 40)
        print(" PREDICTED MULTI-TARGET OUTPUTS")
        print("=" * 40)
        for target_col, val in results.items():
            print(f"  • {target_col:<20} : {val:.6f}")
        print("=" * 40)

        cont = input("\nDo you want to predict for another sample? (y/n): ").strip().lower()
        if cont != 'y':
            print("Pipeline execution completed.")
            break
