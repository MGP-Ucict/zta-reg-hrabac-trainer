import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

print("=" * 70)
print("ZTA-Reg-HRABAC: Core Behavioral AI Training Engine")
print("=" * 70)

# 1. Synthetic Ingestion Telemetry Generation for Cadastral Baseline
# Simulates long-term historical baseline of legitimate, stable property transactions
print("[INFO] Generating baseline property transaction metadata for calibration...")
np.random.seed(42)
normal_price_deviations = np.random.normal(loc=0.05, scale=0.05, size=200)
normal_intervals = np.random.uniform(low=5.0, high=30.0, size=200)
X_train = np.column_stack((normal_price_deviations, normal_intervals))

# 2. Unsupervised Isolation Forest Model Initialisation & Training
print("[INFO] Training unsupervised Isolation Forest core cluster...")
if 'random_state' in IsolationForest.__init__.__code__.co_varnames:
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
else:
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01)

iso_forest.fit(X_train)
print(f"[SUCCESS] Model successfully fitted. Total sample nodes processed: {len(X_train)}")

# 3. Serialization & Export to Shared Storage Matrix
model_filename = 'iso_forest_model.pkl'
print(f"[INFO] Serializing trained neural partition to {model_filename}...")
joblib.dump(iso_forest, model_filename)

print("-" * 70)
print("[STATUS] Training pipeline complete. Engine state locked and preserved.")
print("=" * 70)
