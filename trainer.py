import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

class AssetCentricAIEngine:
    def __init__(self, n_estimators=100, contamination=0.01):
        """
        Initializes the Isolation Forest architecture for anomaly tracking.
        contamination=0.01 aligns with the optimal low error thresholds.
        """
        self.model = IsolationForest(
            n_estimators=n_estimators, 
            contamination=contamination, 
            random_state=42
        )
        self.is_trained = False

    def generate_synthetic_baseline(self, n_samples=200):
        """
        Generates baseline historical telemetry (legitimate asset transactions).
        Features sequence: [Price Volatility Deviation (%), Ingestion Velocity (s)]
        """
        np.random.seed(42)
        # Legitimate transactions: low price skew (0-15%) and stable temporal windows (10-300s)
        normal_price_dev = np.random.uniform(0.0, 0.15, n_samples)
        normal_velocity = np.random.uniform(10.0, 300.0, n_samples)
        
        return np.column_stack((normal_price_dev, normal_velocity))

    def train_baseline_model(self):
        """Trains the analytics core using clean asset historical records."""
        X_train = self.generate_synthetic_baseline()
        self.model.fit(X_train)
        self.is_trained = True
        # Model serialization pipeline execution for background task queuing
        joblib.dump(self.model, 'asset_anomaly_model.pkl')
        print("[AI Engine] Baseline production model trained and serialized successfully.")

    def evaluate_transaction(self, price_deviation, current_velocity):
        """
        Evaluates incoming transaction profiles synchronously at the edge perimeter.
        Returns a normalized asset anomaly score bounded between 0.0 and 1.0.
        """
        if not self.is_trained:
            try:
                self.model = joblib.load('asset_anomaly_model.pkl')
                self.is_trained = True
            except:
                self.train_baseline_model()

        # Formatting the target instance matrix: [\Delta P, AssetVelocity]
        tx_vector = np.array([[price_deviation, current_velocity]])
        
        # Scaling the Isolation Forest sample trace score to a [0, 1] threat level
        raw_score = self.model.score_samples(tx_vector)
        anomaly_score = 1.0 - (raw_score + 1.0) / 2.0
        return anomaly_score

if __name__ == "__main__":
    ai_engine = AssetCentricAIEngine()
    ai_engine.train_baseline_model()
