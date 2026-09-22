import pandas as pd
import numpy as np
import joblib
import os
import sys

# Ensure project root is in sys.path so 'src' can always be resolved
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.train_classifier import extract_features, FEATURE_COLS
except ImportError:
    from train_classifier import extract_features, FEATURE_COLS

def apply_predictions():
    print("Loading trained model...")
    model_path = os.path.join(PROJECT_ROOT, "data/processed/fire_classifier_model.pkl")
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please train model first.")
        return

    loaded = joblib.load(model_path)
    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
    else:
        model = loaded

    targets = [
        (os.path.join(PROJECT_ROOT, "data/processed/hotspots_with_anomalies.csv"), "Live Hotspots"),
        (os.path.join(PROJECT_ROOT, "data/processed/all_tamilnadu_hotspots.csv"), "All Tamil Nadu Hotspots (Historical)")
    ]

    for hotspot_path, label in targets:
        if not os.path.exists(hotspot_path):
            continue

        print(f"\nApplying calibrated ML predictions to {label} ({hotspot_path})...")
        df = pd.read_csv(hotspot_path)

        X = extract_features(df)
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        confidence_scores = probabilities.max(axis=1)

        df["ml_prediction"] = predictions
        df["ml_confidence"] = np.round(confidence_scores, 3)

        if "final_class" in df.columns:
            valid_mask = df["final_class"].notna() & (df["final_class"] != "unclassified")
            agrees = (df.loc[valid_mask, "ml_prediction"] == df.loc[valid_mask, "final_class"]).sum()
            total_valid = valid_mask.sum()
            pct = (agrees / total_valid * 100) if total_valid > 0 else 0
            print(f"ML agrees with rule-based classification: {agrees}/{total_valid} ({pct:.2f}%)")

        print(f"Average ML calibrated confidence: {df['ml_confidence'].mean():.3f}")
        df.to_csv(hotspot_path, index=False)
        print(f"Saved updated predictions to {hotspot_path}")

    return model

if __name__ == "__main__":
    apply_predictions()