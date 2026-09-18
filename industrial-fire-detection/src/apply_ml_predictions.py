import pandas as pd
import joblib

def apply_predictions():
    print("Loading trained model...")
    model = joblib.load("data/processed/fire_classifier_model.pkl")

    print("Loading current hotspots...")
    df = pd.read_csv("data/processed/hotspots_with_anomalies.csv")

    df["daynight_flag"] = df["daynight"].map({"D": 1, "N": 0}).fillna(1)
    df["confidence_num"] = df["confidence"].map({"l": 0, "n": 1, "h": 2}).fillna(1)

    feature_cols = ["frp", "bright_ti4", "nearest_industrial_km", "daynight_flag", "confidence_num"]

    valid_mask = df[feature_cols].notna().all(axis=1)
    X = df.loc[valid_mask, feature_cols]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    confidence_scores = probabilities.max(axis=1)  # highest class probability

    df.loc[valid_mask, "ml_prediction"] = predictions
    df.loc[valid_mask, "ml_confidence"] = confidence_scores

    df["ml_prediction"] = df["ml_prediction"].fillna("N/A")
    df["ml_confidence"] = df["ml_confidence"].fillna(0)

    df["ml_agrees_with_rules"] = df["ml_prediction"] == df["final_class"]

    output_path = "data/processed/hotspots_with_anomalies.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved ML predictions to: {output_path}")
    print(f"\nML agrees with rule-based classification: {df['ml_agrees_with_rules'].sum()}/{len(df)} ({df['ml_agrees_with_rules'].mean()*100:.1f}%)")
    print(f"\nAverage ML confidence: {df['ml_confidence'].mean():.3f}")

if __name__ == "__main__":
    apply_predictions()