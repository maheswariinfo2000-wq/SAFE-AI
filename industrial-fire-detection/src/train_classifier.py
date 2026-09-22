import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

FEATURE_COLS = [
    "frp", "log_frp", "bright_ti4", "bright_ti5", "delta_t", "temp_ratio",
    "frp_density", "frp_per_temp",
    "nearest_industrial_km", "log_industrial_dist",
    "nearest_place_km", "log_place_dist", "dist_ratio",
    "is_urban", "is_village",
    "in_forest_flag", "in_farmland_flag",
    "daynight_flag", "confidence_num", "hour", "solar_hour",
    "latitude", "longitude", "lat_lon_prod"
]

def extract_features(df):
    data = df.copy()
    data["bright_ti4"] = pd.to_numeric(data.get("bright_ti4", 320.0), errors="coerce").fillna(320.0)
    data["bright_ti5"] = pd.to_numeric(data.get("bright_ti5", data["bright_ti4"] - 30.0), errors="coerce").fillna(data["bright_ti4"] - 30.0)
    data["delta_t"] = data["bright_ti4"] - data["bright_ti5"]
    data["temp_ratio"] = data["bright_ti4"] / (data["bright_ti5"].clip(lower=100))

    scan = pd.to_numeric(data.get("scan", 0.5), errors="coerce").fillna(0.5)
    track = pd.to_numeric(data.get("track", 0.5), errors="coerce").fillna(0.5)
    pixel_area = (scan * track).replace(0, 0.25)
    data["frp"] = pd.to_numeric(data.get("frp", 5.0), errors="coerce").fillna(5.0)
    data["frp_density"] = data["frp"] / pixel_area
    data["log_frp"] = np.log1p(data["frp"])
    data["frp_per_temp"] = data["frp"] / data["bright_ti4"].clip(lower=100)

    data["nearest_industrial_km"] = pd.to_numeric(data.get("nearest_industrial_km", 10.0), errors="coerce").fillna(10.0)
    data["log_industrial_dist"] = np.log1p(data["nearest_industrial_km"])
    data["nearest_place_km"] = pd.to_numeric(data.get("nearest_place_km", 5.0), errors="coerce").fillna(5.0)
    data["log_place_dist"] = np.log1p(data["nearest_place_km"])
    data["dist_ratio"] = data["nearest_industrial_km"] / (data["nearest_place_km"] + 0.05)

    place_type = data.get("nearest_place_type", "").astype(str).str.lower()
    data["is_urban"] = place_type.isin(["city", "town", "suburb"]).astype(int)
    data["is_village"] = place_type.isin(["village", "hamlet"]).astype(int)

    data["in_forest_flag"] = data.get("in_forest", False).fillna(False).astype(int)
    data["in_farmland_flag"] = data.get("in_farmland", False).fillna(False).astype(int)

    data["daynight_flag"] = data.get("daynight", "D").map({"D": 1, "N": 0}).fillna(1)
    data["confidence_num"] = data.get("confidence", "n").map({"l": 0, "n": 1, "h": 2}).fillna(1)

    acq_time = pd.to_numeric(data.get("acq_time", 1200), errors="coerce").fillna(1200)
    data["hour"] = (acq_time // 100).clip(0, 23)

    data["latitude"] = pd.to_numeric(data.get("latitude", 11.0), errors="coerce").fillna(11.0)
    data["longitude"] = pd.to_numeric(data.get("longitude", 78.5), errors="coerce").fillna(78.5)
    data["solar_hour"] = (data["hour"] + (data["longitude"] / 15.0)) % 24
    data["lat_lon_prod"] = data["latitude"] * data["longitude"]

    return data[FEATURE_COLS].fillna(0)

def train_model():
    print("Loading labeled training data...")
    if os.path.exists("data/processed/all_tamilnadu_hotspots.csv"):
        df = pd.read_csv("data/processed/all_tamilnadu_hotspots.csv")
    else:
        df = pd.read_csv("data/processed/final_classified_hotspots.csv")

    df = df[df["final_class"].notna() & (df["final_class"] != "unclassified")].copy()

    X = extract_features(df)
    y = df["final_class"]

    print(f"Total training samples: {len(X)}")
    print("\nClass distribution:")
    print(y.value_counts())

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Base estimators
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=18,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=10,
        learning_rate=0.08,
        random_state=42
    )

    # 2. Weighted soft-voting ensemble
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("hgb", hgb)],
        voting="soft",
        weights=[3, 1]
    )

    print("\nEvaluating 5-Fold Stratified Cross-Validation on Base Random Forest...")
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
    print(f"Random Forest CV Accuracy: {rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")

    print("\nEvaluating 5-Fold Stratified Cross-Validation on Weighted Ensemble...")
    ens_scores = cross_val_score(ensemble, X, y, cv=cv, scoring="accuracy")
    print(f"Ensemble CV Accuracy:      {ens_scores.mean():.4f} (+/- {ens_scores.std():.4f})")

    # Detailed Out-of-Fold Performance
    print("\nComputing full Out-of-Fold (OOF) cross-validated predictions...")
    oof_preds = cross_val_predict(rf, X, y, cv=cv)
    oof_acc = accuracy_score(y, oof_preds)
    print(f"OOF Overall Accuracy: {oof_acc:.4f} ({oof_acc*100:.2f}%)")

    print("\nDetailed Out-of-Fold Classification Report:")
    report_dict = classification_report(y, oof_preds, digits=4, output_dict=True)
    print(classification_report(y, oof_preds, digits=4))

    labels = sorted(y.unique())
    cm = confusion_matrix(y, oof_preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])
    print("Confusion Matrix:")
    print(cm_df)

    # Fit RF to inspect feature importances
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nTop 12 Most Important Features:")
    for feat, imp in importances.head(12).items():
        print(f"  - {feat:22s}: {imp:.4f} ({imp*100:.1f}%)")

    # Fit Probability Calibrated Classifier for deployment
    print("\nCalibrating probabilities with CalibratedClassifierCV (sigmoid)...")
    calibrated_model = CalibratedClassifierCV(estimator=rf, cv=5, method="sigmoid")
    calibrated_model.fit(X, y)

    model_payload = {
        "model": calibrated_model,
        "raw_model": rf,
        "features": FEATURE_COLS,
        "classes": calibrated_model.classes_.tolist(),
        "accuracy": float(rf_scores.mean()),
        "oof_accuracy": float(oof_acc),
        "confusion_matrix": cm.tolist(),
        "feature_importances": importances.to_dict()
    }

    output_path = "data/processed/fire_classifier_model.pkl"
    joblib.dump(model_payload, output_path)
    print(f"\nModel successfully saved to: {output_path}")

    return calibrated_model

if __name__ == "__main__":
    train_model()