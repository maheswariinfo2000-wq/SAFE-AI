import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_model():
    print("Loading labeled data (from rule-based classification)...")
    df = pd.read_csv("data/processed/final_classified_hotspots.csv")

    # Build features
    df["daynight_flag"] = df["daynight"].map({"D": 1, "N": 0})
    df["confidence_num"] = df["confidence"].map({"l": 0, "n": 1, "h": 2}).fillna(1)

    feature_cols = ["frp", "bright_ti4", "nearest_industrial_km", "daynight_flag", "confidence_num"]
    df = df.dropna(subset=feature_cols + ["final_class"])

    X = df[feature_cols]
    y = df["final_class"]

    print(f"Training samples: {len(X)}")
    print("Class distribution:\n", y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nTest accuracy: {acc:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nFeature importance:")
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(importance)

    joblib.dump(model, "data/processed/fire_classifier_model.pkl")
    print("\nModel saved to: data/processed/fire_classifier_model.pkl")

    return model

if __name__ == "__main__":
    train_model()