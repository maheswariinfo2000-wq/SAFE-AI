"""
compute_risk_scores.py
======================
Computes a 0–100 composite Risk Score for every hotspot in
hotspots_with_anomalies.csv and writes the result back to the
same file (adding a `risk_score` and `risk_level` column).

Risk Score formula (weighted sum, capped at 100):
  Component                       Weight   Max pts
  ─────────────────────────────── ──────   ───────
  1. ML confidence (if industrial)   25      25
  2. Anomaly severity (FRP/baseline) 25      25
  3. Persistence score (days/7)      20      20
  4. Proximity to residential zone   15      15
  5. Raw FRP intensity               15      15

Risk Levels:
  80–100  → CRITICAL
  60–79   → HIGH
  40–59   → MODERATE
  0–39    → LOW
"""

import pandas as pd
import numpy as np
import os


def _frp_intensity_score(frp: pd.Series, max_pts: float = 15.0) -> pd.Series:
    """Log-scaled FRP score.  FRP ≥ 100 MW → full 15 pts."""
    return (np.log1p(frp.clip(lower=0)) / np.log1p(100.0)).clip(0, 1) * max_pts


def _anomaly_severity_score(row: pd.Series) -> float:
    """How many σ above baseline is the current FRP?  Capped at 5σ → 25 pts."""
    if not row.get("is_anomaly", False):
        return 0.0
    mean = row.get("baseline_frp_mean", np.nan)
    std  = row.get("baseline_frp_std",  np.nan)
    frp  = row.get("frp", 0.0)
    if pd.isna(mean) or pd.isna(std) or std == 0:
        # Anomaly flagged but no baseline → give half credit
        return 12.5
    sigma = (frp - mean) / std
    return min(sigma / 5.0, 1.0) * 25.0


def _persistence_score(is_persistent, max_pts: float = 20.0) -> float:
    """Binary: persistent source (≥4 days in last 7) → full 20 pts."""
    if is_persistent is True or str(is_persistent).lower() in ("true", "1"):
        return max_pts
    return 0.0


def _ml_confidence_score(row: pd.Series, max_pts: float = 25.0) -> float:
    """
    Only awarded for industrial_fire predictions (where confidence is meaningful
    as a risk signal).  Scale: conf * max_pts.
    """
    pred  = row.get("ml_prediction", "") or row.get("final_class", "")
    conf  = row.get("ml_confidence", np.nan)
    if str(pred) == "industrial_fire":
        conf_val = pd.to_numeric(conf, errors="coerce")
        if pd.isna(conf_val):
            conf_val = 0.95
        return float(conf_val) * max_pts
    return 0.0


def _proximity_residential_score(nearest_place_km, nearest_place_type,
                                  max_pts: float = 15.0) -> float:
    """
    Risk increases as a fire gets closer to residential areas.
    City/town within 1 km → full score.
    Village within 2 km  → 70%.
    Hamlet within 5 km   → 40%.
    Otherwise → 0.
    """
    try:
        dist = float(nearest_place_km)
    except (TypeError, ValueError):
        return 0.0

    ptype = str(nearest_place_type).lower()

    if ptype in ("city", "town", "suburb"):
        if dist <= 1.0:
            return max_pts
        elif dist <= 3.0:
            return max_pts * 0.6
        elif dist <= 6.0:
            return max_pts * 0.3
    elif ptype in ("village",):
        if dist <= 2.0:
            return max_pts * 0.7
        elif dist <= 5.0:
            return max_pts * 0.4
    elif ptype in ("hamlet",):
        if dist <= 5.0:
            return max_pts * 0.3

    return 0.0


def _risk_level(score: float) -> str:
    if score >= 80:
        return "CRITICAL"
    elif score >= 60:
        return "HIGH"
    elif score >= 40:
        return "MODERATE"
    return "LOW"


def compute_risk_scores(input_path: str = "data/processed/hotspots_with_anomalies.csv",
                        output_path: str | None = None) -> pd.DataFrame:
    """
    Load hotspot data, compute risk scores, write results.
    Returns the enriched DataFrame.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # ── Component 1: ML confidence ─────────────────────────────────────────
    df["_c1_ml"] = df.apply(_ml_confidence_score, axis=1)

    # ── Component 2: Anomaly severity ──────────────────────────────────────
    df["_c2_anomaly"] = df.apply(_anomaly_severity_score, axis=1)

    # ── Component 3: Persistence ───────────────────────────────────────────
    # is_persistent_source is computed at API time; default to False if absent
    persist_col = df["is_persistent_source"] if "is_persistent_source" in df.columns else pd.Series([False] * len(df))
    df["_c3_persist"] = persist_col.apply(_persistence_score)

    # ── Component 4: Proximity to residential ──────────────────────────────
    near_place_km   = df.get("nearest_place_km",   pd.Series([999.0] * len(df)))
    near_place_type = df.get("nearest_place_type", pd.Series([""] * len(df)))
    df["_c4_proximity"] = [
        _proximity_residential_score(km, pt)
        for km, pt in zip(near_place_km, near_place_type)
    ]

    # ── Component 5: Raw FRP intensity ─────────────────────────────────────
    df["_c5_frp"] = _frp_intensity_score(
        pd.to_numeric(df["frp"], errors="coerce").fillna(0.0)
    )

    # ── Total risk score ───────────────────────────────────────────────────
    df["risk_score"] = (
        df["_c1_ml"] +
        df["_c2_anomaly"] +
        df["_c3_persist"] +
        df["_c4_proximity"] +
        df["_c5_frp"]
    ).clip(0, 100).round(1)

    df["risk_level"] = df["risk_score"].apply(_risk_level)

    # Clean up temp columns
    df.drop(columns=[c for c in df.columns if c.startswith("_c")], inplace=True)

    out = output_path or input_path
    df.to_csv(out, index=False)
    print(f"[Risk Scores] Computed for {len(df)} hotspots -> saved to {out}")
    print(df[["latitude", "longitude", "frp", "risk_score", "risk_level"]].sort_values(
        "risk_score", ascending=False).head(10).to_string(index=False))

    return df


if __name__ == "__main__":
    compute_risk_scores()
