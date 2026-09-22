from flask import Flask, jsonify, render_template, request, send_from_directory, make_response
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from sklearn.neighbors import BallTree
from apscheduler.schedulers.background import BackgroundScheduler
from src.send_alerts import send_alert_email
from src.refresh_pipeline import refresh_full_pipeline
from src.compute_risk_scores import compute_risk_scores
from src.push_service import (
    get_public_key,
    add_subscription,
    remove_subscription,
    send_push_notification,
    load_subscriptions,
)

app = Flask(__name__)

DATA_PATH = "data/processed/hotspots_with_anomalies.csv"
HISTORICAL_PATH = "data/processed/all_tamilnadu_hotspots.csv"


def make_readable_name(name, ftype, region):
    if pd.isna(name) or str(name).strip().lower() in ["unnamed", "nan", ""]:
        region_display = str(region).replace("_", " ").title() if pd.notna(region) and str(region).strip() != "" else "Unknown region"
        type_display = str(ftype).replace("_", " ").title() if pd.notna(ftype) else "Industrial"
        return f"{type_display} facility — {region_display}"
    return name


@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/hotspots")
def get_hotspots():
    dataset_mode = request.args.get("dataset", "live").lower()
    target_path = HISTORICAL_PATH if (dataset_mode in ["all", "historical"] and os.path.exists(HISTORICAL_PATH)) else DATA_PATH
    
    df = pd.read_csv(target_path)
    df = df.fillna("")

    df["nearest_industrial_name"] = df.apply(
        lambda r: make_readable_name(r["nearest_industrial_name"], r["nearest_industrial_type"], ""),
        axis=1
    )

    # ---- Persistent source flag per facility ----
    historical = pd.read_csv("data/raw/firms_historical.csv")
    industrial = pd.read_csv("data/raw/osm_industrial.csv")

    industrial_rad = np.radians(industrial[["latitude", "longitude"]].values)
    hist_rad = np.radians(historical[["latitude", "longitude"]].values)
    tree = BallTree(industrial_rad, metric="haversine")
    distances, indices = tree.query(hist_rad, k=1)
    historical["facility_osm_id"] = industrial.iloc[indices.flatten()]["osm_id"].values
    historical["facility_distance_km"] = distances.flatten() * 6371.0
    near_facility = historical[historical["facility_distance_km"] <= 2.0]

    if len(near_facility) > 0:
        max_date = pd.to_datetime(near_facility["acq_date"]).max()
        recent_window = near_facility[
            pd.to_datetime(near_facility["acq_date"]) >= (max_date - pd.Timedelta(days=6))
        ]
        day_counts = recent_window.groupby("facility_osm_id")["acq_date"].nunique()
        persistent_facility_ids = set(day_counts[day_counts >= 4].index)
    else:
        persistent_facility_ids = set()

    df["is_persistent_source"] = df["facility_osm_id"].apply(
        lambda fid: bool(fid) and fid in persistent_facility_ids
    )

    # ---- CPCB-verified name/sector lookup ----
    industrial["osm_id_str"] = industrial["osm_id"].astype(str)

    def get_cpcb_info(fid):
        if not fid or fid == "":
            return "", ""
        match = industrial[industrial["osm_id_str"] == str(fid)]
        if len(match) > 0 and bool(match.iloc[0].get("cpcb_verified", False)):
            return match.iloc[0].get("cpcb_name", ""), match.iloc[0].get("cpcb_sector", "")
        return "", ""

    cpcb_info = df["facility_osm_id"].apply(get_cpcb_info)
    df["cpcb_name"] = cpcb_info.apply(lambda x: x[0])
    df["cpcb_sector"] = cpcb_info.apply(lambda x: x[1])

    # Ensure all required response fields exist and handle NaNs gracefully
    if "ml_prediction" in df.columns:
        df["ml_prediction"] = df["ml_prediction"].fillna(df.get("final_class", "unclassified"))
    else:
        df["ml_prediction"] = df.get("final_class", "unclassified")

    if "ml_confidence" in df.columns:
        df["ml_confidence"] = pd.to_numeric(df["ml_confidence"], errors="coerce").fillna(0.95)
    else:
        df["ml_confidence"] = 0.95

    if "place_description" not in df.columns:
        df["place_description"] = "Rural locality"
    else:
        df["place_description"] = df["place_description"].fillna("Rural locality")

    if "nearest_place_name" not in df.columns:
        df["nearest_place_name"] = ""
    else:
        df["nearest_place_name"] = df["nearest_place_name"].fillna("")

    # Ensure risk_score and risk_level are present (compute on-the-fly if missing)
    if "risk_score" not in df.columns or df["risk_score"].isna().all():
        try:
            from src.compute_risk_scores import compute_risk_scores as _crs
            df = _crs.__wrapped__(df) if hasattr(_crs, '__wrapped__') else df
        except Exception:
            df["risk_score"] = 0.0
            df["risk_level"] = "LOW"
    if "risk_score" in df.columns:
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0.0)
    else:
        df["risk_score"] = 0.0

    if "risk_level" in df.columns:
        df["risk_level"] = df["risk_level"].fillna("LOW")
    else:
        df["risk_level"] = "LOW"
    df["is_persistent_source"] = df["is_persistent_source"] if "is_persistent_source" in df.columns else False

    cols = [
        "latitude", "longitude", "frp", "confidence",
        "acq_date", "nearest_industrial_km", "nearest_industrial_type",
        "nearest_industrial_name", "final_class", "is_anomaly", "facility_osm_id",
        "ml_prediction", "ml_confidence", "is_persistent_source",
        "cpcb_name", "cpcb_sector",
        "nearest_industrial_latitude", "nearest_industrial_longitude",
        "place_description", "nearest_place_name",
        "risk_score", "risk_level"
    ]
    # Only include columns that actually exist in the dataframe
    cols = [c for c in cols if c in df.columns]
    records = df[cols].to_dict(orient="records")

    return jsonify(records)


@app.route("/api/hotspot-clusters")
def get_hotspot_clusters():
    cluster_path = "data/processed/hotspot_clusters.json"
    if os.path.exists(cluster_path):
        with open(cluster_path, "r") as f:
            clusters = json.load(f)
        return jsonify(clusters)
    return jsonify([])


@app.route("/api/last-updated")
def get_last_updated():
    try:
        mtime = os.path.getmtime(DATA_PATH)
        dt = datetime.fromtimestamp(mtime)
        return jsonify({
            "last_updated": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": mtime
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/summary")
def summary_page():
    return render_template("summary.html")


@app.route("/api/summary")
def get_summary():
    df = pd.read_csv(DATA_PATH)

    class_counts = df["final_class"].value_counts().to_dict()

    total = len(df)
    anomalies = int(df["is_anomaly"].sum())
    avg_ml_confidence = float(df["ml_confidence"].mean()) if "ml_confidence" in df.columns else None
    ml_agreement = float(df["ml_agrees_with_rules"].mean() * 100) if "ml_agrees_with_rules" in df.columns else None

    mtime = os.path.getmtime(DATA_PATH)
    last_updated = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

    industrial_df = pd.read_csv("data/raw/osm_industrial.csv")
    forest_df = pd.read_csv("data/raw/osm_forest.csv")
    farmland_df = pd.read_csv("data/raw/osm_farmland.csv")

    return jsonify({
        "total_hotspots": total,
        "class_counts": class_counts,
        "anomaly_count": anomalies,
        "avg_ml_confidence": avg_ml_confidence,
        "ml_agreement_pct": ml_agreement,
        "last_updated": last_updated,
        "industrial_zones_mapped": len(industrial_df),
        "forest_polygons_mapped": len(forest_df),
        "farmland_polygons_mapped": len(farmland_df)
    })


@app.route("/about")
def about_page():
    return render_template("about.html")


@app.route("/facility/<facility_id>")
def facility_detail(facility_id):
    return render_template("facility.html", facility_id=facility_id)


@app.route("/api/facility/<facility_id>")
def get_facility_data(facility_id):
    # Use string comparison throughout — osm_id column has mixed int/string values
    # (e.g. "cpcb_0") due to CPCB enrichment, so pandas reads it as object dtype.
    # An int == string comparison always returns False in pandas.
    facility_id_str = str(facility_id)

    historical = pd.read_csv("data/raw/firms_historical.csv")
    industrial = pd.read_csv("data/raw/osm_industrial.csv")

    industrial_rad = np.radians(industrial[["latitude", "longitude"]].values)
    hist_rad = np.radians(historical[["latitude", "longitude"]].values)

    tree = BallTree(industrial_rad, metric="haversine")
    distances, indices = tree.query(hist_rad, k=1)

    historical["facility_osm_id"] = industrial.iloc[indices.flatten()]["osm_id"].values
    historical["facility_distance_km"] = distances.flatten() * 6371.0

    facility_history = historical[
        (historical["facility_osm_id"].astype(str) == facility_id_str) &
        (historical["facility_distance_km"] <= 2.0)
    ].sort_values("acq_date")

    facility_match = industrial[industrial["osm_id"].astype(str) == facility_id_str]

    if len(facility_match) == 0:
        return jsonify({
            "error": "Facility not found — this may be because live data refreshed since you viewed the map. Please go back to the dashboard and click a currently listed facility.",
            "facility_id": facility_id_str
        }), 404

    facility_info = facility_match.iloc[0]

    baseline = pd.read_csv("data/processed/facility_baselines.csv")
    baseline_row = baseline[baseline["facility_osm_id"].astype(str) == facility_id_str]

    baseline_mean = None
    baseline_std = None
    if len(baseline_row) > 0:
        baseline_mean = float(baseline_row["baseline_frp_mean"].values[0])
        baseline_std = float(baseline_row["baseline_frp_std"].values[0])

    facility_display_name = make_readable_name(
        facility_info["name"], facility_info["type"], facility_info["region"]
    )

    # If CPCB-verified, override display name with the real company name
    if bool(facility_info.get("cpcb_verified", False)):
        facility_display_name = facility_info.get("cpcb_name", facility_display_name)

    # ---- Persistent Thermal Source detection ----
    detection_dates = sorted(set(facility_history["acq_date"].tolist()))

    if len(detection_dates) > 0:
        end_date = pd.to_datetime(detection_dates[-1])
    else:
        end_date = pd.Timestamp.now()

    timeline = []
    for i in range(6, -1, -1):  # last 7 days, oldest first
        day = end_date - pd.Timedelta(days=i)
        day_str = day.strftime("%Y-%m-%d")
        detected = day_str in detection_dates
        timeline.append({"date": day_str, "detected": bool(detected)})

    detected_day_count = sum(1 for t in timeline if t["detected"])
    is_persistent_source = detected_day_count >= 4  # 4+ of last 7 days

    return jsonify({
        "facility_id": facility_id_str,
        "facility_name": facility_display_name,
        "facility_type": facility_info["type"],
        "region": facility_info["region"],
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "history": facility_history[["acq_date", "frp", "acq_time"]].to_dict(orient="records"),
        "activity_timeline": timeline,
        "detected_day_count": detected_day_count,
        "is_persistent_source": is_persistent_source
    })


@app.route("/api/facility/<facility_id>/frp-history")
def get_facility_frp_history(facility_id):
    """
    Returns a daily FRP time-series for a specific facility.
    Used by the Chart.js trend graph in the dashboard popup.
    Response: { dates: [...], frp_values: [...], baseline_mean: float, baseline_std: float }
    """
    facility_id_str = str(facility_id)

    historical = pd.read_csv("data/raw/firms_historical.csv")
    industrial = pd.read_csv("data/raw/osm_industrial.csv")

    industrial_rad = np.radians(industrial[["latitude", "longitude"]].values)
    hist_rad = np.radians(historical[["latitude", "longitude"]].values)
    tree = BallTree(industrial_rad, metric="haversine")
    distances, indices = tree.query(hist_rad, k=1)

    historical["facility_osm_id"] = industrial.iloc[indices.flatten()]["osm_id"].values
    historical["facility_distance_km"] = distances.flatten() * 6371.0

    fac_data = historical[
        (historical["facility_osm_id"].astype(str) == facility_id_str) &
        (historical["facility_distance_km"] <= 2.0)
    ].copy()

    if len(fac_data) == 0:
        return jsonify({"dates": [], "frp_values": [], "baseline_mean": None, "baseline_std": None})

    # Aggregate to daily max FRP
    fac_data["acq_date"] = pd.to_datetime(fac_data["acq_date"])
    daily = fac_data.groupby("acq_date")["frp"].max().reset_index().sort_values("acq_date")

    baseline = pd.read_csv("data/processed/facility_baselines.csv")
    baseline_row = baseline[baseline["facility_osm_id"].astype(str) == facility_id_str]
    baseline_mean = float(baseline_row["baseline_frp_mean"].values[0]) if len(baseline_row) > 0 else None
    baseline_std  = float(baseline_row["baseline_frp_std"].values[0])  if len(baseline_row) > 0 else None

    return jsonify({
        "dates":          daily["acq_date"].dt.strftime("%Y-%m-%d").tolist(),
        "frp_values":     daily["frp"].round(2).tolist(),
        "baseline_mean":  baseline_mean,
        "baseline_std":   baseline_std,
        "facility_id":    facility_id_str
    })


@app.route("/alerts")
def alerts_page():
    return render_template("alerts.html")


@app.route("/api/alerts")
def get_alerts():
    df = pd.read_csv(DATA_PATH)
    df = df.fillna("")

    anomalies = df[df["is_anomaly"] == True].copy()
    anomalies = anomalies.sort_values("frp", ascending=False)

    anomalies["nearest_industrial_name"] = anomalies.apply(
        lambda r: make_readable_name(r["nearest_industrial_name"], r["nearest_industrial_type"], ""),
        axis=1
    )

    records = anomalies[[
        "latitude", "longitude", "frp", "acq_date",
        "nearest_industrial_km", "nearest_industrial_type",
        "nearest_industrial_name", "facility_osm_id"
    ]].to_dict(orient="records")

    return jsonify(records)


@app.route("/api/send-alert", methods=["POST"])
def trigger_alert_email():
    try:
        send_alert_email()
        return jsonify({"status": "success", "message": "Alert email sent successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/send-push-alert", methods=["POST"])
def trigger_alert_push():
    try:
        if not os.path.exists(DATA_PATH):
            return jsonify({"status": "no_data", "message": "No hotspot data available"}), 404
        df = pd.read_csv(DATA_PATH)
        anomalies = df[df["is_anomaly"] == 1]
        if anomalies.empty:
            return jsonify({"status": "no_anomalies", "message": "No active anomalies to push"})
        
        top = anomalies.sort_values(by="frp", ascending=False).iloc[0]
        fname = top.get("nearest_industrial_name") or "Industrial Zone"
        max_frp = round(float(top.get("frp", 0)), 1)
        count = len(anomalies)
        
        title = f"🔥 ALERT: {count} Fire Anomaly{'ies' if count > 1 else ''} Detected!"
        body = f"Peak FRP {max_frp} MW near {fname}. Immediate response advised."
        
        res = send_push_notification(title=title, body=body, url="/alerts", tag="fire-anomaly")
        return jsonify({
            "status": "sent",
            "message": f"Push alert sent to {res.get('sent', 0)} subscriber(s)",
            "details": res,
            "anomaly_count": count
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/refresh-data", methods=["POST"])
def trigger_refresh():
    try:
        refresh_full_pipeline()
        return jsonify({"status": "success", "message": "Data refreshed successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def scheduled_refresh_job():
    print("\n[AUTO] Scheduled refresh triggered...")
    try:
        refresh_full_pipeline()
        print("[AUTO] Scheduled refresh completed successfully.")
    except Exception as e:
        print(f"[AUTO] Scheduled refresh failed: {e}")


# ============================================================
# FEATURE 7: Industrial Facility Compliance Dashboard
# ============================================================

@app.route("/compliance")
def compliance_page():
    return render_template("compliance.html")


@app.route("/api/facilities")
def get_facilities():
    """Returns per-facility compliance stats: fire frequency, avg FRP, risk, last anomaly."""
    industrial = pd.read_csv("data/raw/osm_industrial.csv")
    historical = pd.read_csv("data/raw/firms_historical.csv")
    hotspots   = pd.read_csv(DATA_PATH)
    baseline   = pd.read_csv("data/processed/facility_baselines.csv")

    industrial_rad = np.radians(industrial[["latitude","longitude"]].values)
    hist_rad       = np.radians(historical[["latitude","longitude"]].values)
    tree = BallTree(industrial_rad, metric="haversine")
    dists, idxs = tree.query(hist_rad, k=1)
    historical["facility_osm_id"]   = industrial.iloc[idxs.flatten()]["osm_id"].values
    historical["facility_dist_km"]  = dists.flatten() * 6371.0
    near = historical[historical["facility_dist_km"] <= 2.0].copy()
    near["acq_date"] = pd.to_datetime(near["acq_date"])

    # Per-facility aggregates from historical
    grp = near.groupby("facility_osm_id").agg(
        fire_count    = ("frp",      "count"),
        avg_frp       = ("frp",      "mean"),
        max_frp       = ("frp",      "max"),
        last_detected = ("acq_date", "max"),
        active_days   = ("acq_date", lambda x: x.dt.date.nunique()),
    ).reset_index()
    grp["last_detected"] = grp["last_detected"].dt.strftime("%Y-%m-%d")

    # Merge risk scores from live data
    live_risk = hotspots[["facility_osm_id","risk_score","risk_level"]].copy() if "risk_score" in hotspots.columns else pd.DataFrame()
    live_risk = live_risk.groupby("facility_osm_id").agg(
        risk_score = ("risk_score", "max"),
        risk_level = ("risk_level", "first"),
    ).reset_index() if len(live_risk) > 0 else pd.DataFrame(columns=["facility_osm_id","risk_score","risk_level"])

    merged = grp.merge(industrial[["osm_id","name","type","region"]], left_on="facility_osm_id", right_on="osm_id", how="left")
    merged = merged.merge(live_risk, on="facility_osm_id", how="left")
    merged = merged.merge(baseline[["facility_osm_id","baseline_frp_mean"]], on="facility_osm_id", how="left")

    merged["risk_score"] = pd.to_numeric(merged.get("risk_score", 0), errors="coerce").fillna(0).round(1)
    merged["risk_level"] = merged.get("risk_level", "LOW").fillna("LOW")
    merged["avg_frp"]    = merged["avg_frp"].round(2)
    merged["max_frp"]    = merged["max_frp"].round(2)
    merged["name"]       = merged.apply(lambda r: make_readable_name(r["name"], r["type"], r["region"]), axis=1)

    # Compliance rating: HIGH fire_count + HIGH risk → NON_COMPLIANT
    def compliance_rating(row):
        if row["fire_count"] >= 20 or row["risk_score"] >= 60:
            return "NON_COMPLIANT"
        elif row["fire_count"] >= 5 or row["risk_score"] >= 30:
            return "WATCH"
        else:
            return "COMPLIANT"

    merged["compliance"] = merged.apply(compliance_rating, axis=1)
    merged = merged.sort_values("risk_score", ascending=False)

    records = merged[[
        "facility_osm_id","name","type","region",
        "fire_count","avg_frp","max_frp","last_detected","active_days",
        "risk_score","risk_level","compliance","baseline_frp_mean"
    ]].fillna("").to_dict(orient="records")

    return jsonify(records)


# ============================================================
# FEATURE 4: Wind-Drift Smoke Plume Prediction (Open-Meteo)
# ============================================================

@app.route("/api/wind-data")
def get_wind_data():
    """Fetches current wind speed+direction from Open-Meteo for a lat/lon."""
    import requests as req_lib
    lat = request.args.get("lat", "11.0")
    lon = request.args.get("lon", "78.5")
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=wind_speed_10m,wind_direction_10m,wind_gusts_10m"
            f"&wind_speed_unit=ms&timezone=Asia%2FKolkata"
        )
        r = req_lib.get(url, timeout=8)
        data = r.json()
        cur  = data.get("current", {})
        return jsonify({
            "lat":          float(lat),
            "lon":          float(lon),
            "wind_speed":   cur.get("wind_speed_10m", 0),
            "wind_dir":     cur.get("wind_direction_10m", 0),
            "wind_gusts":   cur.get("wind_gusts_10m", 0),
            "unit":         "m/s",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# FEATURE 9: SHAP Explainability
# ============================================================

@app.route("/api/shap-explanation")
def get_shap_explanation():
    """Returns top feature contributions for a hotspot's ML prediction."""
    import joblib
    try:
        import shap as shap_lib
        SHAP_AVAILABLE = True
    except ImportError:
        SHAP_AVAILABLE = False

    lat = request.args.get("lat")
    lon = request.args.get("lon")
    if not lat or not lon:
        return jsonify({"error": "lat and lon required"}), 400

    try:
        from src.train_classifier import extract_features
        hotspots = pd.read_csv(DATA_PATH)
        hotspots["latitude"]  = pd.to_numeric(hotspots["latitude"],  errors="coerce")
        hotspots["longitude"] = pd.to_numeric(hotspots["longitude"], errors="coerce")

        dist = ((hotspots["latitude"] - float(lat))**2 + (hotspots["longitude"] - float(lon))**2).pow(0.5)
        idx  = dist.idxmin()
        row  = hotspots.iloc[[idx]]
        X    = extract_features(row)

        model_pkg = joblib.load("data/processed/fire_classifier_model.pkl")
        raw_model = model_pkg.get("raw_model") or model_pkg.get("model")

        if SHAP_AVAILABLE and hasattr(raw_model, "estimators_"):
            explainer = shap_lib.TreeExplainer(raw_model)
            sv        = explainer.shap_values(X)
            # For multi-class, pick the predicted class index
            pred_class = raw_model.predict(X)[0]
            classes    = list(raw_model.classes_)
            cls_idx    = classes.index(pred_class) if pred_class in classes else 0
            vals       = sv[cls_idx][0] if isinstance(sv, list) else sv[0]
            feat_names = model_pkg.get("features", X.columns.tolist())
            contribs   = sorted(zip(feat_names, [round(float(v), 4) for v in vals]),
                                key=lambda x: abs(x[1]), reverse=True)[:8]
            return jsonify({
                "prediction": pred_class,
                "contributions": [{"feature": f, "value": float(X[f].iloc[0]) if f in X.columns else 0, "shap": s}
                                  for f, s in contribs],
                "shap_available": True,
            })
        else:
            # Fallback: feature importances from model
            importances = model_pkg.get("feature_importances", {})
            feat_names  = model_pkg.get("features", X.columns.tolist())
            contribs    = sorted([(f, importances.get(f, 0)) for f in feat_names],
                                 key=lambda x: x[1], reverse=True)[:8]
            return jsonify({
                "prediction": str(raw_model.predict(X)[0]) if hasattr(raw_model, "predict") else "unknown",
                "contributions": [{"feature": f, "value": float(X[f].iloc[0]) if f in X.columns else 0, "shap": round(float(s), 4)}
                                  for f, s in contribs],
                "shap_available": False,
                "note": "shap not installed — showing feature importances instead",
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# FEATURE 3: Gemini AI Incident Summary
# ============================================================

@app.route("/api/ai-summary")
def get_ai_summary():
    """Generates a natural-language incident summary using Gemini AI."""
    import os as _os
    from dotenv import load_dotenv
    load_dotenv()
    GEMINI_KEY = _os.getenv("GEMINI_API_KEY", "")

    lat = request.args.get("lat")
    lon = request.args.get("lon")
    if not lat or not lon:
        return jsonify({"error": "lat and lon required"}), 400

    hotspots = pd.read_csv(DATA_PATH)
    hotspots["latitude"]  = pd.to_numeric(hotspots["latitude"],  errors="coerce")
    hotspots["longitude"] = pd.to_numeric(hotspots["longitude"], errors="coerce")
    dist = ((hotspots["latitude"] - float(lat))**2 + (hotspots["longitude"] - float(lon))**2).pow(0.5)
    row  = hotspots.iloc[dist.idxmin()]

    facility  = make_readable_name(row.get("nearest_industrial_name",""), row.get("nearest_industrial_type",""), "")
    frp       = row.get("frp", "N/A")
    acq_date  = row.get("acq_date", "unknown")
    cls       = str(row.get("final_class", "")).replace("_"," ")
    risk_lvl  = row.get("risk_level","LOW")
    anomaly   = "YES" if str(row.get("is_anomaly","False")).lower() in ("true","1") else "NO"
    baseline  = row.get("baseline_frp_mean","N/A")
    km        = row.get("nearest_industrial_km","N/A")
    place     = row.get("place_description", "Tamil Nadu")

    prompt = (
        f"You are an industrial fire monitoring analyst. Write a concise 3-sentence incident report "
        f"for this hotspot detected in Tamil Nadu, India:\n"
        f"- Facility: {facility}\n"
        f"- Location: {place}\n"
        f"- Classification: {cls}\n"
        f"- Date: {acq_date}\n"
        f"- Fire Radiative Power (FRP): {frp} MW\n"
        f"- Distance to nearest industrial zone: {km} km\n"
        f"- Risk Level: {risk_lvl}\n"
        f"- Anomaly (above 2-sigma baseline): {anomaly}\n"
        f"- Historical baseline FRP: {baseline} MW\n"
        f"Be factual and professional. Mention whether authorities should investigate."
    )

    if not GEMINI_KEY:
        # Return a rule-based summary as fallback
        summary = (
            f"A {cls} thermal anomaly with Fire Radiative Power of {frp} MW was detected at "
            f"{facility} on {acq_date}. "
            f"Risk level is classified as {risk_lvl} and the hotspot is located {km} km from the nearest industrial zone. "
            f"{'This event exceeds the 2-sigma FRP baseline — immediate investigation by TNPCB/CPCB is recommended.' if anomaly=='YES' else 'No baseline anomaly threshold breach detected at this time.'}"
        )
        return jsonify({"summary": summary, "source": "rule-based (no GEMINI_API_KEY set)"})

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            result = model.generate_content(prompt)
        except Exception:
            model = genai.GenerativeModel("gemini-2.0-flash")
            result = model.generate_content(prompt)
        return jsonify({"summary": result.text, "source": "gemini"})
    except Exception as e:
        summary = (
            f"A {cls} thermal event with Fire Radiative Power of {frp} MW was detected at "
            f"{facility} on {acq_date}. "
            f"Risk level is assessed as {risk_lvl} ({km} km from nearest industrial sector). "
            f"{'FRP significantly exceeds historical baseline — field verification recommended.' if anomaly=='YES' else 'FRP remains within regular operational levels.'}"
        )
        return jsonify({"summary": summary, "source": f"rule-based fallback ({str(e)[:80]})"})


# ============================================================
# FEATURE 5: WhatsApp / SMS Alert (Twilio)
# ============================================================

@app.route("/api/send-whatsapp", methods=["POST"])
def send_whatsapp_alert():
    """Sends a WhatsApp alert via Twilio for top anomalies."""
    import os as _os
    from dotenv import load_dotenv
    load_dotenv()

    TWILIO_SID   = _os.getenv("TWILIO_ACCOUNT_SID","")
    TWILIO_TOKEN = _os.getenv("TWILIO_AUTH_TOKEN","")
    TWILIO_FROM  = _os.getenv("TWILIO_WHATSAPP_FROM","whatsapp:+14155238886")
    TWILIO_TO    = _os.getenv("TWILIO_WHATSAPP_TO","")

    if not all([TWILIO_SID, TWILIO_TOKEN, TWILIO_TO]):
        return jsonify({
            "status": "not_configured",
            "message": "Add TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_TO to .env"
        }), 200

    try:
        from twilio.rest import Client
        df       = pd.read_csv(DATA_PATH)
        anomalies = df[df["is_anomaly"] == True].sort_values("frp", ascending=False)

        if len(anomalies) == 0:
            return jsonify({"status": "no_anomalies", "message": "No anomalies to report."})

        top = anomalies.iloc[0]
        msg = (
            f"INDUSTRIAL FIRE ALERT - Tamil Nadu\n"
            f"Facility: {make_readable_name(top.get('nearest_industrial_name',''), top.get('nearest_industrial_type',''), '')}\n"
            f"FRP: {top['frp']} MW | Date: {top['acq_date']}\n"
            f"Risk: {top.get('risk_level','?')} | Dist: {float(top['nearest_industrial_km']):.1f} km\n"
            f"Dashboard: http://127.0.0.1:5000/alerts"
        )
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        message = client.messages.create(body=msg, from_=TWILIO_FROM, to=f"whatsapp:{TWILIO_TO}")
        return jsonify({"status": "sent", "sid": message.sid, "anomaly_count": len(anomalies)})
    except ImportError:
        return jsonify({"status": "error", "message": "twilio package not installed. Run: pip install twilio"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================
# FEATURE 10: Multi-Region Support
# ============================================================

REGION_BBOXES = {
    "tamil_nadu":    {"label": "Tamil Nadu",    "bbox": "76.2,8.0,80.4,13.6",   "center": [10.9, 78.5],  "zoom": 7},
    "karnataka":     {"label": "Karnataka",     "bbox": "74.0,11.5,78.6,18.5",  "center": [14.5, 75.7],  "zoom": 7},
    "andhra_pradesh":{"label": "Andhra Pradesh","bbox": "76.7,12.6,84.8,19.9",  "center": [15.9, 79.7],  "zoom": 7},
    "kerala":        {"label": "Kerala",        "bbox": "74.8,8.2,77.6,12.8",   "center": [10.8, 76.3],  "zoom": 8},
    "telangana":     {"label": "Telangana",     "bbox": "77.2,15.7,81.4,19.9",  "center": [17.9, 79.3],  "zoom": 7},
}

@app.route("/api/regions")
def get_regions():
    return jsonify(list(REGION_BBOXES.values()) + [{"label": r["label"], "bbox": r["bbox"], "center": r["center"], "zoom": r["zoom"], "key": k} for k, r in REGION_BBOXES.items()])


@app.route("/api/firms-region")
def get_firms_region():
    """Fetch FIRMS live hotspots for any region on-the-fly (cached 1h per region)."""
    import requests as req_lib, hashlib, time
    region_key = request.args.get("region", "tamil_nadu")
    region     = REGION_BBOXES.get(region_key)
    if not region:
        return jsonify({"error": f"Unknown region: {region_key}"}), 400

    from dotenv import load_dotenv; load_dotenv()
    API_KEY = os.getenv("FIRMS_API_KEY","")
    if not API_KEY:
        return jsonify({"error": "FIRMS_API_KEY not set"}), 500

    cache_path = f"data/processed/region_cache_{region_key}.json"
    # Use cached result if < 1 hour old
    if os.path.exists(cache_path) and (time.time() - os.path.getmtime(cache_path)) < 3600:
        with open(cache_path) as f:
            return jsonify(json.load(f))

    bbox    = region["bbox"]
    sources = ["VIIRS_NOAA20_NRT","VIIRS_NOAA21_NRT","VIIRS_SNPP_NRT"]
    all_pts = []
    for src in sources:
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{API_KEY}/{src}/{bbox}/3"
        try:
            r   = req_lib.get(url, timeout=15)
            from io import StringIO
            df  = pd.read_csv(StringIO(r.text))
            if "latitude" in df.columns and len(df) > 0:
                all_pts.append(df[["latitude","longitude","frp","acq_date","confidence"]].fillna(""))
        except Exception:
            pass

    if not all_pts:
        return jsonify([])

    combined = pd.concat(all_pts).drop_duplicates(["latitude","longitude","acq_date"]).to_dict(orient="records")
    with open(cache_path, "w") as f:
        json.dump(combined, f)
    return jsonify(combined)


# ============================================================
# FEATURE 6: Sentinel-2 & MODIS Imagery Links
# ============================================================

@app.route("/api/satellite-links")
def get_satellite_links():
    """Returns Sentinel-2, MODIS, and Google Earth Engine links for a lat/lon."""
    lat = request.args.get("lat")
    lon = request.args.get("lon")
    if not lat or not lon:
        return jsonify({"error": "lat and lon required"}), 400

    lat_f, lon_f = float(lat), float(lon)
    delta = 0.05  # ~5km box

    # Copernicus Open Access Hub (Sentinel-2)
    sentinel_url = (
        f"https://dataspace.copernicus.eu/browser/?zoom=13"
        f"&lat={lat_f}&lng={lon_f}"
        f"&themeId=DEFAULT-THEME&visualizationUrl=https%3A%2F%2Fsh.dataspace.copernicus.eu%2Fogc%2Fwms%2Fa91f72b5-f393-4320-bc0f-990129bd9e63"
        f"&evalscript=&datasetId=S2_L2A_CDAS&demSource3D=%22MAPZEN%22"
    )

    # NASA FIRMS fire map
    firms_url = (
        f"https://firms.modaps.eosdis.nasa.gov/map/#d:24hrs;@{lon_f},{lat_f},12z"
    )

    # NASA Worldview (MODIS true color)
    worldview_url = (
        f"https://worldview.earthdata.nasa.gov/?v="
        f"{lon_f-delta},{lat_f-delta},{lon_f+delta},{lat_f+delta}"
        f"&l=VIIRS_NOAA20_Thermal_Anomalies_375m_Day,MODIS_Terra_CorrectedReflectance_TrueColor"
    )

    # Google Earth Engine Timelapse
    gee_url = f"https://earthengine.google.com/timelapse#v={lat_f},{lon_f},13,latLng"

    # Sentinel Hub EO Browser (quick-look)
    eobrowser_url = (
        f"https://apps.sentinel-hub.com/eo-browser/?zoom=14&lat={lat_f}&lng={lon_f}"
        f"&themeId=DEFAULT-THEME&datasetId=S2L2A"
    )

    return jsonify({
        "lat": lat_f, "lon": lon_f,
        "links": {
            "sentinel2_copernicus": {"label": "Sentinel-2 (Copernicus Browser)", "url": sentinel_url},
            "sentinel2_eobrowser":  {"label": "Sentinel-2 (EO Browser)",         "url": eobrowser_url},
            "nasa_worldview":       {"label": "NASA Worldview (MODIS)",            "url": worldview_url},
            "nasa_firms_map":       {"label": "NASA FIRMS Fire Map",               "url": firms_url},
            "google_earth_timelapse":{"label": "Google Earth Timelapse",           "url": gee_url},
        }
    })


# ============================================================
# FEATURE 8: PWA & Web Push Notification Endpoints
# ============================================================

@app.route("/manifest.json")
def pwa_manifest():
    return send_from_directory("static", "manifest.json", mimetype="application/manifest+json")


@app.route("/sw.js")
def service_worker():
    response = make_response(send_from_directory("static", "sw.js", mimetype="application/javascript"))
    # Allow service worker at /sw.js to control all paths under root
    response.headers["Service-Worker-Allowed"] = "/"
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


@app.route("/offline")
def offline_page():
    return render_template("offline.html")


@app.route("/api/push/vapid-public-key", methods=["GET"])
def push_vapid_public_key():
    try:
        pub_key = get_public_key()
        return jsonify({"publicKey": pub_key})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/push/subscribe", methods=["POST"])
def push_subscribe():
    try:
        data = request.get_json(force=True, silent=True) or {}
        if not data.get("endpoint"):
            return jsonify({"error": "Missing subscription endpoint"}), 400
            
        success = add_subscription(data)
        subs = load_subscriptions()
        return jsonify({
            "status": "subscribed" if success else "error",
            "active_subscriptions": len(subs)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/push/unsubscribe", methods=["POST"])
def push_unsubscribe():
    try:
        data = request.get_json(force=True, silent=True) or {}
        endpoint = data.get("endpoint")
        if endpoint:
            remove_subscription(endpoint)
        return jsonify({"status": "unsubscribed"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/push/status", methods=["GET"])
def push_status():
    try:
        subs = load_subscriptions()
        return jsonify({
            "active_subscribers": len(subs),
            "vapid_configured": bool(get_public_key())
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/push/send-test", methods=["POST"])
def push_send_test():
    try:
        data = request.get_json(force=True, silent=True) or {}
        title = data.get("title", "🚨 Fire Anomaly Alert")
        body = data.get("body", "Real-time thermal anomaly detected by satellite sensors.")
        url = data.get("url", "/dashboard")
        
        result = send_push_notification(title=title, body=body, url=url)
        return jsonify({
            "status": "success",
            "result": result
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_refresh_job, "interval", minutes=5)
scheduler.start()


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)