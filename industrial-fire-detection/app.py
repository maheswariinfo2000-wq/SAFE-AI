from flask import Flask, jsonify, render_template
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.neighbors import BallTree
from apscheduler.schedulers.background import BackgroundScheduler
from src.send_alerts import send_alert_email
from src.refresh_pipeline import refresh_full_pipeline

app = Flask(__name__)

DATA_PATH = "data/processed/hotspots_with_anomalies.csv"


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
    df = pd.read_csv(DATA_PATH)
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

    records = df[[
        "latitude", "longitude", "frp", "confidence",
        "acq_date", "nearest_industrial_km", "nearest_industrial_type",
        "nearest_industrial_name", "final_class", "is_anomaly", "facility_osm_id",
        "ml_prediction", "ml_confidence", "is_persistent_source"
    ]].to_dict(orient="records")

    return jsonify(records)


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
    facility_id_int = int(facility_id)

    historical = pd.read_csv("data/raw/firms_historical.csv")
    industrial = pd.read_csv("data/raw/osm_industrial.csv")

    industrial_rad = np.radians(industrial[["latitude", "longitude"]].values)
    hist_rad = np.radians(historical[["latitude", "longitude"]].values)

    tree = BallTree(industrial_rad, metric="haversine")
    distances, indices = tree.query(hist_rad, k=1)

    historical["facility_osm_id"] = industrial.iloc[indices.flatten()]["osm_id"].values
    historical["facility_distance_km"] = distances.flatten() * 6371.0

    facility_history = historical[
        (historical["facility_osm_id"] == facility_id_int) &
        (historical["facility_distance_km"] <= 2.0)
    ].sort_values("acq_date")

    facility_info = industrial[industrial["osm_id"] == facility_id_int].iloc[0]

    baseline = pd.read_csv("data/processed/facility_baselines.csv")
    baseline_row = baseline[baseline["facility_osm_id"] == facility_id_int]

    baseline_mean = None
    baseline_std = None
    if len(baseline_row) > 0:
        baseline_mean = float(baseline_row["baseline_frp_mean"].values[0])
        baseline_std = float(baseline_row["baseline_frp_std"].values[0])

    facility_display_name = make_readable_name(
        facility_info["name"], facility_info["type"], facility_info["region"]
    )

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
        "facility_id": facility_id_int,
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


@app.route("/api/refresh-data", methods=["POST"])
def trigger_refresh():
    try:
        refresh_full_pipeline()
        return jsonify({"status": "success", "message": "Data refreshed successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def scheduled_refresh_job():
    print("\n🔄 [AUTO] Scheduled refresh triggered...")
    try:
        refresh_full_pipeline()
        print("✅ [AUTO] Scheduled refresh completed successfully.")
    except Exception as e:
        print(f"❌ [AUTO] Scheduled refresh failed: {e}")


scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_refresh_job, "interval", minutes=5)
scheduler.start()


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)