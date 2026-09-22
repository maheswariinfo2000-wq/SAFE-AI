import subprocess
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.send_alerts import send_alert_email

def run_step(script_path, step_name):
    print(f"\n--- Running: {step_name} ---")
    full_path = os.path.join(PROJECT_ROOT, script_path) if not os.path.isabs(script_path) else script_path
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, full_path],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True,
        env=env
    )
    if result.stdout:
        print(result.stdout[-500:])
    if result.returncode != 0:
        print(f"ERROR in {step_name}:")
        if result.stderr:
            print(result.stderr[-500:])
        raise RuntimeError(f"{step_name} failed: {result.stderr.strip() or 'Unknown error'}")
    return result.stdout

def refresh_full_pipeline():
    run_step("src/fetch_firms.py", "Fetch latest FIRMS data")
    run_step("src/fuse_data.py", "Fuse with industrial zones")
    run_step("src/final_classification.py", "Land-cover classification")
    run_step("src/anomaly_detection.py", "Anomaly detection")
    run_step("src/apply_ml_predictions.py", "Applying ML predictions")
    run_step("src/compute_risk_scores.py", "Computing risk scores")

    print("\n--- Checking for new fire/anomaly alerts ---")
    try:
        send_alert_email()
    except Exception as e:
        print(f"Alert email step failed (non-critical): {e}")

    print("\n[OK] Pipeline refresh complete!")

if __name__ == "__main__":
    refresh_full_pipeline()