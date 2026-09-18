import subprocess
import sys
from src.send_alerts import send_alert_email

def run_step(script_path, step_name):
    print(f"\n--- Running: {step_name} ---")
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True
    )
    print(result.stdout[-500:])
    if result.returncode != 0:
        print(f"ERROR in {step_name}:")
        print(result.stderr[-500:])
        raise RuntimeError(f"{step_name} failed")
    return result.stdout

def refresh_full_pipeline():
    run_step("src/fetch_firms.py", "Fetch latest FIRMS data")
    run_step("src/fuse_data.py", "Fuse with industrial zones")
    run_step("src/final_classification.py", "Land-cover classification")
    run_step("src/anomaly_detection.py", "Anomaly detection")
    run_step("src/apply_ml_predictions.py", "Applying ML predictions")   # <-- NEW STEP

    print("\n--- Checking for new fire/anomaly alerts ---")
    try:
        send_alert_email()
    except Exception as e:
        print(f"Alert email step failed (non-critical): {e}")

    print("\n✅ Pipeline refresh complete!")

if __name__ == "__main__":
    refresh_full_pipeline()