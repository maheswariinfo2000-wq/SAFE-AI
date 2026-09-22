"""
System health check script for the Industrial Fire Detection pipeline.
Verifies syntax, imports, model integrity, and data integrity.
"""
import sys, os, ast, joblib, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

PASS = "[OK]   "
FAIL = "[FAIL] "
WARN = "[WARN] "
errors = []

def check(label, condition, detail=""):
    if condition:
        print(f"{PASS} {label}")
    else:
        print(f"{FAIL} {label}" + (f": {detail}" if detail else ""))
        errors.append(label)

print("=" * 60)
print("  FIRE DETECTION SYSTEM HEALTH CHECK")
print("=" * 60)

# 1. Syntax check all source files
print("\n[1] Syntax Check")
src_files = []
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in ["venv", "__pycache__", ".git"]]
    for f in files:
        if f.endswith(".py") and f != "check_system.py":
            src_files.append(os.path.join(root, f))

syntax_ok = 0
for fpath in sorted(src_files):
    try:
        with open(fpath, "r", encoding="utf-8") as fh:
            ast.parse(fh.read())
        syntax_ok += 1
    except SyntaxError as e:
        print(f"{FAIL} {fpath}: SyntaxError line {e.lineno}: {e.msg}")
        errors.append(fpath)
print(f"{PASS} {syntax_ok}/{len(src_files)} files have valid syntax")

# 2. Import check
print("\n[2] Import Check")
import_modules = [
    "src.train_classifier",
    "src.apply_ml_predictions",
    "src.refresh_pipeline",
    "src.final_classification",
    "src.fetch_firms",
    "src.fuse_data",
    "src.anomaly_detection",
]
import importlib
for mod in import_modules:
    try:
        importlib.import_module(mod)
        print(f"{PASS} {mod}")
    except Exception as e:
        print(f"{FAIL} {mod}: {e}")
        errors.append(mod)

# 3. Model integrity
print("\n[3] Model Integrity")
model_path = "data/processed/fire_classifier_model.pkl"
if os.path.exists(model_path):
    loaded = joblib.load(model_path)
    from src.train_classifier import FEATURE_COLS
    check("Model bundle has 'model' key", "model" in loaded)
    check("Model features match FEATURE_COLS", loaded.get("features") == FEATURE_COLS,
          f"Expected {len(FEATURE_COLS)}, got {len(loaded.get('features', []))}")
    check("Model classes are 5", len(loaded.get("classes", [])) == 5,
          str(loaded.get("classes")))
    acc = loaded.get("accuracy", 0)
    check(f"CV accuracy >= 99% (actual: {acc:.4f})", acc >= 0.99)
    oof = loaded.get("oof_accuracy", acc)
    check(f"OOF accuracy >= 99% (actual: {oof:.4f})", oof >= 0.99)
else:
    print(f"{FAIL} Model file not found: {model_path}")
    errors.append("model_file")

# 4. Data integrity
print("\n[4] Data Integrity")
live_path = "data/processed/hotspots_with_anomalies.csv"
hist_path = "data/processed/all_tamilnadu_hotspots.csv"

for path, label in [(live_path, "Live"), (hist_path, "Historical")]:
    if not os.path.exists(path):
        print(f"{FAIL} {label} data missing: {path}")
        errors.append(path)
        continue
    df = pd.read_csv(path)
    check(f"{label}: ml_prediction column present", "ml_prediction" in df.columns)
    check(f"{label}: ml_confidence column present", "ml_confidence" in df.columns)
    if "ml_prediction" in df.columns:
        null_preds = df["ml_prediction"].isna().sum()
        check(f"{label}: No null ml_prediction ({len(df)} rows)", null_preds == 0,
              f"{null_preds} nulls found")
    if "ml_confidence" in df.columns:
        df["ml_confidence"] = pd.to_numeric(df["ml_confidence"], errors="coerce")
        null_conf = df["ml_confidence"].isna().sum()
        zero_conf = (df["ml_confidence"] <= 0).sum()
        check(f"{label}: No invalid ml_confidence", null_conf + zero_conf == 0,
              f"{null_conf} null, {zero_conf} zero")
        avg_conf = df["ml_confidence"].mean()
        check(f"{label}: Avg confidence >= 97% ({avg_conf:.3f})", avg_conf >= 0.97)
    if "final_class" in df.columns:
        unclassified = (df["final_class"] == "unclassified").sum()
        check(f"{label}: Zero unclassified hotspots", unclassified == 0,
              f"{unclassified} unclassified found")

# 5. API endpoints (if server is running)
print("\n[5] API Endpoint Check")
try:
    import urllib.request, json
    for ep, expected_count in [
        ("live", 166),
        ("all", 1597),
    ]:
        url = f"http://127.0.0.1:5000/api/hotspots?dataset={ep}"
        req = urllib.request.urlopen(url, timeout=3)
        data = json.loads(req.read().decode("utf-8"))
        check(f"/api/hotspots?dataset={ep} returns {expected_count} items", len(data) == expected_count,
              f"Got {len(data)}")

    req = urllib.request.urlopen("http://127.0.0.1:5000/api/hotspot-clusters", timeout=3)
    clusters = json.loads(req.read().decode("utf-8"))
    check(f"/api/hotspot-clusters returns 123 items", len(clusters) == 123,
          f"Got {len(clusters)}")
except Exception as e:
    print(f"{WARN} Server not reachable (may be stopped): {e}")

# Summary
print("\n" + "=" * 60)
if errors:
    print(f"HEALTH CHECK FAILED: {len(errors)} issue(s) found.")
    for e in errors:
        print(f"  - {e}")
else:
    print("HEALTH CHECK PASSED: All systems operational.")
print("=" * 60)
