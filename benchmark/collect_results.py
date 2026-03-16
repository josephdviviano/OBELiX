"""
Parse all benchmark results (TRM + sklearn) from logs and JSONs into DataFrames.
Outputs:
  - benchmark/results_summary.csv   (one row per model x feature_set x run)
  - benchmark/results_training.csv  (per-epoch training curves)
  - benchmark/optuna_all.csv        (all Optuna trials, both backbones)
  - benchmark/results_folds.csv     (per-fold CV results)
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
TRM = ROOT / "TRM"

# ---- regex patterns ----
CV_RE = re.compile(r"CV result:\s*([\d.]+)\s*\u00b1\s*([\d.]+)")
BEST_SKLEARN_RE = re.compile(r"Best (?:RF|MLP) CV MAE:\s*([\d.]+)\s*\u00b1\s*([\d.]+)")
TEST_RE = re.compile(r"Test MAE.*?:\s*([\d.]+)")
TEST_CIF_RE = re.compile(r"Test MAE \(CIF-only.*?:\s*([\d.]+)")
BEST_EPOCH_RE = re.compile(r"best epoch:\s*(\d+)\s+best val_mae:\s*([\d.]+)")
EPOCH_RE = re.compile(
    r"epoch\s+(\d+)\s+train_mae=([\d.]+)\s+val_mae=([\d.]+)(?:\s+lr=([\d.eE+-]+))?"
)
FOLD_RE = re.compile(r"Fold (\d+)/(\d+)\s*\n\s*val_mae\s*=\s*([\d.]+)")
EARLY_STOP_RE = re.compile(r"early stopping at epoch (\d+)")
CONFIG_RE = re.compile(r"Config:\s*backbone=(\w+),\s*hidden=(\d+),\s*heads=(\d+),\s*L_layers=(\d+),\s*L_cycles=(\d+),\s*H_cycles=(\d+)")


def split_sections(text):
    """Split log text into sections by === HEADER === lines.
    Returns list of (header_str, body_str) tuples where body is everything
    from this header to the next header.
    Skips lines that are purely = signs (separator lines)."""
    header_re = re.compile(r"^===\s*(.+?)\s*===$", re.MULTILINE)
    matches = [m for m in header_re.finditer(text)
               if not re.fullmatch(r"=+", m.group(1).strip())]
    sections = []
    for i, m in enumerate(matches):
        header = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        sections.append((header, body))
    return sections


# ---------------------------------------------------------------------------
# 1. Summary table
# ---------------------------------------------------------------------------

def parse_log_summary(text, model_map, run_label):
    """Parse a log file into summary rows.
    model_map: dict mapping header keywords to (model_name, features) tuples.
    Returns list of dicts."""
    sections = split_sections(text)

    # First pass: collect CV results and test results by (model, features) key
    cv_results = {}   # key -> {cv_mae, cv_std}
    test_results = {} # key -> {test_mae, test_mae_cif}

    for header, body in sections:
        # Determine which model/features this section is for
        key = None
        for kw, (model, feat) in model_map.items():
            if kw in header:
                key = (model, feat)
                break
        if key is None:
            continue

        # CV results
        cv_m = CV_RE.search(body)
        sklearn_m = BEST_SKLEARN_RE.search(body)
        if cv_m:
            cv_results[key] = {"cv_mae": float(cv_m.group(1)), "cv_std": float(cv_m.group(2))}
        elif sklearn_m:
            cv_results[key] = {"cv_mae": float(sklearn_m.group(1)), "cv_std": float(sklearn_m.group(2))}

        # Test results
        test_m = TEST_RE.search(body)
        test_cif_m = TEST_CIF_RE.search(body)
        if test_m:
            if key not in test_results:
                test_results[key] = {}
            # For sklearn log: "Test MAE (Whole dataset)" and "Test MAE (CIF-only)"
            # For TRM log: "Test MAE (log10 scale)" and "Test MAE (CIF-only, log10 scale)"
            if "CIF-only" not in test_m.group(0) and "Whole" not in test_m.group(0):
                test_results[key]["test_mae"] = float(test_m.group(1))
            elif "Whole" in test_m.group(0):
                test_results[key]["test_mae"] = float(test_m.group(1))
        if test_cif_m:
            if key not in test_results:
                test_results[key] = {}
            test_results[key]["test_mae_cif"] = float(test_cif_m.group(1))

        # Also handle sklearn format: two Test MAE lines in same block
        all_test = re.findall(r"Test MAE \(([^)]+)\):\s*([\d.]+)", body)
        for label, val in all_test:
            if key not in test_results:
                test_results[key] = {}
            if "CIF" in label:
                test_results[key]["test_mae_cif"] = float(val)
            else:
                test_results[key]["test_mae"] = float(val)

    # Merge CV + test
    all_keys = set(cv_results.keys()) | set(test_results.keys())
    rows = []
    for key in all_keys:
        model, feat = key
        row = {"model": model, "features": feat, "run": run_label}
        row.update(cv_results.get(key, {}))
        row.update(test_results.get(key, {}))
        rows.append(row)
    return rows


summary_rows = []

# --- TRM best run (job 8952301) ---
best_text = (TRM / "best_8952301.log").read_text()
summary_rows.extend(parse_log_summary(best_text, {
    "Transformer (full dataset": ("TRM-Transformer", "full"),
    "MLP (full dataset": ("TRM-MLP", "full"),
    "Transformer (CIF-only": ("TRM-Transformer", "cif"),
    "MLP (CIF-only": ("TRM-MLP", "cif"),
}, "tuned"))

# --- TRM default-hparam runs ---
for log_file, run_label in [(TRM / "trm_8923922.log", "default_v1"),
                             (TRM / "trm_8943936.log", "default_v2")]:
    if not log_file.exists():
        continue
    text = log_file.read_text()
    summary_rows.extend(parse_log_summary(text, {
        "Transformer backbone": ("TRM-Transformer", "full"),
        "MLP backbone": ("TRM-MLP", "full"),
    }, run_label))

# --- Sklearn baselines (job 8951484) ---
sklearn_log = (ROOT.parent / "sklearn_8951484.log").read_text()
summary_rows.extend(parse_log_summary(sklearn_log, {
    "RF (Full": ("RF", "full"),
    "RF (CIF": ("RF", "cif"),
    "MLP (Full": ("sklearn-MLP", "full"),
    "MLP (CIF": ("sklearn-MLP", "cif"),
}, "tuned"))

df_summary = pd.DataFrame(summary_rows)
# Deduplicate: keep first occurrence per (model, features, run)
df_summary = df_summary.groupby(["model", "features", "run"], as_index=False).first()
print("=== Summary ===")
print(df_summary.to_string(index=False))
df_summary.to_csv(ROOT / "results_summary.csv", index=False)
print(f"\nSaved to {ROOT / 'results_summary.csv'}")


# ---------------------------------------------------------------------------
# 2. Training curves
# ---------------------------------------------------------------------------

def parse_training_curves(text, model, features, run):
    rows = []
    for line in text.splitlines():
        m = EPOCH_RE.search(line)
        if m:
            rows.append({
                "model": model, "features": features, "run": run,
                "epoch": int(m.group(1)),
                "train_mae": float(m.group(2)),
                "val_mae": float(m.group(3)),
                "lr": float(m.group(4)) if m.group(4) else None,
            })
        bm = BEST_EPOCH_RE.search(line)
        if bm and rows:
            rows[-1]["is_best_epoch"] = True
            rows[-1]["best_val_mae"] = float(bm.group(2))
        es = EARLY_STOP_RE.search(line)
        if es and rows:
            rows[-1]["early_stopped"] = True
    return rows


def get_section_body(text, header_substring):
    """Get the body of a section whose header contains header_substring."""
    for header, body in split_sections(text):
        if header_substring in header:
            return body
    return None


training_rows = []

# TRM best run (8952301)
for header_sub, model_name, feat in [
    ("Transformer (full dataset, train/test)", "TRM-Transformer", "full"),
    ("MLP (full dataset, train/test)", "TRM-MLP", "full"),
    ("Transformer (CIF-only, train/test)", "TRM-Transformer", "cif"),
    ("MLP (CIF-only, train/test)", "TRM-MLP", "cif"),
]:
    body = get_section_body(best_text, header_sub)
    if body:
        training_rows.extend(parse_training_curves(body, model_name, feat, "tuned"))

# TRM default runs
for log_file, run_label in [(TRM / "trm_8923922.log", "default_v1"),
                             (TRM / "trm_8943936.log", "default_v2")]:
    if not log_file.exists():
        continue
    text = log_file.read_text()
    for header_sub, model_name in [
        ("Transformer backbone (train/test", "TRM-Transformer"),
        ("MLP backbone (train/test", "TRM-MLP"),
    ]:
        body = get_section_body(text, header_sub)
        if body:
            training_rows.extend(parse_training_curves(body, model_name, "full", run_label))

df_training = pd.DataFrame(training_rows)
if not df_training.empty:
    print(f"\n=== Training curves: {len(df_training)} epoch entries ===")
    print(df_training.groupby(["model", "features", "run"]).agg(
        epochs=("epoch", "max"),
        best_val=("val_mae", "min"),
    ).to_string())
    df_training.to_csv(ROOT / "results_training.csv", index=False)
    print(f"\nSaved to {ROOT / 'results_training.csv'}")
else:
    print("\n(no training curve data found)")


# ---------------------------------------------------------------------------
# 3. Optuna trials
# ---------------------------------------------------------------------------

optuna_dfs = []
for csv_file in sorted(TRM.glob("optuna_*.csv")):
    backbone = csv_file.stem.replace("optuna_", "")
    df = pd.read_csv(csv_file)
    df.insert(0, "backbone", backbone)
    optuna_dfs.append(df)

if optuna_dfs:
    df_optuna = pd.concat(optuna_dfs, ignore_index=True)
    print(f"\n=== Optuna trials: {len(df_optuna)} total ===")
    print(df_optuna.groupby(["backbone", "state"]).agg(
        n=("number", "count"),
        best_cv_mae=("value", "min"),
    ).to_string())
    df_optuna.to_csv(ROOT / "optuna_all.csv", index=False)
    print(f"\nSaved to {ROOT / 'optuna_all.csv'}")


# ---------------------------------------------------------------------------
# 4. Per-fold CV results
# ---------------------------------------------------------------------------

fold_rows = []

# TRM best run
for header_sub, model_name, feat in [
    ("Transformer (full dataset, 5-fold CV)", "TRM-Transformer", "full"),
    ("MLP (full dataset, 5-fold CV)", "TRM-MLP", "full"),
    ("Transformer (CIF-only, 5-fold CV)", "TRM-Transformer", "cif"),
    ("MLP (CIF-only, 5-fold CV)", "TRM-MLP", "cif"),
]:
    body = get_section_body(best_text, header_sub)
    if body:
        for m in FOLD_RE.finditer(body):
            fold_rows.append({
                "model": model_name, "features": feat, "run": "tuned",
                "fold": int(m.group(1)), "n_folds": int(m.group(2)),
                "val_mae": float(m.group(3)),
            })

# TRM default runs
for log_file, run_label in [(TRM / "trm_8923922.log", "default_v1"),
                             (TRM / "trm_8943936.log", "default_v2")]:
    if not log_file.exists():
        continue
    text = log_file.read_text()
    for header_sub, model_name in [
        ("Transformer backbone (5-fold CV)", "TRM-Transformer"),
        ("MLP backbone (5-fold CV)", "TRM-MLP"),
    ]:
        body = get_section_body(text, header_sub)
        if body:
            for m in FOLD_RE.finditer(body):
                fold_rows.append({
                    "model": model_name, "features": "full", "run": run_label,
                    "fold": int(m.group(1)), "n_folds": int(m.group(2)),
                    "val_mae": float(m.group(3)),
                })

if fold_rows:
    df_folds = pd.DataFrame(fold_rows)
    print(f"\n=== Per-fold CV results: {len(df_folds)} entries ===")
    print(df_folds.to_string(index=False))
    df_folds.to_csv(ROOT / "results_folds.csv", index=False)
    print(f"\nSaved to {ROOT / 'results_folds.csv'}")


print("\n\n=== All done! ===")
print("Output files:")
for f in ["results_summary.csv", "results_training.csv", "optuna_all.csv", "results_folds.csv"]:
    p = ROOT / f
    if p.exists():
        print(f"  {p}  ({p.stat().st_size} bytes)")
