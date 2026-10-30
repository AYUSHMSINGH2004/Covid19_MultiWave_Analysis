"""
run_all_modules.py – Master Runner
====================================
Executes all modules (2-7) sequentially, generating all plots,
metrics, and evaluation outputs in a single stretch.

Usage:
    python run_all_modules.py
"""

import subprocess
import sys
import os
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")

modules = [
    ("Module 2: Data Transformation",              "module2_data_transformation.py"),
    ("Module 3: Correlation & Time Series Analysis","module3_correlation_timeseries.py"),
    ("Module 4: Data Summarization & Visualization","module4_summarization_visualization.py"),
    ("Module 5: Clustering Algorithms",             "module5_clustering.py"),
    ("Module 6: Dimensionality Reduction",          "module6_dimensionality_reduction.py"),
    ("Module 7: Model Development & Evaluation",    "module7_model_evaluation.py"),
]

# Optional CLI filter (e.g. python run_all_modules.py 2, or python run_all_modules.py 2 3)
if len(sys.argv) > 1:
    selected_nums = [int(arg) for arg in sys.argv[1:] if arg.isdigit()]
    if selected_nums:
        modules = [m for m in modules if any(f"Module {num}:" in m[0] for num in selected_nums)]

print("=" * 80)
print("COVID-19 MULTI-WAVE ANALYSIS -- MODULE EXECUTION PIPELINE")
print("=" * 80)
print()

total_start = time.time()
results = []

child_env = os.environ.copy()
child_env["PYTHONIOENCODING"] = "utf-8"

def resolve_script(script_name):
    candidate = os.path.join(BASE_DIR, "modules", script_name)
    if os.path.exists(candidate):
        return candidate
    return os.path.join(BASE_DIR, script_name)

for i, (title, script) in enumerate(modules, 1):
    print(f"\n>>> Running {title} [{i}/{len(modules)}]")
    start = time.time()
    script_path = resolve_script(script)

    result = subprocess.run(
        [PYTHON, script_path],
        cwd=BASE_DIR,
        capture_output=False,
        env=child_env
    )

    elapsed = time.time() - start
    status = "[PASSED]" if result.returncode == 0 else "[FAILED]"
    results.append((title, status, elapsed))
    print(f"\n  {status} in {elapsed:.1f}s")

total_elapsed = time.time() - total_start

print("\n\n" + "=" * 80)
print("EXECUTION SUMMARY")
print("=" * 80)
for title, status, elapsed in results:
    print(f"  {status}  {title:<50s} ({elapsed:.1f}s)")
print(f"\n  Total time: {total_elapsed:.1f}s")

# List generated plots
plots_dir = os.path.join(BASE_DIR, "plots")
if os.path.exists(plots_dir):
    plots = sorted([f for f in os.listdir(plots_dir) if f.endswith(".png")])
    print(f"\n  Generated plots ({len(plots)} files in plots/):")
    for p in plots:
        size = os.path.getsize(os.path.join(plots_dir, p))
        print(f"    * {p} ({size / 1024:.0f} KB)")

print("\n" + "=" * 80)
print("ALL MODULES COMPLETE")
print("=" * 80)
