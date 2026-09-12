"""Aggregate results/runs.parquet: mean ± 95% CI over seeds per arm. Usage: python -m fsf.report [runs.parquet]"""
import sys, pandas as pd, numpy as np
df = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else "/workspace/fsf/results/runs.parquet")
cols = ["test.auc_pr", "test.auc_pr_persistence", "test.auc_pr_growth_region", "test.auc_pr_growth_region_persistence", "test.ece", "test.f1_best", "best_eval_auc_pr", "epochs_run", "wall_s"]
def ci(s): return f"{s.mean():.4f} ± {1.96*s.std(ddof=1)/np.sqrt(len(s)) if len(s)>1 else 0:.4f}"
g = df.groupby("name"); out = pd.DataFrame({c: g[c].apply(ci) for c in cols}); out["n_seeds"] = g.size()
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 20); print(out.T.to_string())
