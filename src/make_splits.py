"""Make task-level train/validation/test split files (60/20/20, grouped by Task_ID).

Usage: python make_splits.py <synthetic_data.csv> <output_dir> [seed ...]
Default seeds: 26 (main result) 27 28 29 30 (repeats for robustness).
Writes split_assignments_seed<N>.csv with columns Task_ID,Set (Set = train/validation/test).
All 20 runs of a task go to the same set, so test tasks are never seen in training.
"""
import os
import sys

import numpy as np
import pandas as pd

data_path, out_dir = sys.argv[1:3]
seeds = [int(s) for s in sys.argv[3:]] or [26, 27, 28, 29, 30]

tasks = sorted(pd.read_csv(data_path)["Task_ID"].unique(), key=lambda t: int(t.split("-")[1]))
n = len(tasks)
n_train, n_val = round(0.6 * n), round(0.2 * n)
os.makedirs(out_dir, exist_ok=True)

for seed in seeds:
    order = np.random.default_rng(seed).permutation(tasks)
    sets = ["train"] * n_train + ["validation"] * n_val + ["test"] * (n - n_train - n_val)
    split = pd.DataFrame({"Task_ID": order, "Set": sets})
    split["_n"] = split["Task_ID"].str.split("-").str[1].astype(int)
    split = split.sort_values("_n").drop(columns="_n")
    path = os.path.join(out_dir, f"split_assignments_seed{seed}.csv")
    split.to_csv(path, index=False)
    print(f"seed {seed}: {split['Set'].value_counts().to_dict()} -> {path}")
