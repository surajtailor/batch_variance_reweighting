import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from loss_functions import loss_rmse, loss_nll
from models import MVENetwork
from sklearn.model_selection import KFold
from training_functions import LossConfig
import csv
import sys

# Define the seed
torch.manual_seed(2025)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Whitened
whitened = True

# Initial path name
path_name = r"UCI_Datasets"
#all_datasets = ["bostonHousing","carbon", "concrete", "energy", "naval", "power-plant", "protein-tertiary-structure", "superconductivity", "wine-quality-red", "wine-white", "yacht"]
all_datasets = ["bostonHousing", "energy", "yacht","carbon", "concrete", "kin8nm"]

experiments = [
    ("var_rw_eps_1e1", LossConfig(type="rw", eps=1e-1, clamp=True)),
    ("var_rw_eps_1e2", LossConfig(type="rw", eps=1e-2, clamp=True)),
    ("var_rw_eps_1e3", LossConfig(type="rw", eps=1e-3, clamp=True)),
    ("var_rw_eps_1e4", LossConfig(type="rw", eps=1e-4, clamp=True)),
    # ("baseline", LossConfig(type="baseline", tau=0.0, tau_adjustment=False, clamp=True)),
    # ("beta_025", LossConfig(type="beta", tau=0.25, tau_adjustment=True, clamp=True)),
    # ("beta_05", LossConfig(type="beta", tau=0.5, tau_adjustment=True, clamp=True)),
    # ("beta_075", LossConfig(type="beta", tau=0.75, tau_adjustment=True, clamp=True)),
    # ("beta_1", LossConfig(type="beta", tau=1.0, tau_adjustment=True, clamp=True)),
]

# Constants - adjust if your CSV structure differs
METRICS_ROW = 1        # 0-indexed row containing the values (row 2 = index 1)
RMSE_COL = 2           # 0-indexed column for RMSE (col 2 = index 1)
NLL_COL = 1           # 0-indexed column for NLL  (col 3 = index 2)

results = []

for i, dataset in enumerate(all_datasets):
    print(f"\n===== DATASET: {dataset} =====")
    dataset_results = {exp_name: {"rmse": [], "nll": []} for exp_name, _ in experiments}

    split_dir = os.path.join(path_name, dataset, "splits_10fold")
    kf = KFold(n_splits=10, shuffle=True, random_state=2025)
    N = torch.from_numpy(np.loadtxt(os.path.join(path_name, dataset, "data", "data.txt"))).shape[0]

    for fold, _ in enumerate(kf.split(range(N))):
        for exp_name, _ in experiments:
            csv_path = f"results/{dataset}_fold{fold}_{exp_name}_best.csv"

            try:
                with open(csv_path, newline="") as f:
                    rows = list(csv.reader(f))
                rmse = float(rows[METRICS_ROW][RMSE_COL])
                nll  = float(rows[METRICS_ROW][NLL_COL])
                dataset_results[exp_name]["rmse"].append(rmse)
                dataset_results[exp_name]["nll"].append(nll)
                print(f"  {exp_name} fold {fold} | RMSE: {rmse:.4f}, NLL: {nll:.4f}")
            except FileNotFoundError:
                print(f"  WARNING: missing {csv_path}", file=sys.stderr)
            except (IndexError, ValueError) as e:
                print(f"  WARNING: could not parse {csv_path} – {e}", file=sys.stderr)

    for exp_name in dataset_results:
        rmse_vals = np.array(dataset_results[exp_name]["rmse"])
        nll_vals  = np.array(dataset_results[exp_name]["nll"])

        if len(rmse_vals) == 0:
            print(f"  WARNING: no data collected for {dataset}/{exp_name}, skipping.")
            continue

        results.append({
            "dataset":    dataset,
            "experiment": exp_name,
            "rmse_mean":  rmse_vals.mean(),
            "rmse_std":   rmse_vals.std(),
            "nll_mean":   nll_vals.mean(),
            "nll_std":    nll_vals.std(),
        })

csv_file = ("uci_results_summary_best.csv")
with open(csv_file, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "dataset", "experiment", "rmse_mean", "rmse_std", "nll_mean", "nll_std"
    ])
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to {csv_file}")