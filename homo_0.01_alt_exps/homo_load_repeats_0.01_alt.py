import torch
import numpy as np
import pandas as pd
from loss_functions import loss_rmse, loss_nll
import matplotlib.pyplot as plt

device = torch.device("cpu")

# ===== CONFIG =====
seed_list  = [2025, 2026, 2027, 2028, 2029]
noise_list = [0.01]
whitened   = True

hid = 64
layer = 3
act = "relu"
init = "kaiming"
folder = "results"

#exp_names = ["var_rw_eps_1e1", "var_rw_eps_1e2", "var_rw_eps_1e3", "var_rw_eps_1e4", "baseline", "beta_0.25", "beta_0.5", "beta_0.75", "beta_1"]
exp_names = ["baseline", "beta_0.25", "beta_0.5", "beta_0.75", "beta_1", "var_rw_eps_1e3"]
#exp_names = ["var_rw_eps_1e1", "var_rw_eps_1e2", "var_rw_eps_1e3", "var_rw_eps_1e4"]

titles = {
    "var_rw_eps_1e1":              r"Our method ($\epsilon = 1\times10^{-1}$)",
    "baseline":                   r"Baseline",
    "beta_0.25":                   r"$\beta = 0.25$",
    "beta_0.5":                    r"$\beta = 0.5$",
    "beta_0.75":                   r"$\beta = 0.75$",
    "beta_1":                     r"$\beta = 1.0$",
    "var_rw_eps_1e2":             r"Our method ($\epsilon = 1\times10^{-2}$)",
    "var_rw_eps_1e3":             r"Our method ($\epsilon = 1\times10^{-3}$)",
    "var_rw_eps_1e4":             r"Our method ($\epsilon = 1\times10^{-4}$)"
}

colors = plt.cm.tab10(np.linspace(0, 1.0, len(exp_names)))
axis_fontsize  = 14
legend_fontsize = 10
linewidth      = 1.25

# ===== DATA =====
def sine_function(x, noise_std=0.01):
    return 1* torch.sin(3 * torch.pi * x) + torch.randn_like(x) * noise_std

x_test = torch.linspace(0, 12, 1000).unsqueeze(1)

# ===== COLLECTION =====
# results[noise][exp_name] = {"rmse": [], "nll": []}
# curves[noise][exp_name]  = {"nll": [], "mse": []}
results = {noise: {exp: {"rmse": [], "nll": []} for exp in exp_names} for noise in noise_list}
curves  = {noise: {exp: {"nll": [], "rmse": []} for exp in exp_names} for noise in noise_list}
best = {noise: {exp: {"rmse": [], "nll": []} for exp in exp_names} for noise in noise_list}

for i, noise in enumerate(noise_list):
    for s in seed_list:
        torch.manual_seed(s)
        y_test = sine_function(x_test.squeeze(), noise_std=noise).unsqueeze(1)

        # Regenerate training data to recover whitening statistics
        x_train = torch.linspace(0, 12, 2000)
        y_train = sine_function(x_train, noise_std=noise)

        x_mean = x_train.mean(dim=0, keepdim=True)
        x_std  = x_train.std(dim=0, keepdim=True) + 1e-6
        y_mean = y_train.mean(dim=0, keepdim=True)
        y_std  = y_train.std(dim=0, keepdim=True) + 1e-6

        x_test_std = (x_test - x_mean) / x_std

        for exp_name in exp_names:
            model_path = f"{folder}/homo_{exp_name}_hid_{hid}_layer_{layer}_act_{act}_noise_{noise}_init_{init}_{s}"
            best_path = f"{folder}/homo_{exp_name}_hid_{hid}_layer_{layer}_act_{act}_noise_{noise}_init_{init}_{s}_best.csv"
            metrics_path = f"{folder}/homo_{exp_name}_hid_{hid}_layer_{layer}_act_{act}_noise_{noise}_init_{init}_{s}_metrics.csv"
            # ----- model eval -----
            try:
                model = torch.load(model_path, map_location=device)
                model.eval()
                with torch.no_grad():
                    if whitened:
                        mu, var = model(x_test_std.unsqueeze(1))
                        mu  = mu  * y_std + y_mean
                        var = var * (y_std ** 2)
                    else:
                        mu, var = model(x_test)

                rmse = loss_rmse(mu, y_test)[1].item()
                nll  = loss_nll(mu, var, y_test).item()
                results[noise][exp_name]["rmse"].append(rmse)
                results[noise][exp_name]["nll"].append(nll)
                print(f"  noise={noise} seed={s} {exp_name} | RMSE={rmse:.4f} NLL={nll:.4f}")
            except FileNotFoundError:
                print(f"  MISSING model: {model_path}")

            # ----- metrics curves -----
            try:
                df_m    = pd.read_csv(metrics_path)
                nll_col = df_m["valid_nll"].values.astype(float)
                rmse_col = df_m["valid_rmse"].values.astype(float)
                curves[noise][exp_name]["nll"].append(nll_col)
                curves[noise][exp_name]["rmse"].append(rmse_col)
                print(f"  Loaded curves: noise={noise} seed={s} {exp_name} ({len(nll_col)} epochs)")
            except FileNotFoundError:
                print(f"  MISSING metrics: {metrics_path}")

            # ----- best metrics -----
            try:
                df_b = pd.read_csv(best_path)
                best_nll = df_b["best_valid_nll"].values[0]
                best_rmse = df_b["best_valid_rmse"].values[0]
                best[noise][exp_name]["nll"].append(best_nll)
                best[noise][exp_name]["rmse"].append(best_rmse)
                print(f"  Loaded best metrics: noise={noise} seed={s} {exp_name} | Best RMSE={best_rmse:.4f} Best NLL={best_nll:.4f}")
            except FileNotFoundError:
                print(f"  MISSING best metrics: {best_path}")

# ===== CSV =====
rows = []
for noise in noise_list:
    for exp_name in exp_names:
        rmse_vals = np.array(results[noise][exp_name]["rmse"])
        nll_vals  = np.array(results[noise][exp_name]["nll"])
        if len(rmse_vals) == 0:
            continue
        rows.append({
            "experiment": exp_name,
            "noise":      noise,
            "n_seeds":    len(rmse_vals),
            "rmse_mean":  rmse_vals.mean(),
            "rmse_std":   rmse_vals.std(),
            "nll_mean":   nll_vals.mean(),
            "nll_std":    nll_vals.std(),
        })

df = pd.DataFrame(rows)
df.to_csv(fr"homo_{exp_name}_hid_{hid}_layer_{layer}_act_{act}_noise_{noise}_alt_sweep_summary.csv", index=False)
print(f"\nSaved: homo_{exp_name}_hid_{hid}_layer_{layer}_act_{act}_noise_{noise}_alt_sweep_summary.csv")
print(df.to_string(index=False))

# ===== CSV best =====
rows = []#for noise in noise_list:
for exp_name in exp_names:
    rmse_vals = np.array(best[noise][exp_name]["rmse"])
    nll_vals  = np.array(best[noise][exp_name]["nll"])
    if len(rmse_vals) == 0:
        continue
    rows.append({
        "experiment": exp_name,
        "noise":      noise,
        "n_seeds":    len(rmse_vals),
        "rmse_mean":  rmse_vals.mean(),
        "rmse_std":   rmse_vals.std(),
        "nll_mean":   nll_vals.mean(),
        "nll_std":    nll_vals.std(),
    })

df = pd.DataFrame(rows)
df.to_csv(fr"homo_{exp_name}_hid_{hid}_layer_{layer}_act_{act}_noise_{noise}_alt_best_sweep_summary.csv", index=False)
print(f"\nSaved: homo_{exp_name}_hid_{hid}_layer_{layer}_act_{act}_noise_{noise}_alt_best_sweep_summary.csv")
print(df.to_string(index=False))

for exp_name in exp_names:
    n = sum(len(results[noise][exp_name]["rmse"]) for noise in noise_list)
    expected = len(seed_list) * len(noise_list)
    if n < expected:
        print(f"WARNING: {exp_name} only has {n}/{expected} total entries loaded")

def plot_curve(noise, metric, ylabel, filename, log_scale=False, start=0, end=None, plot_every=10, ylim = None):
    fig, ax = plt.subplots()
    for exp_name, color in zip(exp_names, colors):
        arrays = curves[noise][exp_name][metric]
        if not arrays:
            continue
        min_len = min(len(a) for a in arrays)
        mat     = np.stack([a[:min_len] for a in arrays])

        # Slice epoch range
        _end   = end if end is not None else mat.shape[1]
        mat    = mat[:, start:_end]

        mean   = mat.mean(axis=0)
        std    = mat.std(axis=0)
        epochs = np.arange(start, start + mat.shape[1])

        mean   = mean[::plot_every]
        std    = std[::plot_every]
        epochs = epochs[::plot_every]

        ax.plot(epochs, mean, label=titles[exp_name], linewidth=linewidth, color=color)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Epoch", fontsize=axis_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_fontsize)
    if log_scale:
        ax.set_yscale("log")
        ax.grid(which="both")
    else:
        ax.grid()
    ax.legend(fontsize=legend_fontsize)
    if ylim != None:
        ax.set_ylim(ylim)
    fig.tight_layout()
    fig.savefig(filename, dpi=200, format="pdf")
    print(f"Saved: {filename}")
    plt.close(fig)

for noise in [0.01]:
    tag = str(noise).replace(".", "_")
    plot_curve(noise, "nll", "Valid NLL",       f"homo_hid_{hid}_layer_{layer}_act_{act}_noise_{tag}_alt_valid_nll_sens.pdf",       plot_every=50, start=500, end = 10000, ylim = [-2, 2])
    plot_curve(noise, "nll", "Valid NLL (log)", f"homo_hid_{hid}_layer_{layer}_act_{act}_noise_{tag}_alt_valid_nll_log.pdf",    log_scale=True, plot_every=50, start = 500, end = 10000)
    plot_curve(noise, "nll", "Valid NLL",       f"homo_hid_{hid}_layer_{layer}_act_{act}_noise_{tag}_alt_valid_nll_trunc.pdf",  start=500, end=10000, plot_every=50)
    plot_curve(noise, "rmse", "Valid RMSE",       f"homo_hid_{hid}_layer_{layer}_act_{act}_noise_{tag}_alt_valid_rmse.pdf",        plot_every=50, start = 500, end = 10000)