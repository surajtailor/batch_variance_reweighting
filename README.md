# Batch wise variance informed Gaussian negative log-likelihood re-weighting for effective training of probabilistic neural networks

Code for paper: "Batch wise variance informed Gaussian negative log-likelihood re-weighting for effective training
of probabilistic neural networks". The repository also includes the baseline NLL and β-NLL losses used for comparison, plus a training loop with MLflow tracking, early stopping and hyperparameter scheduling.

## Contents

| File | Purpose |
|---|---|
| `models.py` | `MVENetwork`: separate (or shared-trunk) mean and variance networks; `ArchitectureConfig`; custom `HatActivation` |
| `loss_functions.py` | Baseline Gaussian NLL, β-NLL (`tau`), Var-RW losses, and MSE / RMSE / NLL evaluation helpers |
| `training_functions.py` | Main `trainer` loop, config dataclasses, `EarlyStopping`, `HypScheduler`, LR schedulers |
| `training_functions_uci.py` | Earlier trainer variant used for the UCI experiments (no `z_adjustment`, no `exp_decay`) |
| `utility.py` | `IndexedDataset` (returns the sample index with each item) and `tensor_to_str` |
| `hetero_exps`, `homo_0.01_exps`, `homo_0.01_alt_exps`, `homo_0.3_exps`, `UCI_exps` | Experiment files for the heteroscedastic, homoscedastic (noise 0.01 / 0.3) and UCI benchmark settings |

## Method overview

An MVE network outputs a predictive mean `mu(x)` and variance `var(x)`, trained by minimising the Gaussian NLL. Standard NLL training can under-fit the mean in high-variance regions. This repo implements three families of loss:

1. **Baseline NLL** (`type="baseline"`): the standard Gaussian NLL.
2. **β-NLL** (`type="beta"`): each sample's log-likelihood is scaled by `var^tau` (detached), as in β-NLL. `tau=0` recovers the baseline.
3. **Var-RW** (`type="rw"`, `"rw_linear"`, `"rw_tanh"`, `"rw_alpha_decay"`): per-sample log-likelihoods are reweighted by weights computed from the *detached* predicted variances:
   - Weights come from min-max normalisation of the variance, or, with `z_adjustment=True`, of the log standard deviation, giving weights in [0, 1].
   - Separate weights are used for the mean-oriented term (linear in variance) and the variance-oriented term (quadratic in variance). The training loss is the sum of the two terms.
   - Optional gating blends the weights with their complement (`1 - w`):
     - `rw_linear`: a linear gate (currently fixed to the pure-weight case).
     - `rw_tanh`: a tanh gate driven by the spread of the variances, scaled by `alpha`.
     - `rw_alpha_decay`: a fixed gate `alpha`, which can be scheduled during training.
   - `eps` is a small offset keeping the lowest weight above zero, and can also be scheduled.

Because the weights are computed from detached variances, they act as per-sample loss weights and do not add gradient paths.

### Loss configuration (`LossConfig`)

| Field | Meaning |
|---|---|
| `type` | `baseline`, `beta`, `rw`, `rw_linear`, `rw_tanh`, `rw_alpha_decay` |
| `tau`, `tau_adjustment` | β-NLL exponent and on/off switch (also usable inside the Var-RW losses) |
| `eps` | Offset on the min-max normalised weights |
| `alpha` | Gate scale (`rw_tanh`) or fixed gate value (`rw_alpha_decay`) |
| `clamp` | Clamp variances to avoid `log(0)` / division by zero |
| `z_adjustment` | Normalise in log-std space instead of variance space |

### Hyperparameter scheduling (`HyperConfig`)

When `enable=True`, `eps` (for `rw`, `rw_linear`, `rw_tanh`) or `alpha` (for `rw_alpha_decay`) is updated by `HypScheduler` each epoch. Modes (`type`):

- `product` / `sum`: multiply or add `amount` after `patience` epochs without validation improvement, until `final` is reached
- `schedule`: step through a log or linear grid of `schedule_steps` values
- `exp_decay`: smooth exponential interpolation from `initial` to `final` over `total_steps` (set this to `n_iter`)

### Other training options

- **LR schedulers** (`TrainingConfig.scheduler_type`): `ExponentialLR`, `StepLR`, decaying to a final LR of 1e-5
- **Weight init** (`TrainingConfig.initialise`): `random`, `kaiming`, `xavier`, `orthogonal`
- **Early stopping**: monitors validation NLL and restores the best weights at the end
- **Checkpointing**: `EarlyStoppingConfig.save_period` and `save_freq` save periodic model snapshots

## Outputs

Each run logs to MLflow (validation NLL / RMSE / MSE, training loss, learning rate) and writes to disk:

- `<model_save_name>`: the full saved model (`torch.save(model, ...)`)
- `<model_save_name>_metrics.csv`: per-epoch curves, including `alpha`, `epsilon`, and the variance statistics and gate values for Var-RW runs
- `<model_save_name>_best.csv`: best epoch and its validation NLL / RMSE / MSE

## Citation

If you use this code, please cite the associated publication.
