from torch.optim.lr_scheduler import ExponentialLR, LinearLR, StepLR
from loss_functions import loss_fn, loss_fn_var_weighting
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from utility import IndexedDataset, tensor_to_str
import torch.nn as nn
import pandas as pd
from dataclasses import dataclass
import torch
import numpy as np

@dataclass
class TrainingConfig:
    n_iter: int = 10000
    batch_size: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 0
    scheduler_type: str or None = None
    valid_freq: int = 1
    device: torch.device = torch.device("cpu")
    num_workers: int = 0
    initialise: bool = False

@dataclass
class HyperConfig:
    enable: bool = False
    initial: float = 1e-6
    final: float = 1e-4
    patience: int = 200
    type: str = "product"
    amount: float = 2.0
    min_delta: float = 1e-4# Improvement threshold for epsilon increment
    schedule_style: str = "log"
    schedule_steps: int = 6

@dataclass
class LossConfig:
    type: str = "baseline"
    eps: float = 0
    alpha: float = 0
    tau: float = 0
    tau_adjustment: bool = False
    clamp: bool = False

@dataclass
class EarlyStoppingConfig:
    enabled: bool = False
    patience: int = 100
    min_delta: float = 1e-4
    warmup: int = 0
    save_period: bool = False
    save_freq: int = 10

@dataclass
class ExperimentConfig:
    experiment_name: str = "default"
    run_name: str or None = None
    model_save_name: str = "model"

class EarlyStopping:
    def __init__(self, patience=100, min_delta=1e-4, warmup=0):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup = warmup
        self.best = float("inf")
        self.counter = 0
        self.best_state = None
        self.stop = False
        self.best_iter = 0

    def step(self, metric, model, iter):
        if iter < self.warmup:
            return False

        if metric < self.best - self.min_delta:
            self.best = metric
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.best_iter = iter
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stop = True

        return self.stop

class HypScheduler:
    # Increases or reduces a hyperparameter after patience without improvement
    def __init__(
        self,
        initial,
        final,
        patience=200,
        min_delta=1e-4,
        type="product",
        amount=2.0,
        schedule_steps=6,      # NEW: number of steps in schedule mode
        schedule_style="log"   # "log" or "linear"
    ):
        self.initial = initial
        self.final = final
        self.patience = patience
        self.min_delta = min_delta
        self.type = type
        self.inc_or_dec = "increase" if final > initial else "decrease"
        self.amount = amount

        self.schedule_steps = schedule_steps
        self.schedule_style = schedule_style

        self.current = initial
        self.no_improve_counter = 0
        self.best_metric = None

        # schedule state
        self.schedule = None
        self.step_idx = 0

        if self.type == "schedule":
            self._build_schedule()

    # ----------------------------
    # Build schedule
    # ----------------------------
    def _build_schedule(self):
        if self.schedule_style == "log":
            self.schedule = np.logspace(
                np.log10(self.initial),
                np.log10(self.final),
                self.schedule_steps
            ).tolist()

        elif self.schedule_style == "linear":
            self.schedule = np.linspace(
                self.initial,
                self.final,
                self.schedule_steps
            ).tolist()

        else:
            raise ValueError("schedule_style must be 'log' or 'linear'")

    # ----------------------------
    # Step function
    # ----------------------------
    def step(self, metric):

        # first call
        if self.best_metric is None:
            self.best_metric = metric
            return self.current

        # improvement check
        if metric < self.best_metric - self.min_delta:
            self.best_metric = metric
            self.no_improve_counter = 0
        else:
            self.no_improve_counter += 1

        # update hyperparameter
        if self.no_improve_counter >= self.patience:

            if self.type == "product":
                if self.inc_or_dec == "increase":
                    self.current *= self.amount
                else:
                    self.current /= self.amount

            elif self.type == "sum":
                if self.inc_or_dec == "increase":
                    self.current += self.amount
                else:
                    self.current -= self.amount

            elif self.type == "schedule":
                if self.schedule is None:
                    self._build_schedule()

                self.step_idx = min(self.step_idx + 1, len(self.schedule) - 1)
                self.current = self.schedule[self.step_idx]

            self.no_improve_counter = 0

            # clamp to final
            if self.inc_or_dec == "decrease" and self.current < self.final:
                self.current = self.final
            elif self.inc_or_dec == "increase" and self.current > self.final:
                self.current = self.final

        return self.current

def scheduler_case(scheduler_type, optimizer, initial_lr, num_epochs, final_lr = 1e-4, steps = 3):
    if scheduler_type == "ExponentialLR":
        gamma_value = (final_lr/initial_lr)**(1/num_epochs)
        return ExponentialLR(optimizer, gamma=gamma_value)

    if scheduler_type == "StepLR":
        step_size = num_epochs/steps
        gamma_value = (final_lr/initial_lr)**(1/steps)
        return StepLR(optimizer, step_size, gamma=gamma_value)
    return None

def trainer(
    model,
    train_data,
    valid_data,
    exp_cfg: ExperimentConfig,
    tr_cfg: TrainingConfig,
    l_cfg: LossConfig,
    es_cfg: EarlyStoppingConfig,
    hyp_cfg: HyperConfig
):

    if tr_cfg.initialise:
        model.init_weights("kaiming")

    stopper = EarlyStopping(patience = es_cfg.patience, min_delta=es_cfg.min_delta, warmup=es_cfg.warmup) if es_cfg.enabled else None
    hyp_scheduler = HypScheduler(initial=hyp_cfg.initial, final=hyp_cfg.final, patience=hyp_cfg.patience, min_delta=hyp_cfg.min_delta, type=hyp_cfg.type, amount=hyp_cfg.amount) if hyp_cfg.enable else None
    mlflow.set_experiment(exp_cfg.experiment_name)

    optimizer = optim.Adam(model.parameters(), lr=tr_cfg.learning_rate, weight_decay=tr_cfg.weight_decay)
    scheduler = scheduler_case(tr_cfg.scheduler_type, optimizer, tr_cfg.learning_rate, tr_cfg.n_iter, final_lr=1e-5)

    with mlflow.start_run(run_name=exp_cfg.run_name) as run:
        run_id = run.info.run_id

        # ===== metric storage =====
        metric_rows = []

        best_valid_nll = float("inf")
        best_valid_rmse = None
        best_valid_mse = None
        best_epoch = None

        # ===== params =====
        mlflow.log_params({
            "num_iter": tr_cfg.n_iter,
            "initial_lr": tr_cfg.learning_rate,
            "scheduler": tr_cfg.scheduler_type,
            "type": l_cfg.type,
            "tau": l_cfg.tau,
            "tau_adjustment": l_cfg.tau_adjustment,
            "eps": l_cfg.eps,
            "alpha": l_cfg.alpha
        })

        # ===== dataloaders =====
        train_inputs, train_targets = train_data
        valid_inputs, valid_targets = valid_data

        train_loader = DataLoader(
            IndexedDataset(train_inputs, train_targets),
            shuffle=True,
            batch_size=tr_cfg.batch_size,
            num_workers=tr_cfg.num_workers
        )

        valid_loader = DataLoader(
            IndexedDataset(valid_inputs, valid_targets),
            shuffle=False,
            batch_size=2048,
            num_workers=tr_cfg.num_workers
        )

        # ===== training loop =====
        for epoch in range(tr_cfg.n_iter):
            model.train()
            train_loss_sum = 0.0
            total_batches = 0

            for xb, yb, _ in train_loader:
                xb, yb = xb.to(tr_cfg.device), yb.to(tr_cfg.device)
                mu, var = model(xb)
                if l_cfg.type in ["rw_alpha_decay"]:
                    loss_mu, loss_var,  min_var, max_var, mean_var, extra = loss_fn_var_weighting(mu, var, yb, clamp=l_cfg.clamp, eps=l_cfg.eps, alpha=l_cfg.alpha, type=l_cfg.type)
                    loss = loss_mu + loss_var
                elif l_cfg.type in ["rw", "rw_linear", "rw_tanh"]:
                    loss_mu, loss_var,  min_var, max_var, mean_var, extra = loss_fn_var_weighting(mu, var, yb, clamp=l_cfg.clamp, eps=l_cfg.eps, alpha=l_cfg.alpha, type=l_cfg.type)
                    loss = loss_mu + loss_var
                elif l_cfg.type == "baseline":
                    loss = loss_fn(mu, var, yb, tau=0.0, tau_adjustment=False, clamp=l_cfg.clamp)
                elif l_cfg.type == "beta":
                    loss = loss_fn(mu, var, yb, tau=l_cfg.tau, tau_adjustment=l_cfg.tau_adjustment, clamp=l_cfg.clamp)
                else:
                    raise ValueError(f"Unknown training type: {l_cfg.type}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()
                total_batches += 1

            train_loss_epoch = train_loss_sum / total_batches

            # ===== validation =====
            valid_nll = None
            valid_rmse = None
            valid_mse = None

            if epoch % tr_cfg.valid_freq == 0:
                model.eval()

                n_sum = 0
                nll_sum = 0.0
                mse_sum = 0.0

                with torch.no_grad():
                    for xb, yb, _ in valid_loader:
                        xb, yb = xb.to(tr_cfg.device), yb.to(tr_cfg.device)
                        mu, var = model(xb)

                        se = (yb.squeeze() - mu.squeeze()) ** 2
                        batch_n = yb.numel()

                        nll = 0.5 * torch.log(2 * torch.pi * var.squeeze()) + 0.5 * se / var.squeeze()
                        nll_sum += torch.sum(nll).item()
                        mse_sum += torch.sum(se).item()
                        n_sum += batch_n

                valid_nll = nll_sum / n_sum
                valid_mse = mse_sum / n_sum
                valid_rmse = valid_mse ** 0.5

                mlflow.log_metrics({
                    "valid_nll": valid_nll,
                    "valid_rmse": valid_rmse,
                    "valid_mse": valid_mse
                }, step=epoch)

                # ===== best metric tracking =====
                if valid_nll < best_valid_nll:
                    best_valid_nll = valid_nll
                    best_valid_rmse = valid_rmse
                    best_valid_mse = valid_mse
                    best_epoch = epoch

                if stopper:
                    if stopper.step(valid_nll, model, epoch):
                        print(f"EARLY STOP at epoch {epoch}")
                        break

            # ===== scheduler =====
            if scheduler:
                scheduler.step()

            if hyp_scheduler:
                if l_cfg.type == "rw_alpha_decay":
                    l_cfg.alpha = hyp_scheduler.step(valid_nll if valid_nll is not None else float("inf"))
                elif l_cfg.type in ["rw", "rw_linear", "rw_tanh"]:
                    l_cfg.eps  = hyp_scheduler.step(valid_nll if valid_nll is not None else float("inf"))

            if es_cfg.save_period == True and epoch % es_cfg.save_freq == 0:
                torch.save(model, f"{exp_cfg.model_save_name}_epoch{epoch}")

            lr = optimizer.param_groups[0]["lr"]

            mlflow.log_metric("train_loss", train_loss_epoch, step=epoch)
            mlflow.log_metric("lr", lr, step=epoch)

            metric_rows.append({
                "epoch": epoch,
                "train_loss": train_loss_epoch,
                "valid_nll": valid_nll,
                "valid_rmse": valid_rmse,
                "valid_mse": valid_mse,
                "lr": lr,
                "alpha": l_cfg.alpha if l_cfg.type in ["rw_linear", "rw_tanh", "rw_alpha_decay"] else None,
                "epsilon": l_cfg.eps if l_cfg.type in ["rw", "rw_linear", "rw_tanh", "rw_alpha_decay"] else None,
                #"mean_var": mean_var.item() if l_cfg.type in ["rw", "rw_linear", "rw_tanh", "rw_alpha_decay"] else None,
                #"min_var": min_var.item() if l_cfg.type in ["rw", "rw_linear", "rw_tanh", "rw_alpha_decay"] else None,
                #"max_var": max_var.item() if l_cfg.type in ["rw", "rw_linear", "rw_tanh", "rw_alpha_decay"] else None,
                #"eps_adj": extra[0].item() if l_cfg.type in ["rw", "rw_linear", "rw_tanh", "rw_alpha_decay"] else None,
                #"gate": extra[1].item() if l_cfg.type in ["rw_linear", "rw_tanh", "rw_alpha_decay"] else None
            })

            print(f"Epoch {epoch:5d} | Train {train_loss_epoch:.4f} | Valid {valid_nll}")

            if l_cfg.type in ["rw_linear", "rw_tanh", "rw_alpha_decay"]:
                print(
                    #f"Max Var: {max_var.item():8.4f} | "
                    #f"Min Var: {min_var.item():8.4f} | "
                    #f"Mean Var: {mean_var.item():8.4f}, "
                    f"Epsilon: {l_cfg.eps:.6f}, "
                    f"Alpha: {l_cfg.alpha:.4f}, "
                    #f"Epsilon Adj: {extra[0].item():.6f}, "
                    #f"Gate: {extra[1].item():.4f}"
                )

        # ===== restore best =====
        if stopper and stopper.best_state is not None:
            model.load_state_dict(stopper.best_state)

        # ===== save model =====
        #mlflow.pytorch.log_model(model, exp_cfg.model_save_name)
        torch.save(model, exp_cfg.model_save_name)

        # ===== save curves =====
        metrics_df = pd.DataFrame(metric_rows)
        curve_csv = exp_cfg.model_save_name + "_metrics.csv"
        metrics_df.to_csv(curve_csv, index=False)
        #mlflow.log_artifact(curve_csv)

        # ===== save best =====
        best_df = pd.DataFrame([{
            "best_epoch": best_epoch,
            "best_valid_nll": best_valid_nll,
            "best_valid_rmse": best_valid_rmse,
            "best_valid_mse": best_valid_mse}])

        best_csv = exp_cfg.model_save_name + "_best.csv"
        best_df.to_csv(best_csv, index=False)
        #mlflow.log_artifact(best_csv)

        mlflow.log_metrics({
            "best_valid_nll": best_valid_nll,
            "best_valid_rmse": best_valid_rmse,
            "best_valid_mse": best_valid_mse,
            "best_epoch": best_epoch
        })

    return run_id

def init_glorot(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)