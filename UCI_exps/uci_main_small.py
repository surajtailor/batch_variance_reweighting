import torch
import numpy as np
from training_functions_uci import trainer, TrainingConfig,EarlyStoppingConfig, ExperimentConfig, LossConfig, HyperConfig
from models import MVENetwork, ArchitectureConfig
from sklearn.model_selection import KFold
import os

# GPU SETUP (IMPORTANT)
torch.manual_seed(2025)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Using device:", device)
torch.backends.cudnn.benchmark = True
whitened = True

common_train_cfg = TrainingConfig(
    n_iter = 60000,
    learning_rate=1e-3,
    batch_size=32,
    valid_freq=1,
    scheduler_type=None,
    weight_decay=1e-3,
    num_workers=0,
    device=device,
    initialise = 'kaiming'
)

common_es_cfg = EarlyStoppingConfig(
    enabled=True,
    patience=200,
    min_delta=1e-4,
    warmup=0
)

common_arch_cfg = ArchitectureConfig(
    hidden_size=50,
    num_layers=1,
    activation_func='relu',
    shared = False,
)

hyp_cfg = HyperConfig(
    enable=False,
    initial=1e-6,
    final=1e-1,
    patience=100,
    type="schedule",
    amount=10.0,
    min_delta=1e-4,
    schedule_style="log",
    schedule_steps = 5
)

size_of_train_sets = [455, 9648, 927, 691, 7373, 10740, 8611, 41157, 19136, 1439, 4408, 277]

# Initial path name
path_name = r"UCI_Datasets"

all_datasets = ["bostonHousing", "energy", "yacht"]

experiments = [
    ("var_rw_eps_1e2", LossConfig(type="rw", eps=1e-2, clamp=True)),
    ("var_rw_eps_1e1", LossConfig(type="rw", eps=1e-1, clamp=True)),
    ("var_rw_eps_1e3", LossConfig(type="rw", eps=1e-3, clamp=True)),
    ("var_rw_eps_1e4", LossConfig(type="rw", eps=1e-4, clamp=True)),
]

for i, dataset in enumerate(all_datasets):

    print(f"\n===== DATASET: {dataset} =====")

    data = torch.from_numpy(np.loadtxt(os.path.join(path_name, dataset, "data", "data.txt")))
    feature_index = torch.from_numpy(np.loadtxt(os.path.join(path_name, dataset, "data", "index_features.txt")))
    feature_targets = torch.from_numpy(np.loadtxt(os.path.join(path_name, dataset, "data", "index_target.txt")))

    x_data = data[:, feature_index.long()].float()
    y_data = data[:, feature_targets.long()].float()

    # Create a 10 fold cross validation split using the predefined splits and train models on each split
    split_dir = os.path.join(path_name, dataset, "splits_10fold")
    os.makedirs(split_dir, exist_ok=True)

    N = x_data.shape[0]
    kf = KFold(n_splits=10, shuffle=True, random_state=2025)

    common_arch_cfg.input_size = feature_index.shape[0]
    if feature_targets.ndim == 0:
        common_arch_cfg.output_size = 1
    else:
        common_arch_cfg.output_size = feature_targets.shape[0]

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(N))):
        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)

        # Save to text files
        np.savetxt(os.path.join(split_dir, f"train_idx_fold_{fold}.txt"),train_idx,fmt="%d")
        np.savetxt(os.path.join(split_dir, f"val_idx_fold_{fold}.txt"),val_idx,fmt="%d")
        print(f"Saved 10-fold splits for {dataset}")

        x_train = x_data[train_idx]
        y_train = y_data[train_idx]
        x_valid = x_data[val_idx]
        y_valid = y_data[val_idx]

        # Compute training statistics
        x_mean = x_train.mean(dim=0, keepdim=True)
        x_std = x_train.std(dim=0, keepdim=True) + 1e-6

        y_mean = y_train.mean(dim=0, keepdim=True)
        y_std = y_train.std(dim=0, keepdim=True) + 1e-6

        #  Apply whitening
        x_train_std = (x_train - x_mean) / x_std
        x_valid_std = (x_valid - x_mean) / x_std

        y_train_std = (y_train - y_mean) / y_std
        y_valid_std = (y_valid - y_mean) / y_std

        if whitened:
            # Store train/valid/test datasets
            train_data = (x_train_std, y_train_std)
            valid_data = (x_valid_std, y_valid_std)
        else:
            # Store train/valid/test datasets
            train_data = (x_train, y_train)
            valid_data = (x_valid, y_valid)

        for exp_name, loss_cfg in experiments:
            model_name = f"{dataset}_fold{fold}_{exp_name}"
            exp_cfg = ExperimentConfig(
                experiment_name=model_name,
                run_name=model_name,
                model_save_name=model_name
            )

            if feature_targets.ndim == 0:
                common_arch_cfg.output_size = 1
            else:
                common_arch_cfg.output_size = feature_targets.shape[0]

            model = MVENetwork(
                input_size= common_arch_cfg.input_size,
                output_size= common_arch_cfg.output_size,
                hidden_size= common_arch_cfg.hidden_size,
                num_layers= common_arch_cfg.num_layers,
                activation_func= common_arch_cfg.activation_func,
                shared = common_arch_cfg.shared,
            ).to(device)

            print(f"Running {model_name}")
            print(f"Train size = {x_train.shape[0]} | Iter = {common_train_cfg.n_iter}")

            run_id = trainer(
                model,
                train_data,
                valid_data,
                exp_cfg,
                common_train_cfg,
                loss_cfg,
                common_es_cfg,
                hyp_cfg
            )