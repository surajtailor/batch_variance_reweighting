import torch
from training_functions import trainer, TrainingConfig,EarlyStoppingConfig, ExperimentConfig, LossConfig, HyperConfig
from models import MVENetwork, ArchitectureConfig
import matplotlib.pyplot as plt

# GPU SETUP (IMPORTANT)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Using device:", device)
torch.backends.cudnn.benchmark = False
whitened = True

# Epochs
common_train_cfg = TrainingConfig(
    n_iter = 10000,
    learning_rate=1e-3,
    batch_size=64,
    valid_freq=1,
    scheduler_type=None,
    weight_decay= 1e-5,
    num_workers=0,
    device=device,
    initialise = 'kaiming')

common_es_cfg = EarlyStoppingConfig(
    enabled=False,
    patience=100,
    min_delta=1e-4,
    warmup=0,
    save_period = True,
    save_freq = 500
)

common_arch_cfg = ArchitectureConfig(
    input_size=1,
    output_size=1,
    hidden_size=64,
    num_layers=3,
    activation_func='relu',
    shared = False,
)

hyp_cfg = HyperConfig(
    enable=False,
    initial=1e-6,
    final=1e-1,
    patience=100,
    type="exp_decay",
    total_steps = common_train_cfg.n_iter
)

# Base signal
def sine_function(x, noise_std=0.01):
    return  1* torch.sin(3 * torch.pi * x) + torch.randn_like(x) * noise_std

# Repeat for different initilisations do 10
seed = [2025, 2026, 2027, 2028, 2029]
noise_list = [0.01]
for noise in noise_list:
            for s in seed:
                torch.manual_seed(s)

                # Generate synthetic data
                x_train = torch.linspace(0, 12, 2000)
                x_valid = torch.linspace(0, 12, 1000)

                # Final signal
                y_train = sine_function(x_train, noise_std=noise)
                y_valid = sine_function(x_valid, noise_std=noise)

                train_data = (x_train.unsqueeze(1), y_train.unsqueeze(1))
                valid_data = (x_valid.unsqueeze(1), y_valid.unsqueeze(1))

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
                    train_data = (x_train_std.unsqueeze(1), y_train_std.unsqueeze(1))
                    valid_data = (x_valid_std.unsqueeze(1), y_valid_std.unsqueeze(1))
                else:
                    # Store train/valid/test datasets
                    train_data = (x_train.unsqueeze(1), y_train.unsqueeze(1))
                    valid_data = (x_valid.unsqueeze(1), y_valid.unsqueeze(1))

                #plot train data
                plt.figure(figsize=(8, 6))
                plt.scatter(x_train, y_train, s=10, alpha=0.5, color='tab:blue', label='Train Data')
                plt.title(f"Training Data with Noise Std: {noise}")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.legend()
                plt.show()

                experiments = [
                    ("var_rw_eps_1e2", LossConfig(type="rw", eps=1e-2, clamp=True)),
                    ("var_rw_eps_1e1", LossConfig(type="rw", eps=1e-1, clamp=True)),
                    ("var_rw_eps_1e3", LossConfig(type="rw", eps=1e-3, clamp=True)),
                    ("var_rw_eps_1e4", LossConfig(type="rw", eps=1e-4, clamp=True)),
                ]

                for exp_name, loss_cfg in experiments:
                    model_name = f"homo_{exp_name}_hid_{common_arch_cfg.hidden_size}_layer_{common_arch_cfg.num_layers}_act_{common_arch_cfg.activation_func}_noise_{noise}_init_{common_train_cfg.initialise}_{s}"
                    exp_cfg = ExperimentConfig(
                        experiment_name=model_name,
                        run_name=model_name,
                        model_save_name=model_name
                    )

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