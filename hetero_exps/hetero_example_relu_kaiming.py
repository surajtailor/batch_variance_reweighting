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
    n_iter = 30000,
    learning_rate=1e-3,
    batch_size=256,
    valid_freq=1,
    scheduler_type=None,
    weight_decay= 1e-5,
    num_workers=0,
    device=device,
    initialise = "kaiming",
)

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
    num_layers=5,
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
def base_function(x):
    return 10 * torch.sin(15 * x) + 3 * torch.sin(10 * x) + 4*torch.cos(5 * x)

# Aleatoric noise - multiple bells with different widths and positions
def aleatoric_noise_function(x, noise_std, ):
    bell1 = torch.exp(-0.5 * ((x - 3.0) / 0.6) ** 2)
    bell2 = torch.exp(-0.5 * ((x - 6.0) / 0.6) ** 2)
    bell3 = torch.exp(-0.5 * ((x - 10.0) / 0.8) ** 2)
    # Combine bells with different amplitudes
    combined = 0.7*bell1 + bell2  + 1.5*bell3
    return torch.randn_like(x) * noise_std * combined

# Repeat for different initilisations do 10
seed = [2025, 2026, 2027, 2028, 2029]
for s in seed:
    torch.manual_seed(s)

    # Generate synthetic data
    x_train = torch.linspace(0, 15, 5000)
    x_valid = torch.linspace(0, 15, 1000)

    # Final signal
    # Define aleatoric noise
    noise_std = 5

    # Final signal
    y_train = base_function(x_train) + aleatoric_noise_function(x_train, noise_std)
    y_valid = base_function(x_valid) + aleatoric_noise_function(x_valid, noise_std)

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

    #plot training data
    # Plot training data
    plt.figure()
    plt.scatter(train_data[0].cpu(), train_data[1].cpu(), s=10, alpha=0.5, label="Train Data")
    plt.show()

    experiments = [
        ("var_rw_eps_1e2", LossConfig(type="rw", eps=1e-2, clamp=True)),
        ("var_rw_eps_1e3", LossConfig(type="rw", eps=1e-3, clamp=True)),
        ("var_rw_eps_1e4", LossConfig(type="rw", eps=1e-4, clamp=True)),
        ("var_rw_eps_1e1", LossConfig(type="rw", eps=1e-1, clamp=True)),
    ]

    for exp_name, loss_cfg in experiments:
        model_name = f"hetero_{exp_name}_hid_{common_arch_cfg.hidden_size}_layer_{common_arch_cfg.num_layers}_act_{common_arch_cfg.activation_func}_init_{common_train_cfg.initialise}_{s}"
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