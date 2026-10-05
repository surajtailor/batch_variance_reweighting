import torch

def loss_fn(mu, var, targets, tau= 0.5, tau_adjustment = False, clamp = True):
    mu_detach = mu.detach()
    var_detach = var.detach()

    if clamp:
        var_clamped = torch.clamp(var, min=1e-8) # Clamp the variance to avoid log(0) or division by 0
        var_clamped_detached = torch.clamp(var_detach, min=1e-8, max = 1000)  # Clamp the detached variance to avoid log(0) or division by 0
    else:
        var_clamped = var
        var_clamped_detached = var_detach

    log_likelihood = -(1 / 2) * torch.log(var_clamped.squeeze()) - (1 / 2) * (mu.squeeze() - targets.squeeze()) ** 2 / (var_clamped.squeeze())

    # Adjust the log likelihood and prior by tau if specified
    if tau_adjustment == True:
        scale = (var_clamped_detached.squeeze() ** (tau))
        log_likelihood = scale * log_likelihood

    return -torch.mean(log_likelihood)

def loss_fn_var_weighting(mu, var, targets, clamp = True, cond = 1e-6, eps = 1e-8, alpha = None, type = None, tau_adjustment = False, tau = 0.5, z_adjustment = False):
    mu_detach = mu.detach()
    var_detach = var.detach()

    if clamp:
        var_clamped = torch.clamp(var, min=1e-8) # Clamp the variance to avoid log(0) or division by 0
        var_clamped_detached = torch.clamp(var_detach, min=1e-8, max = 1000)  # Clamp the detached variance to avoid log(0) or division by 0
    else:
        var_clamped = var
        var_clamped_detached = var_detach

    log_likelihood = -(1 / 2) * torch.log(var_clamped.squeeze()) - (1 / 2) * (mu.squeeze() - targets.squeeze()) ** 2 / (var_clamped.squeeze())

    # Calculate statistics of var_clamped_detached for reweighting
    min_var = torch.min(var_clamped_detached, dim=0).values
    max_var = torch.max(var_clamped_detached, dim=0).values
    mean_var = torch.mean(var_clamped_detached, dim=0)

    # Adjust the log likelihood and prior by tau if specified
    if tau_adjustment == True:
        scale = (var_clamped_detached.squeeze() ** (tau))
        log_likelihood = scale * log_likelihood

    if z_adjustment == True:
        # Map to log space
        log_std = torch.log(var_clamped_detached ** 0.5)
        min_log_std = torch.min(log_std)
        max_log_std = torch.max(log_std)

        eps_adj = eps * (max_log_std - min_log_std)

        # Normalise in log space → uniform [0,1] weights
        z_norm = (log_std - min_log_std + eps_adj) / (max_log_std - min_log_std + cond)
        rescale_mu = z_norm
        rescale_var = z_norm
        extra = [eps_adj]
    else:
        # Max min normalization of var_H
        eps_adj = eps * (max_var - min_var)
        eps_adj_var = eps * (2*max_var**2 -2* min_var**2)
        rescale_mu = (var_clamped_detached - min_var + eps_adj.squeeze(0)) / (max_var - min_var + cond)
        rescale_var = (2*(var_clamped_detached**2) - 2*(min_var**2) + eps_adj_var) / (2*(max_var**2) - 2*(min_var**2) + cond)
        extra = [eps_adj]

    if type == "rw_linear":
        # Do a linear based reweighting
        gate = (max_var - min_var) / (alpha*mean_var)
        gate_pos = 1 #torch.clamp(gate, max = 1.0)
        gate_neg = 0 #torch.clamp(1-gate, min =0.0)
        rescale_mu = gate_pos * rescale_mu + gate_neg * (1 - rescale_mu)
        rescale_var = gate_pos * rescale_var + gate_neg * (1 - rescale_var)
        extra.append(gate)

    if type == "rw_tanh":
        gate = torch.tanh(torch.std(var_clamped_detached, dim = 0) / (alpha * torch.sqrt(mean_var + cond)))
        gate = torch.clamp(gate, 0.0, 1.0)
        rescale_mu = gate * rescale_mu + (1 - gate) * (1 - rescale_mu)
        rescale_var = gate * rescale_var + (1 - gate) * (1 - rescale_var)
        extra.append(gate)

    if type == "rw_alpha_decay":
        gate = alpha
        rescale_mu = gate * rescale_mu + (1 - gate) * (1 - rescale_mu)
        rescale_var = gate * rescale_var + (1 - gate) * (1 - rescale_var)
        extra.append(gate)

    log_likelihood_mu_reweighting = (rescale_mu.squeeze())*log_likelihood
    log_likelihood_var_reweighting = (rescale_var.squeeze())*log_likelihood

    return -torch.mean(log_likelihood_mu_reweighting), -torch.mean(log_likelihood_var_reweighting), min_var, max_var, mean_var, extra

def loss_mse(mu, targets):
    se = (mu - targets.squeeze()) ** 2
    return torch.mean(se)

def loss_rmse(mu, targets):
    se = (mu.squeeze() - targets.squeeze()) ** 2
    mse = torch.mean(se)
    rmse = torch.sqrt(mse)
    return mse, rmse

def loss_nll(mu, var, targets):  # ensure variance is sufficiently large
    se = (targets.squeeze() - mu.squeeze()) ** 2
    nll = 0.5 * torch.log(2 * torch.pi * var.squeeze()) + 0.5 * se / var.squeeze()
    return torch.mean(nll)

def logmeanexp(inputs, dim=0):
    input_max = inputs.max(dim=dim)[0]
    return (inputs - input_max).exp().mean(dim=dim).log() + input_max