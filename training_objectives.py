"""
Training objectives:
- ERM: Expected Risk Minimization (standard mean loss)
- TERM: Tilted Empirical Risk Minimization
"""
import torch


def erm_loss(per_sample_losses):
    """
    ERM: Expected Risk Minimization
    L = E[L]
    
    Args:
        per_sample_losses: Tensor of per-sample losses [batch_size]
    
    Returns:
        Mean loss value
    """
    return per_sample_losses.mean()


def term_loss(per_sample_losses, t):
    """
    TERM: Tilted Empirical Risk Minimization
    L = (1/t) * log(E[exp(t*L)])
    
    For t > 0: emphasizes high-loss samples (larger t = stronger emphasis)
    As t -> 0: approaches ERM (mean loss)
    As t -> inf: approaches max loss
    
    Args:
        per_sample_losses: Tensor of per-sample losses [batch_size]
        t: Tilt parameter (t > 0)
    
    Returns:
        TERM loss value (always >= mean loss by Jensen's inequality)
    """
    if t <= 0:
        raise ValueError("TERM tilt parameter t must be positive")
    
    mean_loss = per_sample_losses.mean()
    
    # For very small t, use Taylor expansion to avoid numerical instability
    loss_scale = mean_loss.item() if mean_loss.item() > 0 else 1.0
    if t * loss_scale < 1e-2:
        # Use second-order Taylor expansion for numerical stability
        mean_loss_sq = (per_sample_losses ** 2).mean()
        var_loss = mean_loss_sq - mean_loss ** 2
        return mean_loss + (t / 2.0) * var_loss
    
    # For larger t, use stable log-sum-exp trick
    max_loss = per_sample_losses.max()
    shifted_losses = per_sample_losses - max_loss
    
    exp_arg = t * shifted_losses
    exp_arg_clamped = torch.clamp(exp_arg, min=-50.0, max=0.0)
    exp_terms = torch.exp(exp_arg_clamped)
    
    mean_exp = exp_terms.mean()
    log_mean_exp = torch.log(mean_exp + 1e-10) + max_loss
    
    term_loss_value = (1.0 / t) * log_mean_exp
    
    # Ensure TERM >= ERM (by Jensen's inequality)
    return torch.maximum(term_loss_value, mean_loss)
