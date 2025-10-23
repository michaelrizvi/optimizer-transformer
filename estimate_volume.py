#!/usr/bin/env python3
"""
Estimate volume of Gaussian distribution around trained model weights.

This script implements the following algorithm to find the largest Gaussian
centered at trained weights w* that maintains good test performance:

Given:
  - A trained model with fixed weights w* = {w*_i}
  - A parameter vector ρ = {ρ_i}, initialized (e.g., ρ_i = log(σ₀) small)
  - Trade-off coefficient λ > 0
  - Number of Monte-Carlo samples K per update
  - Learning rate η_ρ for ρ

Repeat until convergence:
  1. For k = 1…K:
       • Sample ε^(k)_i ∼ Normal(0,1) independently for all i
       • Compute perturbed weights:
             w^(k)_i = w*_i + exp(ρ_i) · ε^(k)_i        # reparameterization
       • Evaluate model loss on test data (longer sequences):
             L^(k) = Loss_test( w^(k) )
  2. Compute empirical average loss:
       L̄ = (1/K) ∑_{k=1}^K L^(k)
  3. Compute volume penalty term:
       V = ∑_i ρ_i            # proportional to log(det Σ) for diagonal Σ
  4. Compute objective (to minimize):
       Obj = L̄ − λ · V
  5. Compute gradient ∇_{ρ} Obj via back-prop through w^(k) = w* + exp(ρ)·ε
  6. Update:
       ρ ← ρ − η_ρ · ∇_{ρ} Obj
"""

import torch
import argparse
import os
import random
import numpy as np
import csv
import math
from datetime import datetime
from tqdm import tqdm
from fastargs import get_current_config
from train_length_generalization import create_datasets
from utils import TransformerModels, calculate_transformer_metrics

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = str(device)

def set_random_seeds(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model_from_config(config, device):
    """Create a TransformerModels instance based on config parameters.

    Args:
        config: Configuration object from fastargs
        device: Device to place model on

    Returns:
        TransformerModels instance with model_count=1
    """
    model = TransformerModels(
        vocab_size=config['dataset.vocab_size'],
        d_model=config['model.d_model'],
        n_layers=config['model.n_layers'],
        n_heads=config['model.n_heads'],
        d_ff=config['model.d_ff'],
        max_len=config['model.max_len'],
        model_count=1,  # Single model for evaluation
        device=device,
        dropout=config['model.dropout'],
        sep_token_id=config['dataset.sep_token'],
        pad_token_id=config['dataset.pad_token'],
        init=config['model.init'],
        position_encoding_type=config['model.position_encoding_type'],
        rope_base=config['model.rope_base']
    )

    return model


def flatten_parameters(model):
    """Flatten all model parameters into a single 1D vector.

    Args:
        model: PyTorch model

    Returns:
        torch.Tensor: 1D tensor containing all parameters
    """
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    return torch.cat(params)


def unflatten_and_load_parameters(flat_params, model):
    """Restore flattened parameters back to model structure.

    Args:
        flat_params: 1D tensor containing all parameters
        model: PyTorch model to load parameters into
    """
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param.data.copy_(flat_params[offset:offset + numel].view_as(param.data))
        offset += numel


def save_results_to_csv(results_dict, filename):
    """Save volume estimation results to CSV file.

    Args:
        results_dict: Dictionary with results
        filename: Output CSV filename
    """
    with open(filename, 'w', newline='') as f:
        fieldnames = ['metric', 'value']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for key, value in results_dict.items():
            writer.writerow({'metric': key, 'value': value})


def main():
    """Main function to run volume estimation."""
    import sys

    # Parse our custom arguments FIRST before fastargs sees them
    our_parser = argparse.ArgumentParser(description="Estimate volume of Gaussian around trained model weights")
    our_parser.add_argument('--model_path', type=str,
                           default='saved_models/CountSequenceDataset_SGD_run1_train1-16_test17-32.pt',
                           help='Path to the trained model .pt file')
    our_parser.add_argument('--config', type=str, default='configs/length_gen_count_sgd.yaml',
                           help='Path to the configuration YAML file')
    our_parser.add_argument('--lambda_tradeoff', type=float, default=0.7,
                           help='Trade-off coefficient λ between loss and volume')
    our_parser.add_argument('--K', type=int, default=1000,
                           help='Number of Monte-Carlo samples per iteration')
    our_parser.add_argument('--lr_rho', type=float, default=0.001,
                           help='Learning rate for ρ parameters')
    our_parser.add_argument('--n_iterations', type=int, default=100,
                           help='Number of optimization iterations')
    our_parser.add_argument('--init_sigma', type=float, default=1e-5,
                           help='Initial standard deviation σ₀ (ρ_init = log(σ₀))')
    our_parser.add_argument('--save_results', action='store_true', default=False,
                           help='Whether to save results to CSV file')

    # Add fastargs arguments to the same parser
    config = get_current_config()
    config.augment_argparse(parser=our_parser)

    # Build custom argv that includes both our args and the config file
    filtered_argv = ['estimate_volume.py', '--config-file']

    # Find the config file value in sys.argv
    if '--config' in sys.argv:
        config_idx = sys.argv.index('--config')
        if config_idx + 1 < len(sys.argv):
            filtered_argv.append(sys.argv[config_idx + 1])
        else:
            filtered_argv.append('configs/length_gen_count_sgd.yaml')
    else:
        filtered_argv.append('configs/length_gen_count_sgd.yaml')

    # Parse with the modified argv
    original_argv = sys.argv.copy()
    sys.argv = filtered_argv
    config.collect_argparse_args(our_parser)
    sys.argv = original_argv

    # Now parse our custom args from the original argv
    args = our_parser.parse_args(original_argv[1:])

    config.summary()

    print("=" * 80)
    print("VOLUME ESTIMATION FOR TRAINED MODEL")
    print("=" * 80)
    print(f"Device: {device_name}")
    print(f"Loading model from: {args.model_path}")
    print(f"Loading config from: {args.config}")
    print(f"Trade-off coefficient λ: {args.lambda_tradeoff}")
    print(f"Monte-Carlo samples K: {args.K}")
    print(f"Learning rate η_ρ: {args.lr_rho}")
    print(f"Number of iterations: {args.n_iterations}")
    print(f"Initial σ₀: {args.init_sigma:.2e}")
    print()

    # Set random seeds for reproducibility (using seed=420 as specified)
    run_seed = 420
    set_random_seeds(run_seed)
    print(f"Set random seed to: {run_seed}")
    print()

    # Create datasets using the same approach as training script
    print(f"Creating {config['dataset.name']} datasets...")
    train_data, val_data, test_data = create_datasets(run_seed=run_seed)
    print(f"  Train data shape: {train_data.shape}")
    print(f"  Val data shape: {val_data.shape}")
    print(f"  Test data shape: {test_data.shape}")
    print()

    # Create model
    print("Creating model...")
    model = create_model_from_config(config, device)
    print(f"Model architecture: d_model={config['model.d_model']}, "
          f"n_layers={config['model.n_layers']}, "
          f"n_heads={config['model.n_heads']}, "
          f"d_ff={config['model.d_ff']}")
    print(f"Position encoding: {config['model.position_encoding_type']}")
    print()

    # Load trained model weights
    print(f"Loading trained weights from: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    state_dict = torch.load(args.model_path, map_location=device)

    # Check if state_dict has multiple models (model_count > 1)
    # If so, we need to extract model 0 before loading
    first_param_key = list(state_dict.keys())[0]
    first_param = state_dict[first_param_key]

    # Check if this is a multi-model state dict by looking at the first dimension
    if first_param.dim() >= 2 and first_param.size(0) > 1:
        print(f"Detected multi-model state dict (model_count={first_param.size(0)})")
        print("Extracting model 0...")

        # Create a temporary model with the same model_count to load the full state
        temp_model = TransformerModels(
            vocab_size=config['dataset.vocab_size'],
            d_model=config['model.d_model'],
            n_layers=config['model.n_layers'],
            n_heads=config['model.n_heads'],
            d_ff=config['model.d_ff'],
            max_len=config['model.max_len'],
            model_count=first_param.size(0),
            device=device,
            dropout=config['model.dropout'],
            sep_token_id=config['dataset.sep_token'],
            pad_token_id=config['dataset.pad_token'],
            init=config['model.init'],
            position_encoding_type=config['model.position_encoding_type'],
            rope_base=config['model.rope_base']
        )
        temp_model.load_state_dict(state_dict)

        # Extract model 0
        model = temp_model.get_model_subsets([0])
        print("Successfully extracted model 0")
    else:
        # Single model state dict
        model.load_state_dict(state_dict)
        print("Loaded single model state dict")

    model.to(device)
    print("Model weights loaded successfully")
    print()

    # Flatten parameters to get w*
    print("Flattening model parameters...")
    w_star = flatten_parameters(model)
    num_params = len(w_star)
    print(f"Total number of parameters: {num_params:,}")
    print()

    # Initialize ρ parameters with random noise to break symmetry
    print(f"Initializing ρ parameters with σ₀ = {args.init_sigma:.2e}...")
    rho_init = math.log(args.init_sigma)
    rho = torch.full((num_params,), rho_init, device=device)

    # Add small random noise to break symmetry (important for differentiation!)
    # This allows different parameters to explore different σ values
    noise_scale = 0.1  # 10% relative noise in log-space
    rho = rho + torch.randn(num_params, device=device) * noise_scale
    rho.requires_grad_(True)

    print(f"  ρ_init = log({args.init_sigma:.2e}) = {rho_init:.6f}")
    print(f"  Added Gaussian noise with scale {noise_scale:.3f} to break symmetry")
    print(f"  Initial ρ range: [{rho.min().item():.6f}, {rho.max().item():.6f}]")
    print()

    # Create optimizer for ρ
    optimizer = torch.optim.AdamW([rho], lr=args.lr_rho)
    print(f"Created AdamW optimizer with learning rate {args.lr_rho}")
    print()

    # Evaluate baseline performance with original weights
    print("Evaluating baseline model (w*)...")
    unflatten_and_load_parameters(w_star, model)
    model.eval()
    with torch.no_grad():
        train_loss, train_acc, train_em = calculate_transformer_metrics(train_data, None, model, None)
        val_loss, val_acc, val_em = calculate_transformer_metrics(val_data, None, model, None)
        test_loss, test_acc, test_em = calculate_transformer_metrics(test_data, None, model, None)

    baseline_test_loss = test_loss.item()
    print("Baseline performance:")
    print(f"  Train - loss: {train_loss.item():.3f}, acc: {train_acc.item():.3f}, EM: {train_em.item():.3f}")
    print(f"  Val   - loss: {val_loss.item():.3f}, acc: {val_acc.item():.3f}, EM: {val_em.item():.3f}")
    print(f"  Test  - loss: {test_loss.item():.3f}, acc: {test_acc.item():.3f}, EM: {test_em.item():.3f}")
    print()

    # Main optimization loop
    print("=" * 80)
    print("STARTING VOLUME OPTIMIZATION")
    print("=" * 80)
    print()

    for iteration in range(args.n_iterations):
        # Step 1 & 2: Sample K perturbations and evaluate
        sigma = torch.exp(rho)  # σ = exp(ρ)
        losses = []

        for k in range(args.K):
            # Sample ε^(k) ~ N(0,1)
            epsilon = torch.randn(num_params, device=device)

            # Reparameterization: w^(k) = w* + σ ⊙ ε^(k)
            w_k = w_star + sigma * epsilon

            # Load perturbed weights into model
            unflatten_and_load_parameters(w_k, model)

            # Evaluate on test set (DO NOT detach - let gradients flow)
            model.eval()
            test_loss_k, _, _ = calculate_transformer_metrics(test_data, None, model, None)
            losses.append(test_loss_k)

        # Step 3: Compute average loss L̄
        L_bar = torch.stack(losses).mean()

        # Step 4: Compute volume term V = Σ ρ_i
        V = rho.sum()

        # Step 5: Compute objective = L̄ - λ·V (minimize this)
        objective = L_bar - args.lambda_tradeoff * V

        # Step 6: Gradient descent update
        optimizer.zero_grad()
        objective.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_([rho], max_norm=1.0)

        optimizer.step()

        # Constrain ρ to prevent σ from growing too large
        # Cap σ at 1.0 (i.e., ρ <= log(1.0) = 0)
        with torch.no_grad():
            rho.clamp_(max=np.log(10.0))

        # Early stopping if loss explodes
        #if L_bar.item() > baseline_test_loss * 5.0:
        #    print(f"\nEarly stopping at iteration {iteration}:")
        #    print(f"  L̄ = {L_bar.item():.4f} exceeds 5x baseline ({baseline_test_loss * 5.0:.4f})")
        #    print(f"  Optimization diverging, stopping here.")
        #    break

        # Logging every 10 iterations
        if iteration % 10 == 0 or iteration == args.n_iterations - 1:
            # Evaluate original model w* on all splits (for reference)
            unflatten_and_load_parameters(w_star, model)
            model.eval()
            with torch.no_grad():
                train_loss_orig, train_acc_orig, train_em_orig = calculate_transformer_metrics(train_data, None, model, None)
                val_loss_orig, val_acc_orig, val_em_orig = calculate_transformer_metrics(val_data, None, model, None)
                test_loss_orig, test_acc_orig, test_em_orig = calculate_transformer_metrics(test_data, None, model, None)

            # Evaluate average performance of perturbed models
            # Sample one perturbed model to check its actual performance
            with torch.no_grad():
                epsilon_test = torch.randn(num_params, device=device)
                w_test = w_star + sigma.detach() * epsilon_test
                unflatten_and_load_parameters(w_test, model)
                model.eval()
                train_loss_pert, train_acc_pert, train_em_pert = calculate_transformer_metrics(train_data, None, model, None)
                val_loss_pert, val_acc_pert, val_em_pert = calculate_transformer_metrics(val_data, None, model, None)
                test_loss_pert, test_acc_pert, test_em_pert = calculate_transformer_metrics(test_data, None, model, None)

            # Compute current volume
            current_log_volume = V.item()
            current_volume_product = torch.exp(V).item() if V.item() < 700 else float('inf')

            # Compute variance statistics (σ² for each parameter)
            variances = sigma.detach() ** 2
            mean_var = variances.mean().item()
            min_var = variances.min().item()
            max_var = variances.max().item()
            mean_sigma = sigma.mean().item()
            min_sigma = sigma.min().item()
            max_sigma = sigma.max().item()

            print(f"Iteration {iteration:3d}/{args.n_iterations}:")
            print(f"  Objective: {objective.item():.4f}, L̄: {L_bar.item():.4f}, V: {V.item():.4f}")
            print(f"    Train - loss: {train_loss_pert.item():.3f}, acc: {train_acc_pert.item():.3f}, EM: {train_em_pert.item():.3f}")
            print(f"    Val   - loss: {val_loss_pert.item():.3f}, acc: {val_acc_pert.item():.3f}, EM: {val_em_pert.item():.3f}")
            print(f"    Test  - loss: {test_loss_pert.item():.3f}, acc: {test_acc_pert.item():.3f}, EM: {test_em_pert.item():.3f}")
            print(f"  Volume Statistics:")
            print(f"    Log-volume (Σρ): {current_log_volume:.4f}, Volume (Πσ): {current_volume_product:.4e}")
            print(f"    σ - Mean: {mean_sigma:.6f}, Min: {min_sigma:.6f}, Max: {max_sigma:.6f}")
            print(f"    σ² - Mean: {mean_var:.6f}, Min: {min_var:.6f}, Max: {max_var:.6f}")
            print()

    # Final results
    print("=" * 80)
    print("VOLUME ESTIMATION COMPLETE")
    print("=" * 80)
    print()

    # Compute final statistics
    final_sigma = torch.exp(rho.detach())
    final_log_volume = rho.detach().sum().item()
    final_volume_product = torch.exp(torch.tensor(final_log_volume)).item()

    print("Final Volume Statistics:")
    print(f"  Log-volume (Σ ρ_i): {final_log_volume:.6f}")
    print(f"  Volume (Π σ_i): {final_volume_product:.6e}")
    print(f"  Number of parameters: {num_params:,}")
    print(f"  Mean σ per parameter: {final_sigma.mean().item():.6e}")
    print(f"  Median σ: {final_sigma.median().item():.6e}")
    print(f"  Min σ: {final_sigma.min().item():.6e}")
    print(f"  Max σ: {final_sigma.max().item():.6e}")
    print()

    # Evaluate final model performance (should match baseline since we use w*)
    unflatten_and_load_parameters(w_star, model)
    model.eval()
    with torch.no_grad():
        train_loss, train_acc, train_em = calculate_transformer_metrics(train_data, None, model, None)
        val_loss, val_acc, val_em = calculate_transformer_metrics(val_data, None, model, None)
        test_loss, test_acc, test_em = calculate_transformer_metrics(test_data, None, model, None)

    print("Final model performance (w*):")
    print(f"  Train - loss: {train_loss.item():.3f}, acc: {train_acc.item():.3f}, EM: {train_em.item():.3f}")
    print(f"  Val   - loss: {val_loss.item():.3f}, acc: {val_acc.item():.3f}, EM: {val_em.item():.3f}")
    print(f"  Test  - loss: {test_loss.item():.3f}, acc: {test_acc.item():.3f}, EM: {test_em.item():.3f}")
    print()

    # Save results to CSV if requested
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"volume_estimation_lambda{args.lambda_tradeoff}_K{args.K}_{timestamp}.csv"

        results_dict = {
            'model_path': args.model_path,
            'lambda_tradeoff': args.lambda_tradeoff,
            'K_samples': args.K,
            'lr_rho': args.lr_rho,
            'n_iterations': args.n_iterations,
            'init_sigma': args.init_sigma,
            'num_parameters': num_params,
            'final_log_volume': final_log_volume,
            'final_volume_product': final_volume_product,
            'mean_sigma': final_sigma.mean().item(),
            'median_sigma': final_sigma.median().item(),
            'min_sigma': final_sigma.min().item(),
            'max_sigma': final_sigma.max().item(),
            'final_train_loss': train_loss.item(),
            'final_train_acc': train_acc.item(),
            'final_train_em': train_em.item(),
            'final_val_loss': val_loss.item(),
            'final_val_acc': val_acc.item(),
            'final_val_em': val_em.item(),
            'final_test_loss': test_loss.item(),
            'final_test_acc': test_acc.item(),
            'final_test_em': test_em.item(),
        }

        save_results_to_csv(results_dict, csv_filename)
        print(f"Results saved to: {csv_filename}")
        print()


if __name__ == "__main__":
    main()
