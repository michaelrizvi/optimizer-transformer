#!/usr/bin/env python3
"""
Test robustness of trained counting models by adding Gaussian noise to parameters.

This script:
1. Loads a trained model from a .pt file
2. Creates datasets using the same configuration and seeds as training
3. Evaluates the model before adding noise (baseline)
4. Adds Gaussian noise N(0, sigma^2) to all model parameters
5. Re-evaluates the model after adding noise
6. Saves results to CSV file
"""

import torch
import argparse
import os
import random
import numpy as np
import csv
from datetime import datetime
from tqdm import tqdm
from fastargs import get_current_config
from train_length_generalization import create_datasets
from utils import TransformerModels, calculate_transformer_metrics

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def add_gaussian_noise_to_parameters(model, sigma):
    """Add Gaussian noise N(0, sigma^2) to all model parameters in-place.

    Args:
        model: PyTorch model
        sigma: Standard deviation of Gaussian noise
    """
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * sigma
            param.data += noise


def evaluate_model(model, train_data, val_data, test_data):
    """Evaluate model on train, validation, and test sets.

    Args:
        model: TransformerModels instance
        train_data: Training data tensor
        val_data: Validation data tensor
        test_data: Test data tensor

    Returns:
        Dictionary with metrics for each split
    """
    model.eval()
    with torch.no_grad():
        # Ensure data is on the same device as model
        model_device = next(model.parameters()).device
        train_data = train_data.to(model_device)
        val_data = val_data.to(model_device)
        test_data = test_data.to(model_device)

        # Calculate metrics
        train_loss, train_acc, train_em = calculate_transformer_metrics(
            train_data, None, model, None
        )
        val_loss, val_acc, val_em = calculate_transformer_metrics(
            val_data, None, model, None
        )
        test_loss, test_acc, test_em = calculate_transformer_metrics(
            test_data, None, model, None
        )

        # Extract scalar values (metrics return tensors)
        results = {
            'train_loss': train_loss.item(),
            'train_acc': train_acc.item(),
            'train_em': train_em.item(),
            'val_loss': val_loss.item(),
            'val_acc': val_acc.item(),
            'val_em': val_em.item(),
            'test_loss': test_loss.item(),
            'test_acc': test_acc.item(),
            'test_em': test_em.item()
        }

    return results


def print_results(results, prefix=""):
    """Print evaluation results in a formatted way.

    Args:
        results: Dictionary with metrics
        prefix: Prefix string for the output
    """
    print(f"{prefix}")
    print(f"  Train - loss: {results['train_loss']:.3f}, "
          f"acc: {results['train_acc']:.3f}, "
          f"EM: {results['train_em']:.3f}")
    print(f"  Val   - loss: {results['val_loss']:.3f}, "
          f"acc: {results['val_acc']:.3f}, "
          f"EM: {results['val_em']:.3f}")
    print(f"  Test  - loss: {results['test_loss']:.3f}, "
          f"acc: {results['test_acc']:.3f}, "
          f"EM: {results['test_em']:.3f}")


def save_results_to_csv(before_results, after_results, sigma, filename):
    """Save results to CSV file (legacy single-run version).

    Args:
        before_results: Dictionary with metrics before noise
        after_results: Dictionary with metrics after noise
        sigma: Noise standard deviation
        filename: Output CSV filename
    """
    with open(filename, 'w', newline='') as f:
        fieldnames = ['condition', 'sigma', 'train_loss', 'train_acc', 'train_em',
                      'val_loss', 'val_acc', 'val_em', 'test_loss', 'test_acc', 'test_em']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        # Write before noise results
        row_before = {'condition': 'before_noise', 'sigma': 0.0}
        row_before.update(before_results)
        writer.writerow(row_before)

        # Write after noise results
        row_after = {'condition': 'after_noise', 'sigma': sigma}
        row_after.update(after_results)
        writer.writerow(row_after)


def save_summary_results_to_csv(mean_stats, std_stats, sigma, n_runs, filename):
    """Save summary statistics (mean/std) to CSV file.

    Args:
        mean_stats: Dictionary with mean statistics
        std_stats: Dictionary with std statistics
        sigma: Noise standard deviation
        n_runs: Number of runs performed
        filename: Output CSV filename
    """
    with open(filename, 'w', newline='') as f:
        fieldnames = ['metric', 'statistic', 'sigma', 'n_runs',
                      'train_loss', 'train_acc', 'train_em',
                      'val_loss', 'val_acc', 'val_em',
                      'test_loss', 'test_acc', 'test_em',
                      'train_delta_acc', 'train_delta_em',
                      'val_delta_acc', 'val_delta_em',
                      'test_delta_acc', 'test_delta_em']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        # Write mean statistics
        row_mean = {
            'metric': 'after_noise',
            'statistic': 'mean',
            'sigma': sigma,
            'n_runs': n_runs
        }
        row_mean.update(mean_stats)
        writer.writerow(row_mean)

        # Write std statistics
        row_std = {
            'metric': 'after_noise',
            'statistic': 'std',
            'sigma': sigma,
            'n_runs': n_runs
        }
        row_std.update(std_stats)
        writer.writerow(row_std)


def main():
    """Main function to run robustness test."""
    import sys

    # Parse our custom arguments FIRST before fastargs sees them
    our_parser = argparse.ArgumentParser(description="Test model robustness by adding Gaussian noise")
    our_parser.add_argument('--model_path', type=str, default='saved_models/CountSequenceDataset_SGD_run1_train1-16_test17-32.pt',
                           help='Path to the trained model .pt file')
    our_parser.add_argument('--config', type=str, default='configs/length_gen_count_sgd.yaml',
                           help='Path to the configuration YAML file')
    our_parser.add_argument('--sigma', type=float, default=0.05,
                           help='Standard deviation of Gaussian noise')
    our_parser.add_argument('--n_runs', type=int, default=10,
                           help='Number of runs to perform with different noise samples')
    our_parser.add_argument('--save_results', action='store_true', default=False,
                           help='Whether to save results to CSV file')
    our_parser.add_argument('--dataset_name', type=str, default='CountSequenceDataset',
                           help='Name of the dataset class to use')

    # Add fastargs arguments to the same parser
    config = get_current_config()
    config.augment_argparse(parser=our_parser)

    # Build custom argv that includes both our args and the config file
    # Filter out our custom args before passing to fastargs
    our_args = ['--model_path', '--config', '--sigma', '--save_results', '--dataset_name']

    # Create a filtered argv for fastargs by including only config-file
    filtered_argv = ['test_robustness.py', '--config-file']

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
    print("MODEL ROBUSTNESS TEST - GAUSSIAN NOISE")
    print("=" * 80)
    print(f"Loading model from: {args.model_path}")
    print(f"Loading config from: {args.config}")
    print(f"Noise standard deviation (σ): {args.sigma}")
    print(f"Number of runs: {args.n_runs}")
    print(f"Dataset: {args.dataset_name}")
    print()

    # Set random seeds for dataset creation (using seed=420 as specified)
    run_seed = 420
    set_random_seeds(run_seed)
    print(f"Set random seed to: {run_seed}")
    print()

    # Create datasets using the same approach as training script
    print(f"Creating {config['dataset.name']} datasets...")
    train_data, val_data, test_data = create_datasets(run_seed=run_seed)
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

    # Load trained model weights once to get baseline
    print(f"Loading trained weights from: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print("Model weights loaded successfully")
    print()

    # Evaluate BEFORE adding noise (baseline - only need to do once)
    print("Evaluating baseline model (before noise)...")
    before_results = evaluate_model(model, train_data, val_data, test_data)
    print_results(before_results, prefix="Baseline (before noise):")
    print()

    # Run experiment n_runs times with different noise samples
    print(f"Running robustness test for {args.n_runs} runs with different noise samples...")
    print()

    all_after_results = []
    all_degradations = []

    for run_idx in tqdm(range(args.n_runs), desc="Robustness runs"):
        # Set different seed for each run (for different noise patterns)
        noise_seed = run_seed + run_idx + 1000
        set_random_seeds(noise_seed)

        # Reload model weights from .pt file (reset to trained state)
        model.load_state_dict(state_dict)
        model.to(device)

        # Add Gaussian noise with current seed
        add_gaussian_noise_to_parameters(model, args.sigma)

        # Evaluate after adding noise
        after_results = evaluate_model(model, train_data, val_data, test_data)
        all_after_results.append(after_results)

        # Calculate degradations for this run
        degradation = {
            'train_delta_acc': before_results['train_acc'] - after_results['train_acc'],
            'train_delta_em': before_results['train_em'] - after_results['train_em'],
            'val_delta_acc': before_results['val_acc'] - after_results['val_acc'],
            'val_delta_em': before_results['val_em'] - after_results['val_em'],
            'test_delta_acc': before_results['test_acc'] - after_results['test_acc'],
            'test_delta_em': before_results['test_em'] - after_results['test_em']
        }
        all_degradations.append(degradation)

    print()
    print("All runs completed!")
    print()

    # Calculate mean and std statistics across all runs
    metric_keys = ['train_loss', 'train_acc', 'train_em',
                   'val_loss', 'val_acc', 'val_em',
                   'test_loss', 'test_acc', 'test_em']
    degradation_keys = ['train_delta_acc', 'train_delta_em',
                       'val_delta_acc', 'val_delta_em',
                       'test_delta_acc', 'test_delta_em']

    # Compute mean statistics
    mean_stats = {}
    for key in metric_keys:
        values = [result[key] for result in all_after_results]
        mean_stats[key] = np.mean(values)

    for key in degradation_keys:
        values = [deg[key] for deg in all_degradations]
        mean_stats[key] = np.mean(values)

    # Compute std statistics
    std_stats = {}
    for key in metric_keys:
        values = [result[key] for result in all_after_results]
        std_stats[key] = np.std(values)

    for key in degradation_keys:
        values = [deg[key] for deg in all_degradations]
        std_stats[key] = np.std(values)

    # Print summary statistics
    print("=" * 80)
    print(f"SUMMARY STATISTICS (n={args.n_runs} runs, σ={args.sigma})")
    print("=" * 80)
    print()
    print("After noise performance (mean ± std):")
    print(f"  Train - loss: {mean_stats['train_loss']:.3f} ± {std_stats['train_loss']:.3f}, "
          f"acc: {mean_stats['train_acc']:.3f} ± {std_stats['train_acc']:.3f}, "
          f"EM: {mean_stats['train_em']:.3f} ± {std_stats['train_em']:.3f}")
    print(f"  Val   - loss: {mean_stats['val_loss']:.3f} ± {std_stats['val_loss']:.3f}, "
          f"acc: {mean_stats['val_acc']:.3f} ± {std_stats['val_acc']:.3f}, "
          f"EM: {mean_stats['val_em']:.3f} ± {std_stats['val_em']:.3f}")
    print(f"  Test  - loss: {mean_stats['test_loss']:.3f} ± {std_stats['test_loss']:.3f}, "
          f"acc: {mean_stats['test_acc']:.3f} ± {std_stats['test_acc']:.3f}, "
          f"EM: {mean_stats['test_em']:.3f} ± {std_stats['test_em']:.3f}")
    print()
    print("Performance degradation (mean ± std):")
    print(f"  Train - Δacc: {mean_stats['train_delta_acc']:.3f} ± {std_stats['train_delta_acc']:.3f}, "
          f"ΔEM: {mean_stats['train_delta_em']:.3f} ± {std_stats['train_delta_em']:.3f}")
    print(f"  Val   - Δacc: {mean_stats['val_delta_acc']:.3f} ± {std_stats['val_delta_acc']:.3f}, "
          f"ΔEM: {mean_stats['val_delta_em']:.3f} ± {std_stats['val_delta_em']:.3f}")
    print(f"  Test  - Δacc: {mean_stats['test_delta_acc']:.3f} ± {std_stats['test_delta_acc']:.3f}, "
          f"ΔEM: {mean_stats['test_delta_em']:.3f} ± {std_stats['test_delta_em']:.3f}")
    print()

    # Save results to CSV
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"robustness_summary_sigma{args.sigma}_n{args.n_runs}_{timestamp}.csv"
        save_summary_results_to_csv(mean_stats, std_stats, args.sigma, args.n_runs, csv_filename)
        print(f"Summary results saved to: {csv_filename}")
        print()

    print("=" * 80)
    print("Robustness test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
