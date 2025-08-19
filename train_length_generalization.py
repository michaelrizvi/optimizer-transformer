#!/usr/bin/env python3
"""
Length generalization training script for small transformer models.
Uses CountSequenceDataset to test models on longer sequences than training.
"""

import torch
import os
import argparse
import random
import numpy as np
from utils import TransformerModels, calculate_transformer_metrics
from datasets import CountSequenceDataset
from fastargs import Section, Param, get_current_config
from fastargs.validation import OneOf
from fastargs.decorators import param, section

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with 'pip install wandb' to enable logging.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================================================================
# Configuration Sections - Defaults defined here
# ============================================================================

Section("experiment", "General experiment parameters").params(
    num_runs=Param(int, default=5, desc="Number of experimental runs"),
    device=Param(str, default="cpu", desc="Device to use (cpu/cuda)"),
    seed=Param(int, default=42, desc="Random seed for reproducibility"),
    save_model=Param(bool, default=False, desc="Save trained models after training"),
    save_dir=Param(str, default="saved_models", desc="Directory to save trained models")
)

Section("wandb", "Weights & Biases logging parameters").params(
    enabled=Param(bool, default=False, desc="Enable wandb logging"),
    project=Param(str, default="length-generalization", desc="WandB project name"),
    entity=Param(str, default=None, desc="WandB entity (username/team)"),
    offline=Param(bool, default=False, desc="Run wandb offline"),
    log_during_training=Param(bool, default=True, desc="Log metrics during training loops"),
    log_final_only=Param(bool, default=False, desc="Only log final results, not during training")
)

Section("dataset", "CountSequenceDataset parameters").params(
    # Training data length range
    train_min_range=Param(int, default=1, desc="Minimum sequence length for training"),
    train_max_range=Param(int, default=8, desc="Maximum sequence length for training"), 
    
    # Test/validation data length range
    test_min_range=Param(int, default=9, desc="Minimum sequence length for test/validation"),
    test_max_range=Param(int, default=16, desc="Maximum sequence length for test/validation"),
    
    # Dataset size parameters
    train_samples=Param(int, default=500, desc="Number of training samples"),
    val_samples=Param(int, default=200, desc="Number of validation samples"),
    test_samples=Param(int, default=200, desc="Number of test samples"),
    
    # Vocabulary parameters
    vocab_size=Param(int, default=32, desc="Vocabulary size"),  # Increased to accommodate sep/pad tokens
    sep_token=Param(int, default=30, desc="Separator token ID"),
    pad_token=Param(int, default=31, desc="Padding token ID"),
    max_length=Param(int, default=32, desc="Maximum sequence length (for padding)")
)

Section("model", "TransformerModels parameters").params(
    d_model=Param(int, default=32, desc="Model dimension"),
    n_layers=Param(int, default=2, desc="Number of transformer layers"),
    n_heads=Param(int, default=4, desc="Number of attention heads"),
    d_ff=Param(int, default=64, desc="Feed-forward dimension"),
    max_len=Param(int, default=64, desc="Maximum position encoding length"),
    dropout=Param(float, default=0.0, desc="Dropout probability"),
    model_count=Param(int, default=16, desc="Number of parallel models for pattern search"),
    init=Param(str, default="regular", desc="Parameter initialization method")
)

Section("optimizer", "Training parameters").params(
    name=Param(str, default="PatternSearchFast", desc="Optimizer name (SGD or PatternSearchFast)"),
    
    # SGD parameters
    lr=Param(float, default=0.001, desc="Learning rate"),
    momentum=Param(float, default=0.9, desc="SGD momentum"),
    batch_size=Param(int, default=128, desc="Batch size for SGD"),
    
    # Training parameters
    epochs=Param(int, default=400, desc="Maximum number of epochs"),
    es_acc=Param(float, default=0.99, desc="Early stopping accuracy threshold"),
    
    # Position offset parameters
    use_position_offsets=Param(bool, default=False, desc="Use position offsets during training"),
    max_position_offset=Param(int, default=16, desc="Maximum position offset"),
    
    # Evaluation parameters
    eval_frequency=Param(int, default=10, desc="Evaluate every N epochs"),
    
    # Cosine radius scheduler parameters
    use_cosine_radius_scheduler=Param(bool, default=False, desc="Use cosine scheduler for radius perturbation"),
    cosine_period=Param(int, default=100, desc="Period of cosine scheduler for radius (in epochs)")
)

# ============================================================================
# Dataset Creation Functions
# ============================================================================

@section('dataset')
@param('train_min_range')
@param('train_max_range')
@param('test_min_range')
@param('test_max_range')
@param('train_samples')
@param('val_samples')
@param('test_samples')
@param('vocab_size')
@param('sep_token')
@param('pad_token')
@param('max_length')
def create_datasets(train_min_range, train_max_range, test_min_range, test_max_range,
                   train_samples, val_samples, test_samples, vocab_size, sep_token, pad_token, max_length, run_seed=42):
    """Create train, validation, and test datasets with different sequence length ranges."""
    
    print(f"Creating datasets:")
    print(f"  Train: sequences length {train_min_range}-{train_max_range}, {train_samples} samples")
    print(f"  Val/Test: sequences length {test_min_range}-{test_max_range}, {val_samples}/{test_samples} samples")
    
    # Training dataset - shorter sequences
    train_dataset = CountSequenceDataset(
        n_samples=train_samples,
        min_range_size=train_min_range,
        max_range_size=train_max_range,
        vocab_size=vocab_size,
        sep_token=sep_token,
        pad_token=pad_token,
        seed=run_seed  # Use run-specific seed
    )
    
    # Validation dataset - longer sequences
    val_dataset = CountSequenceDataset(
        n_samples=val_samples,
        min_range_size=test_min_range,
        max_range_size=test_max_range,
        vocab_size=vocab_size,
        sep_token=sep_token,
        pad_token=pad_token,
        seed=run_seed + 1000  # Different but deterministic seed for val data
    )
    
    # Test dataset - longer sequences  
    test_dataset = CountSequenceDataset(
        n_samples=test_samples,
        min_range_size=test_min_range,
        max_range_size=test_max_range,
        vocab_size=vocab_size,
        sep_token=sep_token,
        pad_token=pad_token,
        seed=run_seed + 2000  # Different but deterministic seed for test data
    )
    
    # Convert to tensors
    from torch.nn.utils.rnn import pad_sequence
    
    train_sequences = [train_dataset[i] for i in range(len(train_dataset))]
    val_sequences = [val_dataset[i] for i in range(len(val_dataset))]
    test_sequences = [test_dataset[i] for i in range(len(test_dataset))]
    
    # Pad sequences to same length
    train_data = pad_sequence(train_sequences, batch_first=True, padding_value=pad_token).to(device)
    val_data = pad_sequence(val_sequences, batch_first=True, padding_value=pad_token).to(device)
    test_data = pad_sequence(test_sequences, batch_first=True, padding_value=pad_token).to(device)
    
    print(f"  Created datasets with shapes: train {train_data.shape}, val {val_data.shape}, test {test_data.shape}")
    
    return train_data, val_data, test_data

# ============================================================================
# Model Creation
# ============================================================================

@section('model')
@param('d_model')
@param('n_layers') 
@param('n_heads')
@param('d_ff')
@param('max_len')
@param('dropout')
@param('model_count')
@param('init')
@section('dataset')
@param('vocab_size')
@param('sep_token')
@param('pad_token')
def create_model(d_model, n_layers, n_heads, d_ff, max_len, dropout, model_count,
                vocab_size, sep_token, pad_token, device, init):
    """Create a TransformerModels instance."""
    
    model = TransformerModels(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_len=max_len,
        model_count=model_count,
        device=torch.device(device),
        dropout=dropout,
        sep_token_id=sep_token,
        pad_token_id=pad_token,
        init=init
    )
    
    print(f"Created model: d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}")
    print(f"  Model count: {model_count}, Device: {device}")
    
    return model

# ============================================================================
# Training Functions
# ============================================================================

@section('optimizer')
@param('name')
@param('lr')
@param('momentum') 
@param('batch_size')
@param('epochs')
@param('es_acc')
@param('use_position_offsets')
@param('max_position_offset')
@param('eval_frequency')
@param('use_cosine_radius_scheduler')
@param('cosine_period')
def train(train_data, val_data, test_data, model, name, lr, momentum, batch_size, epochs, 
          es_acc, use_position_offsets, max_position_offset, eval_frequency,
          use_cosine_radius_scheduler, cosine_period):
    """Train the model using specified optimizer."""
    
    print(f"\nStarting training with {name}")
    print(f"  Cosine radius scheduler: {use_cosine_radius_scheduler}")
    if use_cosine_radius_scheduler:
        print(f"  Cosine period: {cosine_period} epochs")
    
    # Move data to device
    device = next(model.parameters()).device
    train_data = train_data.to(device)
    val_data = val_data.to(device) 
    test_data = test_data.to(device)
    
    if name == "SGD":
        return train_sgd(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data, 
            model=model,
            lr=lr,
            momentum=momentum,
            batch_size=batch_size,
            epochs=epochs,
            es_acc=es_acc,
            use_position_offsets=use_position_offsets,
            max_position_offset=max_position_offset,
            eval_frequency=eval_frequency
        )
    elif name == "PatternSearchFast":
        return train_pattern_search(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            model=model,
            epochs=epochs,
            es_acc=es_acc,
            use_position_offsets=use_position_offsets,
            max_position_offset=max_position_offset,
            eval_frequency=eval_frequency,
            use_cosine_radius_scheduler=use_cosine_radius_scheduler,
            cosine_period=cosine_period
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")

def train_sgd(train_data, val_data, test_data, model, lr, momentum, batch_size, epochs, 
              es_acc, use_position_offsets, max_position_offset, eval_frequency):
    """SGD training with early stopping on validation set."""
    from utils import sample_position_offset, calculate_loss_acc_with_offset
    
    # Create optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//4, gamma=0.5)
    
    best_val_acc = 0.0
    for epoch in range(epochs):
        # Training loop
        model.train()
        idx_list = torch.randperm(len(train_data))
        
        for st_idx in range(0, len(train_data), batch_size):
            idx = idx_list[st_idx:min(st_idx + batch_size, len(train_data))]
            
            train_loss, train_acc, _ = calculate_transformer_metrics(train_data[idx], None, model, None)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Evaluation
        if epoch % eval_frequency == 0:
            model.eval()
            with torch.no_grad():
                # Standard evaluation
                train_loss, train_acc, train_em = calculate_transformer_metrics(train_data, None, model, None)
                val_loss, val_acc, val_em = calculate_transformer_metrics(val_data, None, model, None)
                
                # Wandb logging
                config = get_current_config()
                if WANDB_AVAILABLE and config['wandb.enabled'] and config['wandb.log_during_training'] and not config['wandb.log_final_only']:
                    wandb.log({
                        "epoch": epoch,
                        "train/loss": train_loss.item(),
                        "train/accuracy": train_acc.item(),
                        "train/exact_match": train_em.item(),
                        "val/loss": val_loss.item(),
                        "val/accuracy": val_acc.item(),
                        "val/exact_match": val_em.item(),
                        "learning_rate": scheduler.get_last_lr()[0]
                    })
                
                print(f"Epoch {epoch:3d} | Train: acc={train_acc.item():.3f}, em={train_em.item():.3f}, loss={train_loss.item():.3f} | \n"
                    f"Val: acc={val_acc.item():.3f}, em={val_em.item():.3f}, loss={val_loss.item():.3f} | "
                    )

                # Early stopping based on validation accuracy
                if val_acc.mean() >= es_acc:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    return model

@torch.no_grad()
def train_pattern_search(train_data, val_data, test_data, model, epochs, es_acc, 
                        use_position_offsets, max_position_offset, eval_frequency,
                        use_cosine_radius_scheduler, cosine_period):
    """Pattern search training with early stopping on validation set."""
    import math
    
    # Store initial radius for cosine scheduler
    initial_radius = model.radius
    
    for epoch in range(epochs):
        model.train()
        # Pattern search step
        model.pattern_search(train_data, None, None)
        
        # Evaluation
        if epoch % eval_frequency == 0:
            # Standard evaluation
            model.eval
            train_loss, train_acc, train_em = calculate_transformer_metrics(train_data, None, model, None)
            val_loss, val_acc, val_em = calculate_transformer_metrics(val_data, None, model, None)
            
            # Wandb logging
            config = get_current_config()
            if WANDB_AVAILABLE and config['wandb.enabled'] and config['wandb.log_during_training'] and not config['wandb.log_final_only']:
                metrics = {
                    "epoch": epoch,
                    "train/accuracy": train_acc.item(),
                    "train/exact_match": train_em.item(),
                    "val/accuracy": val_acc.item(),
                    "val/exact_match": val_em.item(),
                }
                # Add loss metrics if available
                metrics.update({
                    "train/loss": train_loss.item(),
                    "val/loss": val_loss.item(),
                })
                wandb.log(metrics)

            print(f"Epoch {epoch:3d} | Train: acc={train_acc.item():.3f}, em={train_em.item():.3f}, loss={train_loss.item():.3f} | \n"
                f"Val: acc={val_acc.item():.3f}, em={val_em.item():.3f}, loss={val_loss.item():.3f} | "
                )
            
            # Early stopping based on validation accuracy  
            if val_acc.item() >= es_acc:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Update radius with cosine scheduler at the end of every epoch
        if use_cosine_radius_scheduler:
            # Cosine decay schedule: starts at initial_radius, goes to 0, then back up
            cosine_factor = 0.5 * (1 + math.cos(math.pi * (epoch % cosine_period) / cosine_period))
            model.radius = initial_radius * cosine_factor
    
    return model.get_model_subsets([0]).to(next(model.parameters()).device)

# ============================================================================
# Main Experiment Loop
# ============================================================================

@section('experiment')
@param('num_runs')
@param('device')
@param('seed')
@param('save_model')
@param('save_dir')
def run_experiment(num_runs, device, seed, save_model, save_dir):
    """Run the length generalization experiment for num_runs."""
    
    print("=" * 80)
    print("LENGTH GENERALIZATION EXPERIMENT")
    print("=" * 80)
    
    # Log model saving configuration
    if save_model:
        print(f"Model saving: ENABLED (directory: {save_dir})")
        os.makedirs(save_dir, exist_ok=True)  # Create directory early to catch permission issues
    else:
        print("Model saving: DISABLED")
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    all_results = []
    
    # Get config (already parsed in main)
    config = get_current_config()
    
    for run in range(num_runs):
        print(f"\n{'='*20} RUN {run + 1}/{num_runs} {'='*20}")
        
        # Initialize wandb for this run
        if WANDB_AVAILABLE and config['wandb.enabled']:
            wandb_config = vars(config.get())
            wandb_config['run_number'] = run + 1
            
            # Set wandb mode
            mode = "offline" if config['wandb.offline'] else "online"
            
            # Create descriptive run name
            optimizer_name = config['optimizer.name']
            train_samples = config['dataset.train_samples']
            train_range = f"{config['dataset.train_min_range']}-{config['dataset.train_max_range']}"
            test_range = f"{config['dataset.test_min_range']}-{config['dataset.test_max_range']}"
            run_name = f"{optimizer_name}_s{train_samples}_train{train_range}_test{test_range}_run{run+1}"
            
            wandb.init(
                project=config['wandb.project'],
                entity=config['wandb.entity'],
                name=run_name,
                config=wandb_config,
                mode=mode,
                reinit=True
            )
        
        # Set run-specific seed
        run_seed = seed + run
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        random.seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)
        
        # Create datasets with run-specific seeds
        train_data, val_data, test_data = create_datasets(run_seed=run_seed)
        
        # Create model
        model = create_model(device=device)
        
        # Train model
        trained_model = train(train_data, val_data, test_data, model)
        
        # Final evaluation
        print(f"\nFinal evaluation for run {run + 1}:")
        with torch.no_grad():
            # Ensure model and data are on the same device
            model_device = next(trained_model.parameters()).device
            train_data = train_data.to(model_device)
            val_data = val_data.to(model_device)
            test_data = test_data.to(model_device)
            
            config = get_current_config()
            train_loss, train_acc, train_em = calculate_transformer_metrics(train_data, None, trained_model, None)
            val_loss, val_acc, val_em = calculate_transformer_metrics(val_data, None, trained_model, None)
            test_loss, test_acc, test_em= calculate_transformer_metrics(test_data, None, trained_model, None)
            
            results = {
                'run': run + 1,
                'train_acc': train_acc.mean().item(),
                'train_em': train_em.mean().item(),
                'val_acc': val_acc.mean().item(),
                'val_em': val_em.mean().item(), 
                'test_acc': test_acc.mean().item(),
                'test_em': test_em.mean().item()
            }
            
            # Log final results to wandb
            if WANDB_AVAILABLE and config['wandb.enabled']:
                wandb.log({
                    "final/train_accuracy": results['train_acc'],
                    "final/train_exact_match": results['train_em'],
                    "final/val_accuracy": results['val_acc'],
                    "final/val_exact_match": results['val_em'],
                    "final/test_accuracy": results['test_acc'],
                    "final/test_exact_match": results['test_em'],
                    "final/generalization_gap_acc": results['train_acc'] - results['test_acc'],
                    "final/generalization_gap_em": results['train_em'] - results['test_em']
                })
                
                # Finish this wandb run
                wandb.finish()
            
            all_results.append(results)
            
            # Save model if requested
            if save_model:
                # Create save directory if it doesn't exist
                os.makedirs(save_dir, exist_ok=True)
                
                # Create descriptive filename
                config = get_current_config()
                optimizer_name = config['optimizer.name']
                train_range = f"{config['dataset.train_min_range']}-{config['dataset.train_max_range']}"
                test_range = f"{config['dataset.test_min_range']}-{config['dataset.test_max_range']}"
                filename = f"{optimizer_name}_run{run+1}_train{train_range}_test{test_range}.pt"
                filepath = os.path.join(save_dir, filename)
                
                # Save the trained model
                torch.save(trained_model.state_dict(), filepath)
                print(f"  Model saved to: {filepath}")
            
            print(f"  Train: acc={results['train_acc']:.3f}, em={results['train_em']:.3f}")
            print(f"  Val:   acc={results['val_acc']:.3f}, em={results['val_em']:.3f}")
            print(f"  Test:  acc={results['test_acc']:.3f}, em={results['test_em']:.3f}")
    
    # Summary across all runs
    print(f"\n{'='*20} SUMMARY ACROSS {num_runs} RUNS {'='*20}")
    
    train_accs = [r['train_acc'] for r in all_results]
    val_accs = [r['val_acc'] for r in all_results]
    test_accs = [r['test_acc'] for r in all_results]
    train_ems = [r['train_em'] for r in all_results]
    val_ems = [r['val_em'] for r in all_results]
    test_ems = [r['test_em'] for r in all_results]
    
    print(f"Train Accuracy: {np.mean(train_accs):.3f} ± {np.std(train_accs):.3f}")
    print(f"Val Accuracy:   {np.mean(val_accs):.3f} ± {np.std(val_accs):.3f}")
    print(f"Test Accuracy:  {np.mean(test_accs):.3f} ± {np.std(test_accs):.3f}")
    print(f"Train EM:       {np.mean(train_ems):.3f} ± {np.std(train_ems):.3f}")
    print(f"Val EM:         {np.mean(val_ems):.3f} ± {np.std(val_ems):.3f}")
    print(f"Test EM:        {np.mean(test_ems):.3f} ± {np.std(test_ems):.3f}")
    
    # Length generalization analysis
    generalization_gap_acc = np.mean(train_accs) - np.mean(test_accs)
    generalization_gap_em = np.mean(train_ems) - np.mean(test_ems)
    
    # Log summary to wandb
    if WANDB_AVAILABLE and config['wandb.enabled']:
        # Create a summary run
        summary_run_name = f"{config['optimizer.name']}_s{config['dataset.train_samples']}_train{config['dataset.train_min_range']}-{config['dataset.train_max_range']}_test{config['dataset.test_min_range']}-{config['dataset.test_max_range']}_SUMMARY"
        
        mode = "offline" if config['wandb.offline'] else "online"
        wandb.init(
            project=config['wandb.project'],
            entity=config['wandb.entity'],
            name=summary_run_name,
            config=vars(config.get()),
            mode=mode,
            reinit=True
        )
        
        wandb.log({
            "summary/train_accuracy_mean": np.mean(train_accs),
            "summary/train_accuracy_std": np.std(train_accs),
            "summary/val_accuracy_mean": np.mean(val_accs),
            "summary/val_accuracy_std": np.std(val_accs),
            "summary/test_accuracy_mean": np.mean(test_accs),
            "summary/test_accuracy_std": np.std(test_accs),
            "summary/train_em_mean": np.mean(train_ems),
            "summary/train_em_std": np.std(train_ems),
            "summary/val_em_mean": np.mean(val_ems),
            "summary/val_em_std": np.std(val_ems),
            "summary/test_em_mean": np.mean(test_ems),
            "summary/test_em_std": np.std(test_ems),
            "summary/generalization_gap_acc": generalization_gap_acc,
            "summary/generalization_gap_em": generalization_gap_em,
            "summary/num_runs": num_runs
        })
        
        wandb.finish()
    
    print(f"\nLength Generalization Analysis:")
    print(f"  Accuracy gap (train - test): {generalization_gap_acc:+.3f}")
    print(f"  Exact match gap (train - test): {generalization_gap_em:+.3f}")
    
    if generalization_gap_acc < 0.1 and generalization_gap_em < 0.1:
        print("  ✓ Good length generalization!")
    elif generalization_gap_acc < 0.2 and generalization_gap_em < 0.2:
        print("  ~ Moderate length generalization")
    else:
        print("  ✗ Poor length generalization")

def main():
    """Main function with fastargs integration."""
    parser = argparse.ArgumentParser()
    config = get_current_config()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.summary()
    config_ns = config.get()
    # Use fastargs to handle configuration
    run_experiment()

if __name__ == "__main__":
    main()
