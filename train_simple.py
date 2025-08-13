#!/usr/bin/env python3
"""
Simplified training script for transformer models on counting task.
Focuses on model filtering based on test accuracy threshold.
"""

import torch
import os
import json
import time
import argparse
from tqdm import tqdm
from collections import defaultdict

# Imports from existing codebase
from utils import TransformerModels, calculate_transformer_loss_acc
from datasets import CountSequenceDataset
from optimizer import PatternSearch
from fastargs import get_current_config
from fastargs.decorators import param, section
from fastargs import Section, Param
from fastargs.validation import OneOf

# Config sections - simplified from train_distributed.py
Section("dataset", "Dataset parameters").params(
    name=Param(str, default="counting"),
)
Section("dataset.counting", "Dataset parameters for counting task").params(
    min_range_size=Param(int, default=1),
    max_range_size=Param(int, default=10),
    vocab_size=Param(int, default=20),
    sep_token=Param(int, default=102),
    pad_token=Param(int, default=103)
)
Section("model", "Model architecture parameters").params(
    arch=Param(str, default="transformer"),
    model_count_times_batch_size=Param(int, default=1000),
    init=Param(str, OneOf(("uniform", "regular")), default="regular")
)
Section("model.transformer", "Transformer model parameters").params(
    d_model=Param(int, default=32),
    n_layers=Param(int, default=2),
    n_heads=Param(int, default=4),
    d_ff=Param(int, default=128),
    max_len=Param(int, default=64),
    dropout=Param(float, default=0.1)
)
Section("optimizer").params(
    name=Param(str, OneOf(["SGD", "PatternSearchFast"]), default='SGD'),
    lr=Param(float, default=0.001),
    momentum=Param(float, default=0.9),
    epochs=Param(int, default=400),
    batch_size=Param(int, default=100),
    scheduler=Param(bool, default=True),
)
Section("training").params(
    test_acc_threshold=Param(float, default=0.9, desc="Test accuracy threshold for model filtering"),
    max_epochs=Param(int, default=1000, desc="Maximum epochs before giving up"),
    save_frequency=Param(int, default=10, desc="Save results every N epochs"),
    device=Param(str, default="cuda" if torch.cuda.is_available() else "cpu"),
    seed=Param(int, default=42)
)
Section("output").params(
    results_file=Param(str, default="results.json", desc="File to save training results"),
    log_file=Param(str, default="training.log", desc="File to save training logs")
)

def get_optimizer_and_scheduler(name, model, scheduler=False, lr=None, momentum=0):
    """Get optimizer and scheduler - simplified from train_distributed.py"""
    if name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif name == "PatternSearchFast":
        # Use the PatternSearch from optimizer.py 
        optimizer = PatternSearch(model.parameters())
    else:
        raise ValueError(f"Unknown optimizer: {name}")
    
    if scheduler and name != "PatternSearchFast":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    else:
        scheduler = None
        
    return optimizer, scheduler

def train_sgd(train_data, test_data, model, loss_func, optimizer, scheduler, 
              batch_size, epochs, test_acc_threshold, save_frequency, results_file):
    """SGD training with model filtering based on test accuracy"""
    
    results = defaultdict(list)
    surviving_models = set(range(model.model_count))
    
    print(f"Starting SGD training with {model.model_count} models")
    print(f"Test accuracy threshold: {test_acc_threshold}")
    
    for epoch in range(epochs):
        # Training step
        idx_list = torch.randperm(len(train_data))
        for st_idx in range(0, len(train_data), batch_size):
            idx = idx_list[st_idx:min(st_idx + batch_size, len(train_data))]
            train_loss, train_acc = loss_func(train_data[idx], None, model, None)
            
            optimizer.zero_grad()
            train_loss.sum().backward()
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Evaluation and model filtering
        if epoch % save_frequency == 0 or epoch == epochs - 1:
            with torch.no_grad():
                train_loss, train_acc = loss_func(train_data, None, model, None)
                test_loss, test_acc = loss_func(test_data, None, model, None)
                
                # Filter models based on test accuracy
                models_to_remove = []
                for model_idx in surviving_models:
                    if test_acc[model_idx].item() >= test_acc_threshold:
                        models_to_remove.append(model_idx)
                        print(f"Model {model_idx} reached threshold {test_acc_threshold:.3f} "
                              f"with test acc {test_acc[model_idx].item():.4f}")
                
                # Remove models that reached threshold
                for model_idx in models_to_remove:
                    surviving_models.remove(model_idx)
                
                # Save results for all models (including removed ones)
                results['epoch'].append(epoch)
                results['train_loss'].append(train_loss.cpu().tolist())
                results['test_loss'].append(test_loss.cpu().tolist())
                results['train_acc'].append(train_acc.cpu().tolist())
                results['test_acc'].append(test_acc.cpu().tolist())
                results['surviving_models'].append(list(surviving_models))
                
                print(f"Epoch {epoch}: Surviving models: {len(surviving_models)}")
                if len(surviving_models) > 0:
                    remaining_test_acc = test_acc[[model_idx for model_idx in surviving_models]]
                    print(f"  Best remaining test acc: {remaining_test_acc.max().item():.4f}")
                    print(f"  Worst remaining test acc: {remaining_test_acc.min().item():.4f}")
                
                # Save intermediate results
                with open(results_file, 'w') as f:
                    json.dump(dict(results), f, indent=2)
                
                # Check termination condition
                if len(surviving_models) == 0:
                    print("All models reached the test accuracy threshold!")
                    break
                
                # Update model to only include surviving models (optional optimization)
                # For simplicity, we keep all models but just track which ones are "active"
    
    return dict(results)

def train_ps_fast(train_data, test_data, model, loss_func, optimizer, 
                  epochs, test_acc_threshold, save_frequency, results_file):
    """Pattern Search training with model filtering"""
    
    results = defaultdict(list)
    surviving_models = set(range(model.model_count))
    
    print(f"Starting Pattern Search training with {model.model_count} models")
    print(f"Test accuracy threshold: {test_acc_threshold}")
    
    for epoch in range(epochs):
        # Pattern search step - uses model.pattern_search internally
        model.pattern_search(train_data, None, None)  # loss_func not used in our implementation
        
        # Evaluation and model filtering
        if epoch % save_frequency == 0 or epoch == epochs - 1:
            with torch.no_grad():
                train_loss, train_acc = loss_func(train_data, None, model, None)
                test_loss, test_acc = loss_func(test_data, None, model, None)
                
                # Filter models based on test accuracy
                models_to_remove = []
                for model_idx in surviving_models:
                    if test_acc[model_idx].item() >= test_acc_threshold:
                        models_to_remove.append(model_idx)
                        print(f"Model {model_idx} reached threshold {test_acc_threshold:.3f} "
                              f"with test acc {test_acc[model_idx].item():.4f}")
                
                # Remove models that reached threshold
                for model_idx in models_to_remove:
                    surviving_models.remove(model_idx)
                
                # Save results for all models
                results['epoch'].append(epoch)
                results['train_loss'].append(train_loss.cpu().tolist())
                results['test_loss'].append(test_loss.cpu().tolist())
                results['train_acc'].append(train_acc.cpu().tolist())
                results['test_acc'].append(test_acc.cpu().tolist())
                results['surviving_models'].append(list(surviving_models))
                
                print(f"Epoch {epoch}: Surviving models: {len(surviving_models)}")
                if len(surviving_models) > 0:
                    remaining_test_acc = test_acc[[model_idx for model_idx in surviving_models]]
                    print(f"  Best remaining test acc: {remaining_test_acc.max().item():.4f}")
                    print(f"  Worst remaining test acc: {remaining_test_acc.min().item():.4f}")
                
                # Save intermediate results
                with open(results_file, 'w') as f:
                    json.dump(dict(results), f, indent=2)
                
                # Check termination condition
                if len(surviving_models) == 0:
                    print("All models reached the test accuracy threshold!")
                    break
    
    return dict(results)

@section('dataset')
@param('counting.min_range_size')
@param('counting.max_range_size') 
@param('counting.vocab_size')
@param('counting.sep_token')
@param('counting.pad_token')
def get_dataset(min_range_size=None, max_range_size=None, vocab_size=None, 
                sep_token=None, pad_token=None):
    """Get counting dataset"""
    
    # Create train and test datasets
    train_dataset = CountSequenceDataset(
        min_range_size=min_range_size,
        max_range_size=max_range_size,
        vocab_size=vocab_size,
        sep_token_id=sep_token,
        pad_token_id=pad_token,
        num_samples=1000,  # Fixed for simplicity
        max_length=64
    )
    
    test_dataset = CountSequenceDataset(
        min_range_size=min_range_size,
        max_range_size=max_range_size,
        vocab_size=vocab_size,
        sep_token_id=sep_token,
        pad_token_id=pad_token,
        num_samples=200,   # Smaller test set
        max_length=64
    )
    
    return train_dataset, test_dataset

@section('model')
@param('model_count_times_batch_size')
@param('transformer.d_model')
@param('transformer.n_layers')
@param('transformer.n_heads')
@param('transformer.d_ff')
@param('transformer.max_len')
@param('transformer.dropout')
@param('dataset.counting.vocab_size')
@param('dataset.counting.sep_token')
@param('dataset.counting.pad_token')
@param('training.device')
def get_model(model_count_times_batch_size=None, d_model=None, n_layers=None, n_heads=None,
              d_ff=None, max_len=None, dropout=None, vocab_size=None, sep_token=None,
              pad_token=None, device=None):
    """Create transformer model"""
    
    # Calculate actual model count (simplified)
    model_count = model_count_times_batch_size // 100  # Assume batch size of 100
    
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
        pad_token_id=pad_token
    )
    
    return model.to(device)

def main():
    """Main training loop"""
    
    # Parse config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    
    # Parse args and load config
    config = get_current_config()
    config.augment_argparse(parser)
    config.collect()
    
    # Set random seed
    torch.manual_seed(config['training.seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['training.seed'])
    
    # Get dataset
    print("Loading dataset...")
    train_dataset, test_dataset = get_dataset()
    train_data = train_dataset.data
    test_data = test_dataset.data
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Get model
    print("Creating model...")
    model = get_model()
    print(f"Model created with {model.model_count} models")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(
        name=config['optimizer.name'],
        model=model,
        scheduler=config['optimizer.scheduler'],
        lr=config['optimizer.lr'],
        momentum=config['optimizer.momentum']
    )
    
    print(f"Using optimizer: {config['optimizer.name']}")
    
    # Setup results file
    os.makedirs(os.path.dirname(config['output.results_file']), exist_ok=True)
    
    # Training
    start_time = time.time()
    
    if config['optimizer.name'] == 'SGD':
        results = train_sgd(
            train_data=train_data,
            test_data=test_data,
            model=model,
            loss_func=calculate_transformer_loss_acc,
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=config['optimizer.batch_size'],
            epochs=min(config['optimizer.epochs'], config['training.max_epochs']),
            test_acc_threshold=config['training.test_acc_threshold'],
            save_frequency=config['training.save_frequency'],
            results_file=config['output.results_file']
        )
    elif config['optimizer.name'] == 'PatternSearchFast':
        results = train_ps_fast(
            train_data=train_data,
            test_data=test_data,
            model=model,
            loss_func=calculate_transformer_loss_acc,
            optimizer=optimizer,
            epochs=min(config['optimizer.epochs'], config['training.max_epochs']),
            test_acc_threshold=config['training.test_acc_threshold'],
            save_frequency=config['training.save_frequency'],
            results_file=config['output.results_file']
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer.name']}")
    
    training_time = time.time() - start_time
    
    # Save final results with metadata
    final_results = {
        'config': dict(config),
        'training_time': training_time,
        'results': results
    }
    
    with open(config['output.results_file'], 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Results saved to: {config['output.results_file']}")
    
    # Print summary
    if len(results['epoch']) > 0:
        final_epoch = results['epoch'][-1]
        final_surviving = results['surviving_models'][-1]
        print(f"Final epoch: {final_epoch}")
        print(f"Models that reached threshold: {model.model_count - len(final_surviving)}")
        print(f"Remaining models: {len(final_surviving)}")

if __name__ == "__main__":
    main()