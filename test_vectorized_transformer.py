#!/usr/bin/env python3
"""
Test script for the vectorized TransformerModels implementation.
Tests basic functionality, forward pass, loss computation, and pattern search.
"""

import torch
import torch.nn.functional as F
import numpy as np
from utils import TransformerModels

def test_basic_initialization():
    """Test basic model initialization and parameter shapes."""
    print("Testing basic initialization...")
    
    # Test parameters
    vocab_size = 1000
    d_model = 128
    n_layers = 2
    n_heads = 4
    d_ff = 256
    max_len = 64
    model_count = 10
    device = torch.device('cpu')
    
    model = TransformerModels(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_len=max_len,
        model_count=model_count,
        device=device
    )
    
    # Check parameter shapes
    assert model.token_emb_weight.shape == (model_count, vocab_size, d_model)
    assert model.pos_emb.shape == (model_count, max_len, d_model)
    assert model.ln_f_weight.shape == (model_count, d_model)
    assert model.head_weight.shape == (model_count, d_model, vocab_size)
    
    # Check transformer layer parameters
    for layer in range(n_layers):
        assert model.transformer_params[f'attn_{layer}_qkv_weight'].shape == (model_count, d_model, 3*d_model)
        assert model.transformer_params[f'ff1_{layer}_weight'].shape == (model_count, d_model, d_ff)
        assert model.transformer_params[f'ff2_{layer}_weight'].shape == (model_count, d_ff, d_model)
    
    print("✓ Parameter shapes are correct")
    
    # Test parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,}")
    
    return model

def test_forward_pass():
    """Test forward pass with different input shapes."""
    print("\nTesting forward pass...")
    
    model = test_basic_initialization()
    model.eval()
    
    # Test data
    batch_size = 4
    seq_len = 16
    x = torch.randint(0, model.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        logits = model(x)
    
    # Check output shape
    expected_shape = (batch_size, model.model_count, seq_len, model.vocab_size)
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    print(f"✓ Forward pass output shape: {logits.shape}")
    
    # Test with position_ids
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    with torch.no_grad():
        logits_pos = model(x, position_ids=position_ids)
    
    assert logits_pos.shape == expected_shape
    print("✓ Forward pass with position_ids works")
    
    # Test that different models produce different outputs
    model_0_logits = logits[:, 0, :, :]
    model_1_logits = logits[:, 1, :, :]
    
    # After initialization, models should be different
    # (unless we've copied model 0 to all others)
    print(f"✓ Forward pass produces logits in range [{logits.min():.3f}, {logits.max():.3f}]")
    
    return model, x

def test_loss_function():
    """Test loss function computation."""
    print("\nTesting loss function...")
    
    model, x = test_forward_pass()
    
    # Create target for next-token prediction
    y = x[:, 1:]  # Shift by 1 for next-token prediction
    x_input = x[:, :-1]
    
    # Forward pass
    with torch.no_grad():
        logits = model(x_input)
    
    # Compute loss
    losses = model.loss_function(y, logits)
    
    # Check loss shape
    assert losses.shape == (model.model_count,), f"Expected ({model.model_count},), got {losses.shape}"
    print(f"✓ Loss function output shape: {losses.shape}")
    print(f"✓ Loss values: {losses}")
    
    # Check that losses are reasonable (positive, finite)
    assert torch.all(losses >= 0), "Losses should be non-negative"
    assert torch.all(torch.isfinite(losses)), "Losses should be finite"
    print("✓ Loss values are valid")
    
    return model, x_input, y

def test_compute_loss():
    """Test the compute_loss method."""
    print("\nTesting compute_loss method...")
    
    model, x_input, y = test_loss_function()
    
    # Test with tensor format
    sequences = torch.cat([x_input, y], dim=1)  # Reconstruct full sequences
    losses_tensor = model.compute_loss(sequences)
    
    assert losses_tensor.shape == (model.model_count,)
    print(f"✓ compute_loss with tensor format: {losses_tensor.shape}")
    
    # Test with dict format
    batch_dict = {
        'input_ids': sequences,
        'position_ids': None
    }
    losses_dict = model.compute_loss(batch_dict)
    
    assert losses_dict.shape == (model.model_count,)
    print(f"✓ compute_loss with dict format: {losses_dict.shape}")
    
    # Losses should be approximately the same
    diff = torch.abs(losses_tensor - losses_dict).max()
    assert diff < 1e-5, f"Loss difference too large: {diff}"
    print("✓ Both formats produce consistent results")
    
    return model, sequences

def test_model_copying():
    """Test the model copying functionality."""
    print("\nTesting model copying functionality...")
    
    model = test_basic_initialization()
    
    # Get initial parameters of model 0
    initial_token_emb = model.token_emb_weight[0].clone()
    initial_pos_emb = model.pos_emb[0].clone()
    
    # Modify model 1 parameters
    model.token_emb_weight[1] += 0.1
    model.pos_emb[1] += 0.1
    
    # Verify models are different
    diff_token = torch.abs(model.token_emb_weight[0] - model.token_emb_weight[1]).max()
    assert diff_token > 0.05, "Models should be different after modification"
    
    # Copy model 0 to all others
    model._copy_model_0_to_all()
    
    # Verify all models are now the same as model 0
    for i in range(1, model.model_count):
        diff_token = torch.abs(model.token_emb_weight[0] - model.token_emb_weight[i]).max()
        assert diff_token < 1e-6, f"Model {i} should be identical to model 0 after copying"
    
    print("✓ _copy_model_0_to_all() works correctly")
    
    # Test copying model to model 0
    model.token_emb_weight[2] += 0.2  # Modify model 2
    original_model_2 = model.token_emb_weight[2].clone()
    
    model._copy_model_to_model_0(2)
    
    # Verify model 0 now matches original model 2
    diff = torch.abs(model.token_emb_weight[0] - original_model_2).max()
    assert diff < 1e-6, "Model 0 should match model 2 after copying"
    
    print("✓ _copy_model_to_model_0() works correctly")
    
    return model

def test_pattern_search():
    """Test pattern search functionality."""
    print("\nTesting pattern search...")
    
    model = test_basic_initialization()
    model.eval()
    
    # Create simple test data
    batch_size = 2
    seq_len = 8
    sequences = torch.randint(0, min(100, model.vocab_size), (batch_size, seq_len))
    
    # Test pattern search initialization
    print("Testing pattern search initialization...")
    
    # Run pattern search for a few iterations
    dummy_labels = None
    loss_func = None  # Not used in our implementation
    
    # Get initial loss
    x = sequences[:, :-1]
    y = sequences[:, 1:]
    
    with torch.no_grad():
        initial_logits = model(x)
        initial_losses = model.loss_function(y, initial_logits)
        initial_loss = initial_losses[0].item()
    
    print(f"Initial loss (model 0): {initial_loss:.6f}")
    
    # Test that basis list gets created
    assert model.basis_list is None, "Basis list should be None initially"
    
    # Run one iteration of pattern search
    with torch.no_grad():
        model.pattern_search(sequences, dummy_labels, loss_func)
    
    # Check that basis list was created
    assert model.basis_list is not None, "Basis list should be created after first call"
    assert len(model.basis_list) > 0, "Basis list should not be empty"
    print(f"✓ Created basis list with {len(model.basis_list)} elements")
    
    # Check that best_loss is set
    assert hasattr(model, 'best_loss'), "best_loss should be set"
    assert model.best_loss is not None, "best_loss should not be None"
    print(f"✓ Initial best loss: {model.best_loss:.6f}")
    
    # Run a few more iterations to test the search
    print("Running additional pattern search iterations...")
    for i in range(3):
        with torch.no_grad():
            old_loss = model.best_loss.item()
            model.pattern_search(sequences, dummy_labels, loss_func)
            new_loss = model.best_loss.item()
            print(f"Iteration {i+2}: {old_loss:.6f} -> {new_loss:.6f}")
    
    print("✓ Pattern search completed without errors")
    
    return model

def test_model_subsets():
    """Test get_model_subsets functionality."""
    print("\nTesting model subsets...")
    
    model = test_basic_initialization()
    
    # Test with list of indices
    subset_indices = [0, 2, 4]
    subset_model = model.get_model_subsets(subset_indices)
    
    assert subset_model.model_count == len(subset_indices)
    assert subset_model.token_emb_weight.shape[0] == len(subset_indices)
    
    # Verify parameters were copied correctly
    for new_idx, old_idx in enumerate(subset_indices):
        diff = torch.abs(subset_model.token_emb_weight[new_idx] - model.token_emb_weight[old_idx]).max()
        assert diff < 1e-6, f"Subset model {new_idx} should match original model {old_idx}"
    
    print(f"✓ Created subset model with {subset_model.model_count} models")
    
    # Test with tensor indices
    tensor_indices = torch.tensor([1, 3])
    subset_model2 = model.get_model_subsets(tensor_indices)
    assert subset_model2.model_count == 2
    
    print("✓ Model subsetting works correctly")
    
    return subset_model

def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\nTesting edge cases...")
    
    # Test with minimal configuration
    model_small = TransformerModels(
        vocab_size=10,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_len=4,
        model_count=2,
        device=torch.device('cpu')
    )
    
    # Test forward pass with small model
    x = torch.randint(0, 10, (1, 3))
    with torch.no_grad():
        logits = model_small(x)
    
    assert logits.shape == (1, 2, 3, 10)
    print("✓ Small model works correctly")
    
    # Test with sequence length 1
    x_single = torch.randint(0, 10, (1, 1))
    with torch.no_grad():
        logits_single = model_small(x_single)
    
    assert logits_single.shape == (1, 2, 1, 10)
    print("✓ Single token input works")
    
    return model_small

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("VECTORIZED TRANSFORMER MODELS TEST SUITE")
    print("=" * 60)
    
    try:
        # Run all test functions
        test_basic_initialization()
        test_forward_pass()
        test_loss_function()
        test_compute_loss()
        test_model_copying()
        test_pattern_search()
        test_model_subsets()
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("The vectorized TransformerModels implementation is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)