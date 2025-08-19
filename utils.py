import torch
import matplotlib.pyplot as plt
from torch import nn
from matplotlib.colors import ListedColormap
import numpy as np
import torch.nn.functional as F
from itertools import chain
import random
from transformer import DecoderOnlyTransformer

def sample_position_offset(max_offset, seq_len, max_len):
    """
    Sample random positional offset ensuring seq_len + offset <= max_len.
    
    Args:
        max_offset: Maximum desired offset
        seq_len: Length of sequence
        max_len: Maximum positional encoding length
        
    Returns:
        int: Random offset in valid range [0, min(max_offset, max_len - seq_len)]
             Returns 0 if seq_len >= max_len (no valid offset possible)
    """
    if seq_len >= max_len:
        # No valid offset possible - sequence too long for max_len
        return 0
        
    max_valid_offset = min(max_offset, max_len - seq_len)
    if max_valid_offset <= 0:
        return 0
    return torch.randint(0, max_valid_offset + 1, (1,)).item()

def evaluate_with_offsets(data, labels, model, loss_func, num_offset_tests=5, batch_size=None):
    """
    Test model performance across different positional offsets.
    
    Args:
        data: Input sequences
        labels: Target labels (ignored for transformer counting task)
        model: Transformer model to evaluate
        loss_func: Loss function (ignored - uses model's loss function)
        num_offset_tests: Number of different offsets to test
        batch_size: Optional batch size for evaluation
        
    Returns:
        tuple: (average_losses, average_accuracies) across offsets
    """
    max_offset = max(0, model.max_len - data.size(1))
    
    # Test at evenly spaced offsets
    if max_offset == 0:
        # No room for offsets, just use 0
        test_offsets = [0]
    else:
        test_offsets = [int(i * max_offset / max(1, num_offset_tests - 1)) 
                       for i in range(min(num_offset_tests, max_offset + 1))]
    
    all_losses = []
    all_accs = []
    
    for offset in test_offsets:
        if batch_size is None:
            x = data[:, :-1]
            y = data[:, 1:]
            
            with torch.no_grad():
                logits = model(x, position_offset=offset)
                loss = model.loss_function(y, logits)
                acc = calculate_counting_accuracy(y, logits, model.sep_token_id, model.pad_token_id)
                
            all_losses.append(loss)
            all_accs.append(acc)
        else:
            # Batched evaluation
            batch_losses = []
            batch_accs = []
            
            for i in range(0, len(data), batch_size):
                batch_data = data[i:min(i+batch_size, len(data))]
                x = batch_data[:, :-1]
                y = batch_data[:, 1:]
                
                with torch.no_grad():
                    logits = model(x, position_offset=offset)
                    loss = model.loss_function(y, logits)
                    acc = calculate_counting_accuracy(y, logits, model.sep_token_id, model.pad_token_id)
                
                batch_losses.append(loss)
                batch_accs.append(acc)
            
            # Average across batches
            all_losses.append(torch.stack(batch_losses).mean(dim=0))
            all_accs.append(torch.stack(batch_accs).mean(dim=0))
    
    # Return average across offsets
    if len(all_losses) > 1:
        avg_loss = torch.stack(all_losses).mean(dim=0)
        avg_acc = torch.stack(all_accs).mean(dim=0)
        return avg_loss, avg_acc
    else:
        return all_losses[0], all_accs[0]

def calculate_loss_acc_with_offset(data, labels, model, loss_func, position_offset, batch_size=None):
    """
    Calculate loss and accuracy with a specific positional offset.
    Similar to calculate_transformer_loss_acc but uses position_offset.
    """
    if batch_size is None:
        # For next-token prediction: input = seq[:-1], target = seq[1:]
        x = data[:, :-1]  # All tokens except last
        y = data[:, 1:]   # All tokens except first (shifted by 1)
        
        logits = model(x, position_offset=position_offset)  # (batch_size, model_count, seq_len, vocab_size)
        
        # Use model's loss function which handles padding
        loss = model.loss_function(y, logits)  # (model_count,)
        
        # Calculate token-wise accuracy on answer portion only
        acc = calculate_counting_accuracy(y, logits, model.sep_token_id, model.pad_token_id)
        
        return loss, acc
    else:
        # Batched processing
        all_losses = []
        all_accs = []
        
        for i in range(0, len(data), batch_size):
            batch_data = data[i:min(i+batch_size, len(data))]
            
            x = batch_data[:, :-1]
            y = batch_data[:, 1:]
            
            logits = model(x, position_offset=position_offset)
            loss = model.loss_function(y, logits)
            acc = calculate_counting_accuracy(y, logits, model.sep_token_id, model.pad_token_id)
            
            all_losses.append(loss)
            all_accs.append(acc)
        
        # Average across batches
        loss = torch.stack(all_losses).mean(dim=0)
        acc = torch.stack(all_accs).mean(dim=0)
        
        return loss, acc

def calculate_loss_acc_vanilla(data, labels, model, loss_func, batch_size=None):
    if batch_size is None:
        pred = model(data)  # pred.shape = (# of examples, # model counts , output_dim)
    else:
        pred = []
        for i in range(0, len(data), batch_size):
            pred_cur = model(data[i:min(i+batch_size, len(data))])
            pred.append(pred_cur)
        pred = torch.cat(pred, dim=0)
    n, m, o = pred.shape
    loss = loss_func(pred.view(n * m, o), labels.repeat_interleave(m)).view(n, m).mean(dim=0)
    acc = (pred.view(n * m, o).argmax(dim=1) == labels.repeat_interleave(m)).view(n, m).float().mean(dim=0)
    return loss, acc


def calculate_transformer_metrics(data, labels, model, loss_func, batch_size=None):
    """
    Calculate loss and accuracy for transformer models on counting task.
    For counting task, 'data' contains the full sequences and 'labels' is ignored.
    """
    if batch_size is None:
        # For next-token prediction: input = seq[:-1], target = seq[1:]
        x = data[:, :-1]  # All tokens except last
        y = data[:, 1:]   # All tokens except first (shifted by 1)
        
        logits = model(x)  # (batch_size, model_count, seq_len, vocab_size)
        
        # Use model's loss function which handles padding
        losses = model.loss_function(y, logits)  # (model_count,)
        best_idx = losses.argmin()
        loss = losses[best_idx] 
        best_logits = logits[:,best_idx]
        best_logits = best_logits.unsqueeze(1)
        # Calculate token-wise accuracy on answer portion only
        acc = calculate_counting_accuracy(y, best_logits, model.sep_token_id, model.pad_token_id)
        em = calculate_exact_match_accuracy(y, best_logits, model.sep_token_id, model.pad_token_id)

        # WE RETURN THE WHOLE LOSS VECTOR OVER MODELS
        # WE SHOULD RETURN ONLY LOSS of best model and only logits of best model
        return loss, acc, em 
    else:
        # Batched processing
        all_losses = []
        all_accs = []
        
        for i in range(0, len(data), batch_size):
            batch_data = data[i:min(i+batch_size, len(data))]
            
            x = batch_data[:, :-1]
            y = batch_data[:, 1:]
            
            logits = model(x)
            loss = model.loss_function(y, logits)
            acc = calculate_counting_accuracy(y, logits, model.sep_token_id, model.pad_token_id)
            
            all_losses.append(loss)
            all_accs.append(acc)
        
        # Average across batches
        loss = torch.stack(all_losses).mean(dim=0)
        acc = torch.stack(all_accs).mean(dim=0)
        
        return loss, acc


def calculate_transformer_exactmatch(data, labels, model, loss_func, batch_size=None):

    """
    Calculate loss and accuracy for transformer models on counting task.
    For counting task, 'data' contains the full sequences and 'labels' is ignored.
    """
    if batch_size is None:
        # For next-token prediction: input = seq[:-1], target = seq[1:]
        x = data[:, :-1]  # All tokens except last
        y = data[:, 1:]   # All tokens except first (shifted by 1)
        
        logits = model(x)  # (batch_size, model_count, seq_len, vocab_size)
        
        # Calculate token-wise accuracy on answer portion only
        acc = calculate_exact_match_accuracy(y, logits, model.sep_token_id, model.pad_token_id)
        
        return acc
    else:
        # Batched processing
        all_accs = []
        
        for i in range(0, len(data), batch_size):
            batch_data = data[i:min(i+batch_size, len(data))]
            
            x = batch_data[:, :-1]
            y = batch_data[:, 1:]
            
            logits = model(x)
            acc = calculate_exact_match_accuracy(y, logits, model.sep_token_id, model.pad_token_id)
            
            all_accs.append(acc)
        
        # Average across batches
        acc = torch.stack(all_accs).mean(dim=0)
        
        return acc

def calculate_counting_accuracy(target, logits, sep_token_id, pad_token_id):
    """
    Calculate token-wise accuracy only on the counting part (after separator).
    """
    preds = logits.argmax(dim=-1)  # (batch_size, model_count, seq_len)
    batch_size, model_count, seq_len = preds.size()
    
    model_accs = []
    
    for model_idx in range(model_count):
        correct_tokens = 0
        total_tokens = 0
        
        for batch_idx in range(batch_size):
            # Find separator token position
            sep_positions = (target[batch_idx] == sep_token_id).nonzero(as_tuple=True)[0]
            
            if len(sep_positions) > 0:
                sep_pos = sep_positions[0].item()
                answer_start = sep_pos + 1
                
                # Check tokens after separator (ignoring pad tokens)
                answer_target = target[batch_idx, answer_start:]
                answer_pred = preds[batch_idx, model_idx, answer_start:]
                
                # Mask out padding tokens
                answer_mask = answer_target != pad_token_id
                valid_positions = answer_mask.sum().item()
                
                if valid_positions > 0:
                    correct = (answer_pred[answer_mask] == answer_target[answer_mask]).sum().item()
                    correct_tokens += correct
                    total_tokens += valid_positions
        
        if total_tokens > 0:
            acc = correct_tokens / total_tokens
        else:
            acc = 0.0
        
        model_accs.append(torch.tensor(acc))
    
    return torch.stack(model_accs)


def calculate_exact_match_accuracy(target, logits, sep_token_id, pad_token_id):
    """
    Calculate exact match accuracy - 1 if entire answer sequence is correct, 0 otherwise.
    Only computed on the part after the separator token.
    """
    preds = logits.argmax(dim=-1)  # (batch_size, model_count, seq_len)
    batch_size, model_count, seq_len = preds.size()
    
    model_exact_matches = []
    
    for model_idx in range(model_count):
        exact_matches = []
        
        for batch_idx in range(batch_size):
            # Find separator token position
            sep_positions = (target[batch_idx] == sep_token_id).nonzero(as_tuple=True)[0]
            
            if len(sep_positions) == 0:
                # No separator found - compare full sequence
                mask = target[batch_idx] != pad_token_id
                correct_per_token = (preds[batch_idx, model_idx] == target[batch_idx]) | ~mask
                exact_matches.append(correct_per_token.all().float())
            else:
                sep_pos = sep_positions[0].item()
                answer_start = sep_pos + 1
                
                # Check answer portion only
                answer_target = target[batch_idx, answer_start:]
                answer_pred = preds[batch_idx, model_idx, answer_start:]
                
                # Mask out padding tokens
                answer_mask = answer_target != pad_token_id
                
                if answer_mask.sum() == 0:
                    exact_matches.append(torch.tensor(1.0))
                else:
                    # All non-pad answer tokens must match
                    correct_answer_tokens = (answer_pred == answer_target) | ~answer_mask
                    exact_matches.append(correct_answer_tokens.all().float())
        
        model_exact_matches.append(torch.stack(exact_matches).mean())
    
    return torch.stack(model_exact_matches)


def make_permutation_invariant(m1, m2):
    # shape (1, model_count, out_d, in_d)
    sort_idx = m1[:, :, 0:1, :].sort(dim=3).indices
    new_m1 = torch.gather(m1, dim=3, index=sort_idx.repeat(1, 1, m1.shape[2], 1))
    new_m2 = torch.gather(m2, dim=2, index=sort_idx[:, :, 0, :, None])
    return new_m1, new_m2

def change_minimas_to_matrices(minimas, hidden_units):
    matrix1 = minimas[:, 0:hidden_units*2].reshape(1, -1, 2, hidden_units)
    matrix2 = minimas[:, hidden_units*2:hidden_units*3].reshape(1, -1, hidden_units, 1)
    bias2 = minimas[:, hidden_units*3:].reshape(1, -1, 1, 1)
    return matrix1, matrix2, bias2

def change_matrices_to_minimas(m1, m2, b2, hidden_units):
    minimas = torch.cat([
        m1.reshape(-1, 2*hidden_units),
        m2.reshape(-1, hidden_units*1),
        b2.reshape(-1, 1)], dim=1)
    return minimas

def visualize_decision_boundary(models_list, data=None, xlims=(-2,2), ylims=(-2, 2), filename='test.png'):
    model_count = models_list[0].model_count
    fig, axes = plt.subplots(nrows=model_count//3,
                             ncols=len(models_list)*3,
                             figsize=(len(models_list) * 3*3, (model_count) * 3//3))
    axes = np.reshape(axes, (model_count//3, len(models_list), 3))
    axes = np.transpose(axes, (0, 2, 1))
    axes = np.reshape(axes, (model_count, len(models_list)))
    axes = axes.T

    # X.reshape(3, 5, 3).permute(0, 2, 1).reshape(9, 5)
    # axes = axes.T
    for row_i, models in enumerate(models_list):
        models = models.cuda()
        xx, yy = np.meshgrid(np.arange(xlims[0], xlims[1], 0.01), np.arange(ylims[0], ylims[1], 0.01))
        grid_data = torch.cat([torch.tensor(xx.ravel())[:, None], torch.tensor(yy.ravel())[:, None]], dim=1).float().cuda()
        batch_size = 30
        predictions = []
        with torch.no_grad():
            for i in range(0, len(grid_data), batch_size):
                predictions.append(models(grid_data[i:min(i+batch_size, len(grid_data))]))
            predictions = torch.cat(predictions, dim=0)
        predictions = torch.softmax(predictions, dim=2).round()
        cm = plt.cm.hot
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])

        for model_i in range(model_count):
            grid_score = predictions[:, model_i, 1].cpu().detach().reshape(xx.shape)
            axes[row_i, model_i].contourf(xx, yy, grid_score, cmap=cm, alpha=0.8, vmin=0.4, vmax=0.6, rasterized=True)
            axes[row_i, model_i].set_axis_off()
            if data is not None:
                x, y = data

                axes[row_i, model_i].scatter(
                    x[y==0, 0].cpu(), x[y==0, 1].cpu(), c=np.zeros((len(x[y==0]), 1)), cmap=cm_bright,
                    edgecolors="k", rasterized=True, s=5
                )
                axes[row_i, model_i].scatter(
                    x[y==1, 0].cpu(), x[y==1, 1].cpu(), c=np.ones((len(x[y==1]), 1)), cmap=cm_bright,
                    edgecolors="k", marker='x', rasterized=True, s=5
                )
    # place some rectangular patches
            pad_x=0.004
            pad_y=0.02
            x0 = axes[row_i, 6].get_position().x0
            y0 = axes[row_i, 6].get_position().y0
            w = axes[row_i, 2].get_position().x1-x0
            h = axes[row_i, 2].get_position().y1-y0
            rect = plt.Rectangle(
                # (lower-left corner), width, height
                (x0-pad_x, y0-pad_y), w+pad_x*2, h+pad_y*2, fill=False, color="k", lw=6, 
                zorder=1000, transform=fig.transFigure, figure=fig
            )
            fig.patches.extend([rect])
    fig.savefig(filename, bbox_inches='tight', dpi=100)
    plt.close()

def calculate_sharpness_random_gaussian(m1, m2, b2, data, sigma=1, sample_count=100):
    # calculate stability with respect to gaussian noise

    # m1.shape (1, model_count, 2, hidden_units)
    # m2.shape (1, model_count, 2, hidden_units)
    model_count = m1.shape[1]
    x, y = data

    # add noise to the model

    train_accs = []
    for i in range(sample_count):
        m1_noise = torch.randn_like(m1) * sigma
        m2_noise = torch.randn_like(m2) * sigma
        b2_noise = torch.randn_like(b2) * sigma

        m1 += m1_noise
        m2 += m2_noise
        b2 += b2_noise
        # calculate prediction accuracy
        # repeat
        predictions = torch.sigmoid(torch.clamp(x @ m1, 0) @ m2 + b2)
        predictions = predictions > 0.5
        # predictions.shape (data count, model_count, 1, 1)
        train_acc = (y == predictions).float().mean(dim=0)[:, 0]
        train_accs.append(train_acc)
        # train_accs.shape (model_count, 1)
        m1 -= m1_noise
        m2 -= m2_noise
        b2 -= b2_noise
    train_accs = torch.cat(train_accs, dim=1)
    train_accs_mean = train_accs.mean(dim=1)
    return train_accs_mean

def calculate_norm(m1, m2, b2):
    m1_normsq = (m1 ** 2).sum(dim=(2, 3), keepdim=True)
    m2_normsq = (m2 ** 2).sum(dim=(2, 3), keepdim=True)
    b2_normsq = (b2 ** 2).sum(dim=(2, 3), keepdim=True)
    total_normsq = m1_normsq + m2_normsq + b2_normsq
    total_norm = total_normsq ** 0.5
    return total_norm

def calculate_sharpness_random_dir(m1, m2, b2, data, sample_count=10):
    # calculate stability with respect to gaussian noise

    # m1.shape (1, model_count, 2, hidden_units)
    # m2.shape (1, model_count, 2, hidden_units)
    model_count = m1.shape[1]
    x, y = data

    # add noise to the model

    train_accs = []
    biggest_rs = []
    for i in range(sample_count):
        biggest_r = torch.tensor(0.0, device=m1.device, dtype=torch.float)

        m1_noise_unit = torch.randn_like(m1)
        m2_noise_unit = torch.randn_like(m2)
        b2_noise_unit = torch.randn_like(b2)
        total_norm = calculate_norm(m1_noise_unit, m2_noise_unit, b2_noise_unit)
        m1_noise_unit = m1_noise_unit / total_norm
        m2_noise_unit = m2_noise_unit / total_norm
        b2_noise_unit = b2_noise_unit / total_norm

        for r in np.linspace(0, 3, 100):
            m1_noise = m1_noise_unit * r
            m2_noise = m2_noise_unit * r
            b2_noise = b2_noise_unit * r

            m1 += m1_noise
            m2 += m2_noise
            b2 += b2_noise
            # calculate prediction accuracy
            # repeat
            predictions = torch.sigmoid(torch.clamp(x @ m1, 0) @ m2 + b2)
            predictions = predictions > 0.5
            # predictions.shape (data count, model_count, 1, 1)
            train_acc = (y == predictions).float().mean(dim=0)[:, 0]
            # train_accs.shape (model_count, 1)
            biggest_r = torch.where(train_acc == 1,
                                    torch.tensor(r, device=train_acc.device, dtype=torch.float),
                                    biggest_r)


            m1 -= m1_noise
            m2 -= m2_noise
            b2 -= b2_noise
            if (train_acc==1).sum() == 0:
                break
        nan_tensor = torch.tensor(float('NaN'), device=biggest_r.device, dtype=torch.float)
        biggest_r = torch.where(biggest_r==3, nan_tensor, biggest_r)
        biggest_rs.append(biggest_r)
    biggest_rs = torch.cat(biggest_rs, dim=1)
    biggest_rs_mean = biggest_rs.nanmean(dim=1)
    nan_count = (biggest_rs == float('nan')).sum(dim=1)
    return biggest_rs_mean, nan_count

def calculate_sharpness_sam(m1, m2, b2, data, rho=1):
    # calculate stability with respect to gaussian noise

    # m1.shape (1, model_count, 2, hidden_units)
    # m2.shape (1, model_count, 2, hidden_units)
    model_count = m1.shape[1]
    x, y = data

    # add noise to the model

    train_accs = []
    m1.requires_grad = True
    m2.requires_grad = True
    b2.requires_grad = True
    # backward pass
    predictions = torch.sigmoid(torch.clamp(x @ m1, 0) @ m2 + b2)
    losses_ori = ((predictions - y)**2).mean(dim=0)
    loss_ori = losses_ori.sum()
    m1_grad, m2_grad, b2_grad = torch.autograd.grad(loss_ori, [m1, m2, b2])
    grad_norm = calculate_norm(m1_grad, m2_grad, b2_grad)
    m1_grad = m1_grad/grad_norm*rho
    m2_grad = m2_grad/grad_norm*rho
    b2_grad = b2_grad/grad_norm*rho

    with torch.no_grad():
        m1 += m1_grad
        m2 += m2_grad
        b2 += b2_grad

        # predictions.shape (data count, model_count, 1, 1)
        predictions = torch.sigmoid(torch.clamp(x @ m1, 0) @ m2 + b2)
        losses_attacked = ((predictions - y)**2).mean(dim=0)[:, 0, 0]
        train_acc_attacked = (y == (predictions>0.5)).float().mean(dim=0)[:, 0, 0]
        # train_accs.shape (model_count, 1)
        m1 -= m1_grad
        m2 -= m2_grad
        b2 -= b2_grad
    return train_acc_attacked, losses_attacked

def test_acc_by_bin(test_acc, bin_metric, bin_count=10):
    bin_metric = bin_metric.cpu()
    intervals = np.linspace(bin_metric.min(), bin_metric.max(), bin_count+1)
    for l, u in zip(intervals[:-1], intervals[1:]):
        idx = ((bin_metric >= l) & (bin_metric <= u))
        print(f"interval: {l.item(): 0.3f}, {u.item(): 0.3f}, count:{idx.sum().item()} "
              f"test accs: {test_acc[idx].mean().cpu().item(): 0.3f}")

class MLPModels(nn.Module):
    def __init__(self, input_dim, output_dim, layers, hidden_units, model_count, device):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = layers
        self.hidden_units = hidden_units
        self.device = device
        self.model_count = model_count
        self.weights = []
        for layer_i in range(layers+1):
            if layer_i == 0:
                self.weights.append(nn.Parameter(torch.rand((1, model_count, input_dim, hidden_units), device=device) * 2 - 1))
            elif layer_i == layers:
                self.weights.append(nn.Parameter(torch.rand((1, model_count, hidden_units, output_dim), device=device) * 2 - 1))
            else:
                self.weights.append(nn.Parameter(torch.rand((1, model_count, hidden_units, hidden_units), device=device) * 2 - 1))
        self.bias = nn.Parameter(torch.randn((1, model_count, output_dim), device=device) * 2 - 1)
        self.weights = torch.nn.ParameterList(self.weights)

    def reinitialize(self):
        for matrix in self.weights:
            torch.nn.init.uniform_(matrix.data, a=-1, b=1)
        torch.nn.init.uniform_(self.bias.data, a=-1, b=1)

    @torch.no_grad()
    def reset_parameters(self):
        import math
        for weight in self.weights:
            stdv = 1. / math.sqrt(weight.shape[3])
            weight.data.uniform_(-stdv, stdv)
        torch.nn.init.uniform_(self.bias.data, a=-1, b=1)

    def forward(self, x):
        # the forward takes in x shaped [# of examples, input_dim]
        # outputs [# of examples, model_count, logit_count]
        x = x[:, None, None]
        for matrix in self.weights[:-1]:
            x = x @ matrix
            x = torch.clamp(x, 0)
        x = x @ self.weights[-1]
        x = x.squeeze(2)
        x = x + self.bias
        return x

    def get_feature(self, x, cat_one=False):
        x = x[:, None, None]
        for matrix in self.weights[:-1]:
            x = x @ matrix
            x = torch.clamp(x, 0)
        x = x.squeeze(2)
        if cat_one:
            x = torch.cat((x, torch.ones(*x.shape[:2], 1, device=x.device)), dim=2)
        return x

    @torch.no_grad()
    def get_grad_norms(self):
        grad_square = 0
        for weight in self.weights:
            grad_square += (weight.grad**2).sum(dim=(0,2,3))
        grad_square += (self.bias.grad ** 2).sum(dim=(0, 2))
        grad_norm = grad_square ** 0.5
        return grad_norm

    def zero_grad(self):
        for para in self.parameters():
            para.grad = None

    @torch.no_grad()
    def get_weights_by_idx(self, idx):
        return {name: para[:, idx].cpu() for name, para in self.state_dict().items()}

    def get_model_subsets(self, idx):
        model_count = len(idx)
        new_model = MLPModels(
            input_dim=self.input_dim, output_dim=self.output_dim,
            layers=self.layers, hidden_units=self.hidden_units,
            model_count=model_count, device=self.device)
        new_model.load_state_dict(self.get_weights_by_idx(idx))
        return new_model

    @torch.no_grad()
    def normalize(self):
        cum_norm = 1
        for weight in self.weights:
            cur_norm = weight.norm(dim=(2,3), keepdim=True)
            weight.data /= cur_norm
            cum_norm *= cur_norm
        cum_norm = cum_norm.squeeze(3)
        self.bias.data /= cum_norm

    def forward_normalize(self, x):
        cum_norm = 1
        for weight in self.weights:
            cur_norm = weight.norm(dim=(2,3), keepdim=True)
            cum_norm *= cur_norm
        cum_norm = cum_norm.squeeze(3)
        return self.forward(x)/cum_norm

    @torch.no_grad()
    def make_permutation_invariant(self):
        weights = self.weights
        for i in range(len(weights)-1):
            sort_idx = weights[i][:, :, 0:1, :].sort(dim=3).indices
            weights[i].data.copy_(
                torch.gather(weights[i], dim=3, index=sort_idx.repeat(1, 1, weights[i].shape[2], 1))
            )
            weights[i+1].data.copy_(
                torch.gather(weights[i + 1], dim=2, index=sort_idx.permute(0, 1, 3, 2).repeat(1, 1, 1, weights[i+1].shape[3]))
            )

    @torch.no_grad()
    def shorten(self, count):
        idx = torch.arange(count)
        return self.get_model_subsets(idx)

    @torch.no_grad()
    def get_vectorized_weights(self):
        # return (# of models, # of parameters) as a tensor
        vectorized_weights = []
        for weight in chain(self.weights, [self.bias]):
            vectorized_weights.append(weight.data.reshape(self.model_count, -1).detach().cpu())
        vectorized_weights = torch.cat(vectorized_weights, dim=1)
        return vectorized_weights

class LeNetModels(nn.Module):
    def __init__(self, output_dim, width_factor, model_count, dataset, feature_dim=None):
        super(LeNetModels, self).__init__()
        self.model_count = model_count
        self.output_dim = output_dim
        self.width_factor = width_factor
        self.dataset = dataset
        if feature_dim is None:
            self.feature_dim = int(84 * width_factor)
        else:
            self.feature_dim = feature_dim
        if dataset == "cifar10":
            self.conv1 = nn.Conv2d(3*model_count,
                                    int(6*width_factor)*model_count,
                                    5, groups=model_count
                                   )

        elif dataset == "mnist":
            self.conv1 = nn.Conv2d(
                1*model_count,
                int(6*width_factor)*model_count,
                5, groups=model_count
            )
        self.conv2 = nn.Conv2d(int(6*width_factor)*model_count,
                               int(16*width_factor)*model_count,
                               5, groups=model_count)
        if dataset == "cifar10":
            self.fc1 = nn.Conv2d(int(16*width_factor)*5*5*model_count,
                                 int(120*width_factor)*model_count,
                                 1,
                                 groups=model_count)
        elif dataset == "mnist":
            self.fc1 = nn.Conv2d(int(16*width_factor)*4*4*model_count,
                                 int(120*width_factor)*model_count,
                                 1,
                                 groups=model_count)
        self.fc2 = nn.Conv2d(int(120*width_factor)*model_count,
                                 int(self.feature_dim*model_count),
                                 1,
                                 groups=model_count)
        self.fc3 = nn.Conv2d(int(self.feature_dim*model_count),
                                 output_dim*model_count,
                                 1,
                                 groups=model_count)
        self.basis_list = None
        self.curr_idx = 0
        self.radius= 1

    def forward(self, x):
        # the forward takes in x shaped [# of examples, input_dim, H, W]
        # outputs [# of examples, model_count, logit_count]
        x = x.repeat(1, self.model_count, 1, 1)
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.reshape(out.size(0), -1, 1, 1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = out.view(out.size(0), self.model_count, self.output_dim)
        return out

    @torch.no_grad()
    def pattern_search(self, x, y, loss_func):
        import random
        if self.basis_list is None:
            self.basis_list = []
            for para in self.parameters():
                para_flatten = para.data.view(self.model_count, -1)
                for p in range(para_flatten.shape[1]):
                    self.basis_list.append((para_flatten, p, "+"))
                    self.basis_list.append((para_flatten, p, "-"))
        random.shuffle(self.basis_list)
        self.curr_idx = 0

        while True:
            # replicate the first model and duplicate the weights across models
            for para in self.parameters():
                original_shape = para.shape
                para_reshaped = para.data.view(self.model_count, -1, *original_shape[2:])
                para_reshaped[1:] = para_reshaped[0:1]


            # modify each model at one index location
            for i in range(1,self.model_count):
                if self.curr_idx >= len(self.basis_list):
                    import pdb; pdb.set_trace()
                para, p_i, op = self.basis_list[self.curr_idx]
                if op == "+":
                    para[i, p_i] += self.radius
                else:
                    para[i, p_i] -= self.radius
                self.curr_idx += 1
                if self.curr_idx >= len(self.basis_list):
                    print("went over everything")
                    random.shuffle(self.basis_list)
                    self.radius /= 2
                    self.curr_idx = 0
                    break

            # forward and select the model with the best losses, and it into index 0
            pred = self.forward_normalize(x)
            n, m, o = pred.shape
            loss = loss_func(pred.view(n * m, o), y.repeat_interleave(m)).view(n, m).mean(dim=0)

            best_idx = loss.min(dim=0).indices

            for para in self.parameters():
                original_shape = para.shape
                para_reshaped = para.data.view(self.model_count, -1, *original_shape[2:])
                para_reshaped[:] = para_reshaped[best_idx:best_idx+1]
            if best_idx != 0:
                break

    @torch.no_grad()
    def greedy_random(self, x, y, loss_func):
        for _ in range(30):
            iter_max = 100
            for i in range(iter_max):
                # add noise to the all models beside the zero indexed model
                for para in self.parameters():
                    original_shape = para.shape
                    para_reshaped = para.data.view(self.model_count, -1, *original_shape[2:])
                    para_reshaped[1:] = para_reshaped[0:1]
                    para_reshaped[1:] += torch.randn_like(para_reshaped[1:])*self.radius

                # forward and select the model with the best losses, and it into index 0
                pred = self.forward_normalize(x)
                n, m, o = pred.shape
                loss = loss_func(pred.view(n * m, o), y.repeat_interleave(m)).view(n, m).mean(dim=0)

                best_idx = loss.min(dim=0).indices

                for para in self.parameters():
                    original_shape = para.shape
                    para_reshaped = para.data.view(self.model_count, -1, *original_shape[2:])
                    para_reshaped[:] = para_reshaped[best_idx:best_idx + 1]
                if best_idx != 0:
                    return
            print(f"radius decreased to {self.radius/2}")
            self.radius /= 2




    def get_feature(self, x, cat_one=False):
        x = x.repeat(1, self.model_count, 1, 1)
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.reshape(out.size(0), -1, 1, 1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = out.view(out.size(0), self.model_count, self.feature_dim)

        if cat_one:
            out = torch.cat((out, torch.ones(*out.shape[:2], 1, device=x.device)), dim=2)
        return out

    @torch.no_grad()
    def get_model_subsets(self, idx):
        model_count = len(idx)
        new_model = LeNetModels(output_dim=self.output_dim,
                                width_factor=self.width_factor,
                                model_count=model_count,
                                feature_dim=self.feature_dim,
                                dataset=self.dataset)
        new_model.load_state_dict(self.get_weights_by_idx(idx))
        return new_model

    @torch.no_grad()
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    @torch.no_grad()
    def reinitialize(self, mult=1):
        for para in self.parameters():
            torch.nn.init.uniform_(para.data, a=-mult, b=mult)


    @torch.no_grad()
    def reinitialize_sphere(self, mult=1):
        overall_norm_square = 0
        for para in self.parameters():
            torch.nn.init.normal_(para.data)
            original_shape = para.shape
            para_reshaped = para.data.view(self.model_count, -1, *original_shape[2:])
            sum_dim = tuple((d for d in range(1, len(para_reshaped.shape))))
            overall_norm_square += (para_reshaped ** 2).sum(dim=sum_dim)
        overall_norm = overall_norm_square ** 0.5
        for para in self.parameters():
            original_shape = para.shape
            para_reshaped = para.data.view(self.model_count, -1, *original_shape[2:])
            new_norm_shape = (-1, ) + tuple((1 for i in range(len(para_reshaped.shape)-1)))
            para_reshaped /= (overall_norm.view(new_norm_shape)/mult)
        
    @torch.no_grad()
    def get_weights_by_idx(self, idx):
        weight_dict = {}
        for name, para in self.state_dict().items():
            original_shape = para.shape
            para_reshaped = para.reshape(self.model_count, -1, *original_shape[2:])
            para_selected = para_reshaped[idx]
            para_selected = para_selected.reshape(-1, *original_shape[1:])
            weight_dict[name] = para_selected.clone().detach().cpu()
        return weight_dict

    @torch.no_grad()
    def shorten(self, count):
        idx = torch.arange(count)
        return self.get_model_subsets(idx)

    @torch.no_grad()
    def normalize(self):
        cum_norm = 1
        for layer in [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]:
            cur_weight = layer.weight
            original_shape = cur_weight.shape
            cur_weight = cur_weight.view(self.model_count, -1, *original_shape[2:])
            cur_norm = cur_weight.norm(p=2, dim=(1, 2, 3), keepdim=True) / 3
            cur_weight /= cur_norm
            cum_norm *= cur_norm.view(self.model_count, -1)
            biasview = layer.bias.data.view(self.model_count, -1)
            biasview /= cum_norm

    @torch.no_grad()
    def forward_normalize(self, x):
        x = self.forward(x)
        cum_norm = 1
        for layer in [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]:
            cur_weight = layer.weight
            original_shape = cur_weight.shape
            cur_weight = cur_weight.view(self.model_count, -1, *original_shape[2:])
            cur_norm = cur_weight.norm(p=2, dim=(1, 2, 3), keepdim=True) /3
            cum_norm *= cur_norm.view(self.model_count, -1)
        x /= cum_norm
        return x

class LinearModels(nn.Module):
    def __init__(self, input_dim, output_dim, model_count, device):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.model_count = model_count
        self.weight = nn.Parameter(torch.rand((1, model_count, input_dim, output_dim), device=device) * 2 - 1)
        self.bias = nn.Parameter(torch.randn((1, model_count, output_dim), device=device) * 2 - 1)

    def reinitialize(self):
        torch.nn.init.uniform_(self.weight.data, a=-1, b=1)
        torch.nn.init.uniform_(self.bias.data, a=-1, b=1)

    @torch.no_grad()
    def reset_parameters(self):
        import math
        stdv = 1. / math.sqrt(self.weight.shape[3])
        self.weight.data.uniform_(-stdv, stdv)
        torch.nn.init.uniform_(self.bias.data, a=-1, b=1)

    def forward(self, x):
        # the forward takes in x shaped [# of examples, input_dim]
        # outputs [# of examples, model_count, logit_count]
        x = x.view(x.shape[0], -1)
        x = x[:, None, None]
        x = x @ self.weight
        x = x.squeeze(2)
        x = x + self.bias
        return x

    def forward_normalize(self, x):
        cur_norm = self.weight.norm(dim=(2,3), keepdim=True).squeeze(3)
        return self.forward(x)/cur_norm

    @torch.no_grad()
    def normalize(self):
        weight_norm = self.weight.norm(dim=(2,3), keepdim=True)
        self.weight.data /= weight_norm
        self.bias.data /= weight_norm.squeeze(3)

    @torch.no_grad()
    def get_grad_norms(self):
        grad_square = 0
        grad_square += (self.weight.grad**2).sum(dim=(0,2,3))
        grad_square += (self.bias.grad ** 2).sum(dim=(0, 2))
        grad_norm = grad_square ** 0.5
        return grad_norm

    def zero_grad(self):
        for para in self.parameters():
            para.grad = None

    @torch.no_grad()
    def get_weights_by_idx(self, idx):
        return {name: para[:, idx].cpu() for name, para in self.state_dict().items()}

    def get_model_subsets(self, idx):
        model_count = len(idx)
        new_model = LinearModels(
            input_dim=self.input_dim, output_dim=self.output_dim,
            model_count=model_count, device=self.device)
        new_model.load_state_dict(self.get_weights_by_idx(idx))
        return new_model


class TransformerModels(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len, model_count, device, dropout=0.1, sep_token_id=102, pad_token_id=103, init='regular'):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_len = max_len
        self.model_count = model_count
        self.device = device
        self.dropout = dropout
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        
        # Vectorized parameters following LeNet pattern
        # Token embeddings: (model_count, vocab_size, d_model)
        self.token_emb_weight = nn.Parameter(torch.randn(model_count, vocab_size, d_model))
        
        # Position embeddings: (model_count, max_len, d_model)  
        self.pos_emb = nn.Parameter(torch.zeros(model_count, max_len, d_model))
        
        # Transformer layer parameters
        self.transformer_params = nn.ParameterDict()
        
        for layer in range(n_layers):
            # Layer norm 1
            self.transformer_params[f'ln1_{layer}_weight'] = nn.Parameter(torch.ones(model_count, d_model))
            self.transformer_params[f'ln1_{layer}_bias'] = nn.Parameter(torch.zeros(model_count, d_model))
            
            # Attention QKV projection: (model_count, d_model, 3*d_model)
            self.transformer_params[f'attn_{layer}_qkv_weight'] = nn.Parameter(torch.randn(model_count, d_model, 3*d_model))
            self.transformer_params[f'attn_{layer}_qkv_bias'] = nn.Parameter(torch.zeros(model_count, 3*d_model))
            
            # Attention output projection: (model_count, d_model, d_model)
            self.transformer_params[f'attn_{layer}_out_weight'] = nn.Parameter(torch.randn(model_count, d_model, d_model))
            self.transformer_params[f'attn_{layer}_out_bias'] = nn.Parameter(torch.zeros(model_count, d_model))
            
            # Layer norm 2
            self.transformer_params[f'ln2_{layer}_weight'] = nn.Parameter(torch.ones(model_count, d_model))
            self.transformer_params[f'ln2_{layer}_bias'] = nn.Parameter(torch.zeros(model_count, d_model))
            
            # Feed forward 1: (model_count, d_model, d_ff)
            self.transformer_params[f'ff1_{layer}_weight'] = nn.Parameter(torch.randn(model_count, d_model, d_ff))
            self.transformer_params[f'ff1_{layer}_bias'] = nn.Parameter(torch.zeros(model_count, d_ff))
            
            # Feed forward 2: (model_count, d_ff, d_model)
            self.transformer_params[f'ff2_{layer}_weight'] = nn.Parameter(torch.randn(model_count, d_ff, d_model))
            self.transformer_params[f'ff2_{layer}_bias'] = nn.Parameter(torch.zeros(model_count, d_model))
        
        # Final layer norm
        self.ln_f_weight = nn.Parameter(torch.ones(model_count, d_model))
        self.ln_f_bias = nn.Parameter(torch.zeros(model_count, d_model))
        
        # Output projection: (model_count, d_model, vocab_size)
        self.head_weight = nn.Parameter(torch.randn(model_count, d_model, vocab_size))
        self.head_bias = nn.Parameter(torch.zeros(model_count, vocab_size))
        
        # Pattern search state
        self.basis_list = None
        self.curr_idx = 0
        self.radius = 1 
        
        # Initialize parameters
        if init == 'regular':
            self._init_vectorized_params()
        elif init == 'uniform':
            self.reinitialize()
        else:
            # cast an error or smth
            raise ValueError(f"Unknown initialization method: {init}")

    def _init_vectorized_params(self):
        """Initialize parameters using Xavier/Kaiming initialization."""
        with torch.no_grad():
            # Initialize token embeddings
            nn.init.normal_(self.token_emb_weight, mean=0.0, std=0.02)
            
            # Initialize position embeddings 
            nn.init.zeros_(self.pos_emb)
            
            # Initialize transformer layer parameters
            for layer in range(self.n_layers):
                # Layer norm weights to 1, biases to 0 (already done)
                
                # QKV projection - Xavier uniform
                nn.init.xavier_uniform_(self.transformer_params[f'attn_{layer}_qkv_weight'])
                nn.init.zeros_(self.transformer_params[f'attn_{layer}_qkv_bias'])
                
                # Attention output projection
                nn.init.xavier_uniform_(self.transformer_params[f'attn_{layer}_out_weight'])  
                nn.init.zeros_(self.transformer_params[f'attn_{layer}_out_bias'])
                
                # Feed forward layers
                nn.init.xavier_uniform_(self.transformer_params[f'ff1_{layer}_weight'])
                nn.init.zeros_(self.transformer_params[f'ff1_{layer}_bias'])
                nn.init.xavier_uniform_(self.transformer_params[f'ff2_{layer}_weight'])
                nn.init.zeros_(self.transformer_params[f'ff2_{layer}_bias'])
            
            # Final layer norm (already initialized to 1s and 0s)
            # Output head
            nn.init.xavier_uniform_(self.head_weight)
            nn.init.zeros_(self.head_bias)

    def vectorized_token_embedding(self, x):
        """Vectorized token embedding lookup across all models.
        
        Args:
            x: (batch_size, seq_len) - input token ids
            
        Returns:
            (batch_size, model_count, seq_len, d_model) - embeddings for all models
        """
        batch_size, seq_len = x.size()
        
        # Ensure x has the correct integer dtype
        x = x.long()
        
        # self.token_emb_weight: (model_count, vocab_size, d_model)
        # We need to gather embeddings for each model separately
        embeddings = torch.zeros(batch_size, self.model_count, seq_len, self.d_model, 
                                 device=x.device, dtype=self.token_emb_weight.dtype)
        
        for model_idx in range(self.model_count):
            # Get embeddings for this model: (batch_size, seq_len, d_model)
            model_embeddings = F.embedding(x, self.token_emb_weight[model_idx])
            embeddings[:, model_idx] = model_embeddings
            
        return embeddings

    def vectorized_position_embedding(self, position_ids):
        """Vectorized position embedding lookup.
        
        Args:
            position_ids: (batch_size, seq_len) - position indices
            
        Returns:
            (batch_size, model_count, seq_len, d_model) - position embeddings
        """
        batch_size, seq_len = position_ids.size()
        
        # Ensure position_ids has the correct integer dtype
        position_ids = position_ids.long()
        
        # Similar to token embeddings
        pos_embeddings = torch.zeros(batch_size, self.model_count, seq_len, self.d_model,
                                     device=position_ids.device, dtype=self.pos_emb.dtype)
        
        for model_idx in range(self.model_count):
            # Gather position embeddings for this model
            model_pos_emb = torch.gather(
                self.pos_emb[model_idx].unsqueeze(0).expand(batch_size, -1, -1),
                1,
                position_ids.unsqueeze(-1).expand(-1, -1, self.d_model)
            )
            pos_embeddings[:, model_idx] = model_pos_emb
            
        return pos_embeddings

    def vectorized_layer_norm(self, x, weight, bias):
        """Vectorized layer normalization.
        
        Args:
            x: (batch_size, model_count, seq_len, d_model)
            weight: (model_count, d_model) 
            bias: (model_count, d_model)
            
        Returns:
            (batch_size, model_count, seq_len, d_model) - normalized
        """
        # Layer norm along the last dimension (d_model)
        mean = x.mean(dim=-1, keepdim=True)  # (batch_size, model_count, seq_len, 1)
        var = x.var(dim=-1, unbiased=False, keepdim=True)  # (batch_size, model_count, seq_len, 1)
        
        normalized = (x - mean) / torch.sqrt(var + 1e-5)
        
        # Apply per-model weight and bias
        # weight/bias: (model_count, d_model) -> (1, model_count, 1, d_model)
        weight = weight.unsqueeze(0).unsqueeze(2)
        bias = bias.unsqueeze(0).unsqueeze(2) 
        
        return normalized * weight + bias

    def vectorized_attention(self, x, layer_idx):
        """Vectorized multi-head self-attention with causal masking.
        
        Args:
            x: (batch_size, model_count, seq_len, d_model)
            layer_idx: which transformer layer
            
        Returns:
            (batch_size, model_count, seq_len, d_model) - attention output
        """
        batch_size, model_count, seq_len, d_model = x.size()
        
        # Get QKV weights and biases for this layer
        qkv_weight = self.transformer_params[f'attn_{layer_idx}_qkv_weight']  # (model_count, d_model, 3*d_model)
        qkv_bias = self.transformer_params[f'attn_{layer_idx}_qkv_bias']      # (model_count, 3*d_model)
        out_weight = self.transformer_params[f'attn_{layer_idx}_out_weight']  # (model_count, d_model, d_model)
        out_bias = self.transformer_params[f'attn_{layer_idx}_out_bias']      # (model_count, d_model)
        
        # Vectorized QKV projection
        # x: (batch_size, model_count, seq_len, d_model)
        # qkv_weight: (model_count, d_model, 3*d_model)
        qkv = torch.einsum('bmsd,mde->bmse', x, qkv_weight) + qkv_bias.unsqueeze(0).unsqueeze(2)
        # qkv: (batch_size, model_count, seq_len, 3*d_model)
        
        # Reshape and split into Q, K, V
        qkv = qkv.reshape(batch_size, model_count, seq_len, self.n_heads, 3 * self.d_model // self.n_heads)
        qkv = qkv.transpose(2, 3)  # (batch_size, model_count, n_heads, seq_len, 3*head_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (batch_size, model_count, n_heads, seq_len, head_dim)
        
        # Compute attention scores
        head_dim = self.d_model // self.n_heads
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        # attn_scores: (batch_size, model_count, n_heads, seq_len, seq_len)
        
        # Apply causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Apply attention to values
        attn_out = torch.matmul(attn_weights, v)  # (batch_size, model_count, n_heads, seq_len, head_dim)
        
        # Concatenate heads
        attn_out = attn_out.transpose(2, 3).contiguous().reshape(batch_size, model_count, seq_len, d_model)
        
        # Output projection
        output = torch.einsum('bmsd,mde->bmse', attn_out, out_weight) + out_bias.unsqueeze(0).unsqueeze(2)
        
        return output

    def vectorized_feed_forward(self, x, layer_idx):
        """Vectorized feed forward network.
        
        Args:
            x: (batch_size, model_count, seq_len, d_model)
            layer_idx: which transformer layer
            
        Returns:
            (batch_size, model_count, seq_len, d_model)
        """
        # Get FF weights and biases
        ff1_weight = self.transformer_params[f'ff1_{layer_idx}_weight']  # (model_count, d_model, d_ff)
        ff1_bias = self.transformer_params[f'ff1_{layer_idx}_bias']      # (model_count, d_ff)
        ff2_weight = self.transformer_params[f'ff2_{layer_idx}_weight']  # (model_count, d_ff, d_model)
        ff2_bias = self.transformer_params[f'ff2_{layer_idx}_bias']      # (model_count, d_model)
        
        # First linear layer + ReLU
        hidden = torch.einsum('bmsd,mde->bmse', x, ff1_weight) + ff1_bias.unsqueeze(0).unsqueeze(2)
        hidden = F.relu(hidden)
        
        # Dropout
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        
        # Second linear layer
        output = torch.einsum('bmsd,mde->bmse', hidden, ff2_weight) + ff2_bias.unsqueeze(0).unsqueeze(2)
        
        return output

    def vectorized_transformer_layer(self, x, layer_idx):
        """Complete transformer layer with residual connections.
        
        Args:
            x: (batch_size, model_count, seq_len, d_model)
            layer_idx: which transformer layer
            
        Returns:
            (batch_size, model_count, seq_len, d_model)
        """
        # Pre-norm attention
        ln1_weight = self.transformer_params[f'ln1_{layer_idx}_weight']
        ln1_bias = self.transformer_params[f'ln1_{layer_idx}_bias']
        normed_x = self.vectorized_layer_norm(x, ln1_weight, ln1_bias)
        
        # Self-attention with residual connection
        attn_out = self.vectorized_attention(normed_x, layer_idx)
        x = x + attn_out
        
        # Pre-norm feed forward
        ln2_weight = self.transformer_params[f'ln2_{layer_idx}_weight'] 
        ln2_bias = self.transformer_params[f'ln2_{layer_idx}_bias']
        normed_x = self.vectorized_layer_norm(x, ln2_weight, ln2_bias)
        
        # Feed forward with residual connection
        ff_out = self.vectorized_feed_forward(normed_x, layer_idx)
        x = x + ff_out
        
        return x

    def vectorized_output_projection(self, x):
        """Final output projection to vocabulary.
        
        Args:
            x: (batch_size, model_count, seq_len, d_model)
            
        Returns:
            (batch_size, model_count, seq_len, vocab_size)
        """
        # self.head_weight: (model_count, d_model, vocab_size)
        # self.head_bias: (model_count, vocab_size)
        logits = torch.einsum('bmsd,mdv->bmsv', x, self.head_weight) + self.head_bias.unsqueeze(0).unsqueeze(2)
        return logits

    def forward(self, x, position_ids=None, position_offset=None):
        # x shape: (batch_size, seq_len)  
        # Output shape: (batch_size, model_count, seq_len, vocab_size)
        batch_size, seq_len = x.size()
        
        # Token embeddings - vectorized across models
        token_emb = self.vectorized_token_embedding(x)  # (batch_size, model_count, seq_len, d_model)
        
        # Position embeddings with optional offset
        # Original default behavior: use positions [0, 1, 2, ..., seq_len-1]
        pos_emb = self.pos_emb[:, :seq_len].unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Add embeddings
        hidden = token_emb + pos_emb  # (batch_size, model_count, seq_len, d_model)
        
        # Pass through transformer layers
        for layer_idx in range(self.n_layers):
            hidden = self.vectorized_transformer_layer(hidden, layer_idx)
            
        # Final layer norm
        hidden = self.vectorized_layer_norm(hidden, self.ln_f_weight, self.ln_f_bias)
        
        # Output projection
        logits = self.vectorized_output_projection(hidden)  # (batch_size, model_count, seq_len, vocab_size)
        
        return logits

    def loss_function(self, target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Cross entropy loss ignoring pad tokens - fixed per-model calculation."""
        # Handle different input dimensions
        if logits.dim() == 4:  # (B, M, S, V)
            batch_size, model_count, seq_len, vocab_size = logits.size()
        elif logits.dim() == 3:  # (B, S, V) - single model case  
            batch_size, seq_len, vocab_size = logits.size()
            model_count = 1
            logits = logits.unsqueeze(1)  # (B, 1, S, V)
        else:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")
        
        # Ensure target is (B, S)
        if target.dim() == 1:
            # Infer batch_size and seq_len from logits shape
            if logits.dim() == 4:
                expected_batch_size, _, expected_seq_len, _ = logits.size()
                expected_total = expected_batch_size * expected_seq_len
                if target.size(0) == expected_total:
                    target = target.reshape(expected_batch_size, expected_seq_len)
                else:
                    raise ValueError(f"Target size {target.size(0)} doesn't match expected {expected_total}")
            else:
                raise ValueError(f"Cannot infer target shape from logits shape {logits.shape}")
        
        # Compute loss per model separately (avoids partitioning bug)
        losses = []
        for model_idx in range(model_count):
            # Get logits and target for this model
            model_logits = logits[:, model_idx, :, :]  # (B, S, V)
            model_target = target  # (B, S) - same for all models
            
            # Create mask for valid (non-padded) tokens
            mask = model_target != self.pad_token_id  # (B, S)
            
            if mask.sum() == 0:
                # No valid tokens
                losses.append(torch.tensor(0.0, device=logits.device))
                continue
                
            # Flatten and filter
            flat_logits = model_logits.reshape(-1, vocab_size)  # (B*S, V)
            flat_target = model_target.reshape(-1)  # (B*S,)
            flat_mask = mask.reshape(-1)  # (B*S,)
            
            # Apply mask
            filtered_logits = flat_logits[flat_mask]  # (valid_tokens, V)
            filtered_target = flat_target[flat_mask]  # (valid_tokens,)
            
            # Compute cross entropy for this model
            model_loss = torch.nn.functional.cross_entropy(
                filtered_logits, filtered_target, reduction='mean'
            )
            losses.append(model_loss)
        
        return torch.stack(losses)  # (model_count,)

    def calculate_exact_match(self, target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Calculate exact match - 1 if answer portion after separator is correct, 0 otherwise."""
        # Get predicted tokens - always argmax on last dimension
        preds = logits.argmax(dim=-1)  # (batch_size, model_count, seq_len)
        
        batch_size = target.size(0)
        model_exact_matches = []
        
        for model_idx in range(self.model_count):
            exact_matches = []
            model_preds = preds[:, model_idx, :]  # (batch_size, seq_len)
            
            for batch_idx in range(batch_size):
                # Find separator token position in target sequence
                sep_positions = (target[batch_idx] == self.sep_token_id).nonzero(as_tuple=True)[0]
                
                if len(sep_positions) == 0:
                    # No separator found, fall back to full sequence match
                    mask = target[batch_idx] != self.pad_token_id
                    correct_per_token = (model_preds[batch_idx] == target[batch_idx]) | ~mask
                    exact_matches.append(correct_per_token.all().float())
                else:
                    # Get position after separator - this is where the answer starts
                    sep_pos = sep_positions[0].item()
                    answer_start = sep_pos + 1
                    
                    # Only check tokens after separator (ignoring pad tokens)
                    answer_target = target[batch_idx, answer_start:]
                    answer_pred = model_preds[batch_idx, answer_start:]
                    
                    # Create mask for non-pad tokens in answer portion
                    answer_mask = answer_target != self.pad_token_id
                    
                    if answer_mask.sum() == 0:
                        # No answer tokens to check
                        exact_matches.append(torch.tensor(1.0))
                    else:
                        # Check if all non-pad answer tokens match
                        correct_answer_tokens = (answer_pred == answer_target) | ~answer_mask
                        exact_matches.append(correct_answer_tokens.all().float())
            
            model_exact_matches.append(torch.stack(exact_matches).mean())
        
        return torch.stack(model_exact_matches)

    def compute_loss(self, batch):
        """Compute loss for a batch - used by both training and validation."""
        # Handle both old tensor format and new dict format with position_ids
        if isinstance(batch, dict):
            sequences = batch['input_ids']
            position_ids = batch.get('position_ids', None)
        else:
            # Legacy tensor format
            sequences = batch
            position_ids = None
            
        # For next-token prediction: input = seq[:-1], target = seq[1:]
        x = sequences[:, :-1]  # All tokens except last
        y = sequences[:, 1:]   # All tokens except first (shifted by 1)
        
        # Adjust position_ids for shifted input if present
        pos_ids = position_ids[:, :-1] if position_ids is not None else None
        
        logits = self(x, position_ids=pos_ids)
        return self.loss_function(y, logits)

    @torch.no_grad()
    def pattern_search(self, data, dummy_labels, loss_func, position_offset=None):
        """Pattern search following LeNet approach - adapted for TransformerModels."""
        import random
        
        # Ensure model is in eval mode for consistent loss computation
        #self.eval()
        
        # Create x and y for next-token prediction
        x = data[:, :-1]
        y = data[:, 1:]
        
        # Initialize basis list if not already done (LeNet approach #1)
        if self.basis_list is None:
            self.basis_list = []
            for param in self.parameters():
                if param.dim() >= 2 and param.size(0) == self.model_count:
                    # Flatten parameter keeping model_count dimension (like LeNet)
                    param_flatten = param.data.view(self.model_count, -1)
                    for p in range(param_flatten.shape[1]):
                        self.basis_list.append((param_flatten, p, "+"))
                        self.basis_list.append((param_flatten, p, "-"))
            random.shuffle(self.basis_list)
            print(f"Created basis list with {len(self.basis_list)} elements")
        
        random.shuffle(self.basis_list)
        
        # CRITICAL FIX: Reset curr_idx at the beginning of each pattern_search call
        # This ensures each call starts from the beginning of the basis_list,
        # preventing state corruption between consecutive pattern_search calls
        self.curr_idx = 0
        
        # LeNet approach #2: No best_loss initialization - use immediate comparison
        # LeNet approach #3: Simple while loop
        while True:
            #logits = self(x, position_ids=None, position_offset=position_offset)  # (batch_size, model_count, seq_len, vocab_size)
            #losses = self.loss_function(y, logits)  # (model_count,)
            #print("Loss of model 0 at beginning of loop:", losses[0])
            for param in self.parameters():
                if param.dim() >= 2 and param.size(0) == self.model_count:
                    param_reshaped = param.data.view(self.model_count, -1)
                    param_reshaped[1:] = param_reshaped[0:1].clone()
            
            # LeNet approach #4: Apply perturbations to each model sequentially
            for i in range(1, self.model_count):
                if self.curr_idx >= len(self.basis_list):
                    print("went over everything")
                    random.shuffle(self.basis_list)
                    self.radius /= 2
                    self.curr_idx = 0
                    break
                
                param_flatten, p_i, op = self.basis_list[self.curr_idx]
                if op == "+":
                    param_flatten[i, p_i] += self.radius
                else:
                    param_flatten[i, p_i] -= self.radius
                self.curr_idx += 1
            
            # Forward pass and evaluate all models with position offset
            logits = self(x, position_ids=None, position_offset=position_offset)  # (batch_size, model_count, seq_len, vocab_size)
            losses = self.loss_function(y, logits)  # (model_count,)
            
            # Find best model and compare with original (model 0)
            best_idx = losses.argmin()
            original_loss = losses[0]  # Loss of model 0 (original)
            best_loss = losses[best_idx]
            
            print(f"Model 0 loss: {original_loss:.4f}), best loss: {best_loss:.4f} (model {best_idx})")
            
            # Only update if we found actual improvement
            if best_idx != 0 and best_loss < original_loss:
                print(f"Pattern search: improvement found! Copying model {best_idx} to model 0")
                # Copy ONLY the best model to model 0, not to all models
                for param in self.parameters():
                    if param.dim() >= 2 and param.size(0) == self.model_count:
                        param_reshaped = param.data.view(self.model_count, -1)
                        param_reshaped[0] = param_reshaped[best_idx].clone() # Copy only to model 0
                break  # Found improvement, exit
            elif best_idx == 0:
                print("Model 0 is already best, continuing search...")
                # Continue searching - model 0 is already optimal among current perturbations
            else:
                print("No improvement found in this iteration, continuing search...")
                # Continue searching - no perturbation improved upon model 0
    
    def _copy_model_0_to_all(self):
        """Copy model 0 parameters to all other models (LeNet pattern)."""
        for param in self.parameters():
            if param.dim() >= 2 and param.size(0) == self.model_count:
                # This is a vectorized parameter with model_count as first dimension
                with torch.no_grad():
                    # Copy model 0 to all other models
                    param.data[1:] = param.data[0:1].expand_as(param.data[1:])
    
    def _copy_model_to_model_0(self, source_model_idx):
        """Copy parameters from source_model_idx to model 0."""
        for param in self.parameters():
            if param.dim() >= 2 and param.size(0) == self.model_count:
                # This is a vectorized parameter with model_count as first dimension
                with torch.no_grad():
                    param.data[0] = param.data[source_model_idx].clone()

    @torch.no_grad()
    def reset_parameters(self):
        """Reset all model parameters using proper initialization."""
        self._init_vectorized_params()

    @torch.no_grad()
    def reinitialize(self, mult=1):
        """Reinitialize all model parameters with uniform distribution."""
        with torch.no_grad():
            for param in self.parameters():
                nn.init.uniform_(param, -mult, mult)

    def forward_normalize(self, x):
        """Forward pass with normalization - for compatibility with existing code."""
        return self.forward(x)

    @torch.no_grad()
    def get_weights_by_idx(self, idx):
        """Get weights for specific model indices from vectorized parameters."""
        if isinstance(idx, torch.Tensor):
            if idx.dtype == torch.bool:
                # Convert boolean tensor to integer indices
                idx = idx.nonzero(as_tuple=True)[0].cpu().numpy()
            else:
                idx = idx.cpu().numpy()
        elif isinstance(idx, int):
            idx = [idx]
        
        weights_list = []
        for model_idx in idx:
            model_weights = {}
            
            # Extract parameters for this specific model
            model_weights['token_emb_weight'] = self.token_emb_weight[model_idx].clone().cpu()
            model_weights['pos_emb'] = self.pos_emb[model_idx].clone().cpu()
            model_weights['ln_f_weight'] = self.ln_f_weight[model_idx].clone().cpu()
            model_weights['ln_f_bias'] = self.ln_f_bias[model_idx].clone().cpu()
            model_weights['head_weight'] = self.head_weight[model_idx].clone().cpu()
            model_weights['head_bias'] = self.head_bias[model_idx].clone().cpu()
            
            # Extract transformer layer parameters
            for layer_idx in range(self.n_layers):
                for param_key in ['ln1_weight', 'ln1_bias', 'attn_qkv_weight', 'attn_qkv_bias',
                                  'attn_out_weight', 'attn_out_bias', 'ln2_weight', 'ln2_bias',
                                  'ff1_weight', 'ff1_bias', 'ff2_weight', 'ff2_bias']:
                    full_key = f'{param_key}_{layer_idx}' if layer_idx > 0 else param_key
                    if full_key in self.transformer_params:
                        model_weights[full_key] = self.transformer_params[full_key][model_idx].clone().cpu()
            
            weights_list.append(model_weights)
        
        return weights_list

    def get_model_subsets(self, idx):
        """Create a new TransformerModels with subset of models."""
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        elif isinstance(idx, int):
            idx = [idx]
            
        new_model_count = len(idx)
        new_model = TransformerModels(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            max_len=self.max_len,
            model_count=new_model_count,
            device=self.device,
            dropout=self.dropout,
            sep_token_id=self.sep_token_id,
            pad_token_id=self.pad_token_id
        )
        
        # Copy the selected model parameters
        with torch.no_grad():
            for new_idx, old_idx in enumerate(idx):
                # Copy basic parameters
                new_model.token_emb_weight.data[new_idx] = self.token_emb_weight.data[old_idx]
                new_model.pos_emb.data[new_idx] = self.pos_emb.data[old_idx]
                new_model.ln_f_weight.data[new_idx] = self.ln_f_weight.data[old_idx]
                new_model.ln_f_bias.data[new_idx] = self.ln_f_bias.data[old_idx]
                new_model.head_weight.data[new_idx] = self.head_weight.data[old_idx]
                new_model.head_bias.data[new_idx] = self.head_bias.data[old_idx]
                
                # Copy transformer layer parameters
                for param_name in self.transformer_params:
                    new_model.transformer_params[param_name].data[new_idx] = self.transformer_params[param_name].data[old_idx]
        
        return new_model
if __name__ == "__main__":
    model = MLPModels(input_dim=2, output_dim=2,
              layers=1, hidden_units=3,
              model_count=3000, device=torch.device('cuda:0'))

    x = torch.randn((10, 2))
    print("This should be (10, 3000, 2)", model(x.cuda()).shape)
    model = LeNetModels(output_dim=2, width_factor=1, model_count=10, dataset='mnist').cuda()
    x_ori = torch.randn((10, 1, 28, 28)).cuda()
    out = model(x_ori)
    print(f"This should be (10, 20, 2): {out.shape}")
    for i in range(10):
        print("===="*10)
        weight = model.conv1.weight[6*i:6*(i+1), :, :, :]
        bias = model.conv1.bias[6*i:6*(i+1)]
        x = F.conv2d(x_ori, weight, bias)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        weight = model.conv2.weight[16*i:16*(i+1)]
        bias = model.conv2.bias[16*i:16*(i+1)]
        x = F.conv2d(x, weight, bias)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.reshape(10, 16 * 4 * 4, 1, 1)
        x = F.conv2d(x, model.fc1.weight[120 * i:120 * (i + 1)], model.fc1.bias[120 * i:120 * (i + 1)])
        x = F.relu(x)
        x = F.conv2d(x, model.fc2.weight[84 * i:84 * (i + 1)], model.fc2.bias[84 * i:84 * (i + 1)])
        x = F.relu(x)
        x = F.conv2d(x, model.fc3.weight[2*i:2*(i+1)], model.fc3.bias[2*i:2*(i+1)])

        print(f"this should be close to zero: {(x.flatten() - out[:, i:i+1].flatten()).abs().max().cpu().item(): 0.3f}" )

    model = LeNetModels(output_dim=2, width_factor=1, model_count=10, dataset='cifar10').cuda()
    x_ori = torch.randn((10, 3, 32, 32)).cuda()
    out = model(x_ori)
    print(f"This should be (10, 20, 2): {out.shape}")
    for i in range(10):
        print("===="*10)
        weight = model.conv1.weight[6*i:6*(i+1), :, :, :]
        bias = model.conv1.bias[6*i:6*(i+1)]
        x = F.conv2d(x_ori, weight, bias)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        weight = model.conv2.weight[16*i:16*(i+1)]
        bias = model.conv2.bias[16*i:16*(i+1)]
        x = F.conv2d(x, weight, bias)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.reshape(10, 16 * 5 * 5, 1, 1)
        x = F.conv2d(x, model.fc1.weight[120 * i:120 * (i + 1)], model.fc1.bias[120 * i:120 * (i + 1)])
        x = F.relu(x)
        x = F.conv2d(x, model.fc2.weight[84 * i:84 * (i + 1)], model.fc2.bias[84 * i:84 * (i + 1)])
        x = F.relu(x)
        x = F.conv2d(x, model.fc3.weight[2*i:2*(i+1)], model.fc3.bias[2*i:2*(i+1)])

        print(f"this should be close to zero: {(x.flatten() - out[:, i:i+1].flatten()).abs().max().cpu().item(): 0.3f}" )



























