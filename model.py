import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from tqdm.auto import tqdm
from collections import OrderedDict, defaultdict

import matplotlib.pyplot as plt
import itertools

class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []

    def give_name(self, name):
        # Called by the model at initialisation
        self.name = name

    def add_hook(self, hook, dir='fwd'):
        # Hook format is fn(activation, hook_name)
        # Change it into PyTorch hook format (this includes input and output,
        # which are the same for a HookPoint)
        def full_hook(module, module_input, module_output):
            return hook(module_output, name=self.name)
        if dir=='fwd':
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir=='bwd':
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")

    def remove_hooks(self, dir='fwd'):
        if (dir=='fwd') or (dir=='both'):
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []
        if (dir=='bwd') or (dir=='both'):
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []
        if dir not in ['fwd', 'bwd', 'both']:
            raise ValueError(f"Invalid direction {dir}")

    def forward(self, x):
        return x


class GradientLogger:

    def __init__(self):
        self.grads = defaultdict(list)

    def register_hooks(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad, n=name: self.grads[n].append(
                    grad.detach().cpu().clone()))

    def save(self, path):
        torch.save(dict(self.grads), path)

    def load(self, path="grads.pt"):
        self.grads = torch.load(path)

    def plot_grad_norms(self):
        for name, grad_list in self.grads.items():
            norms = [g.norm().item() for g in grad_list]
            plt.plot(norms, label=name)
        plt.xlabel("Training Step")
        plt.ylabel("Gradient Norm")
        plt.legend()
        plt.title("Gradient Norms per Parameter")
        plt.show()

    def plot_grad_hist(self, step=-1):
        """Plot histogram of gradients at a given step (default last)."""
        for name, grad_list in self.grads.items():
            if grad_list:
                plt.hist(grad_list[step].flatten().numpy(),
                         bins=50,
                         alpha=0.5,
                         label=name)
        plt.xlabel("Gradient Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.title(f"Gradient Distribution at Step {step}")
        plt.show()


# Embed & Unembed
class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab)/np.sqrt(d_model))

    def forward(self, x):
        return torch.einsum('dbp -> bpd', self.W_E[:, x])

class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab)/np.sqrt(d_vocab))

    def forward(self, x):
        return (x @ self.W_U)

# Positional Embeddings
class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(max_ctx, d_model)/np.sqrt(d_model))

    def forward(self, x):
        return x+self.W_pos[:x.shape[-2]]

# RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, d_model, epsilon = 1e-4, model=[None]):
        super().__init__()
        self.model = model
        self.w_ln = nn.Parameter(torch.ones(d_model))
        self.b_ln = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x):
        if self.model[0].use_ln:
            # Compute root mean square over the last dimension
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.epsilon)
            x = x / rms
            x = x * self.w_rms
            x = x + self.b_rms
            return x
        else:
            return x
# Attention
class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_v, d_k, n_ctx, model):
        super().__init__()
        self.model = model
        self.W_K = nn.Parameter(torch.randn(num_heads, d_k, d_model)/np.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(num_heads, d_k, d_model)/np.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(num_heads, d_v, d_model)/np.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(d_model, d_v * num_heads)/np.sqrt(d_model))
        self.register_buffer('mask', torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_k
        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()

    def forward(self, x):
        k = self.hook_k(torch.einsum('ihd,bpd->biph', self.W_K, x))
        q = self.hook_q(torch.einsum('ihd,bpd->biph', self.W_Q, x))
        v = self.hook_v(torch.einsum('ihd,bpd->biph', self.W_V, x))
        attn_scores_pre = torch.einsum('biph,biqh->biqp', k, q)
        attn_scores_masked = torch.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
        attn_matrix = self.hook_attn(F.softmax(self.hook_attn_pre(attn_scores_masked/np.sqrt(self.d_head)), dim=-1))
        z = self.hook_z(torch.einsum('biph,biqp->biqh', v, attn_matrix))
        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        out = torch.einsum('df,bqf->bqd', self.W_O, z_flat)
        return out, attn_scores_masked # biqh

# MLP Layers
class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, act_type, model):
        super().__init__()
        self.model = model
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model)/np.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp)/np.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.act_type = act_type
        # self.ln = LayerNorm(d_mlp, model=self.model)
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
        assert act_type in ['ReLU', 'GeLU']

    def forward(self, x):
        x = torch.einsum('md,bpd->bpm', self.W_in, x) + self.b_in

        if self.act_type=='ReLU':
            x = F.relu(x)
        elif self.act_type=='GeLU':
            x = F.gelu(x)
        x = self.hook_post(x)
        x = torch.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out
        return x

# Transformer Block
class TransformerBlock(nn.Module):

    def __init__(self, d_model, d_mlp, d_v, d_k, num_heads, n_ctx, act_type,
                 model):
        super().__init__()
        self.model = model
        # self.ln1 = LayerNorm(d_model, model=self.model)
        self.attn = Attention(d_model,
                              num_heads,
                              d_v,
                              d_k,
                              n_ctx,
                              model=self.model)
        # self.ln2 = LayerNorm(d_model, model=self.model)
        self.mlp = MLP(d_model, d_mlp, act_type, model=self.model)
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x):
        attn_out, attn_scores_masked = self.attn((self.hook_resid_pre(x)))
        self.hook_attn_out(attn_out)
        x = self.hook_resid_mid(x + attn_out)
        x = self.hook_resid_post(x + self.hook_mlp_out(self.mlp((x))))
        return x, attn_scores_masked


# Full transformer
class Transformer(nn.Module):
    def __init__(self, num_layers, d_vocab, d_model, d_mlp, d_v, d_k, num_heads, n_ctx, act_type, use_cache=False, use_ln=True):
        super().__init__()
        self.cache = {}
        self.use_cache = use_cache

        self.embed = Embed(d_vocab, d_model)
        self.pos_embed = PosEmbed(n_ctx, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, d_mlp, d_v, d_k, num_heads, n_ctx, act_type, model=[self]) for i in range(num_layers)])
        # self.ln = LayerNorm(d_model, model=[self])
        self.unembed = Unembed(d_vocab, d_model)
        self.use_ln = use_ln

        for name, module in self.named_modules():
            if type(module)==HookPoint:
                module.give_name(name)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)

        attn_scores_masked = []
        for block in self.blocks:
            x, attn_scores_masked_block = block(x)
            attn_scores_masked.append(attn_scores_masked_block)

        # x = self.ln(x)
        x = self.unembed(x)
        return x[:,-1], attn_scores_masked

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def hook_points(self):
        return [module for name, module in self.named_modules() if 'hook' in name]

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks('fwd')
            hp.remove_hooks('bwd')

    def cache_all(self, cache, incl_bwd=False):
        # Caches all activations wrapped in a HookPoint
        def save_hook(tensor, name):
            cache[name] = tensor.detach()
        def save_hook_back(tensor, name):
            cache[name+'_grad'] = tensor[0].detach()
        for hp in self.hook_points():
            hp.add_hook(save_hook, 'fwd')
            if incl_bwd:
                hp.add_hook(save_hook_back, 'bwd')

def params_pad_to_shape(source_state, target_state, function_preserving=True):

    final_state = OrderedDict()
    for name, target_tensor in target_state.items():
        # print(f"{name}, shape: {target_tensor.shape}")

        if name in source_state:
            source_tensor = source_state[name]
            # print(f"Found in source state dict, shape: {source_tensor.shape}, sum: {source_tensor.sum().item()}" )

            assert len(source_tensor.shape) == len(target_tensor.shape)

            to_pad_shape = []
            for dim in range(len(target_tensor.shape)):
                dim_diff = target_tensor.shape[dim] - source_tensor.shape[dim]
                to_pad_shape.append([0, dim_diff])

            # in pytorch the dimensions needs to be reversed
            to_pad_shape = to_pad_shape[::-1]
            # print("to pad shape:", to_pad_shape)

            zero_init = False
            head_pad = 0
            s_n_head = 0
            # MLP expansion Sec 3.1
            if "W_out" in name and to_pad_shape[0][1] > 0:
                # print(f"Expanding MLP: {name}")
                zero_init = True

            # Heads addition Sec 3.2 or Head expansion Sec 3.3
            if ("W_O" in name and to_pad_shape[0][1] > 0):
                name_of_val_param = name[:-1] + "V" # W_V for the W_O
                # check if W_V dim has changed at dim=1
                t_n_head, t_d_head, t_d_model = target_state[name_of_val_param].shape
                sx_n_head, s_d_head, s_d_model = source_state[name_of_val_param].shape
                head_pad = t_d_head - s_d_head
                s_n_head = sx_n_head
                zero_init = (t_n_head - s_n_head) > 0
                to_pad_shape[0][1] -= s_n_head * head_pad
                # print(f"Updated zero padding shape {to_pad_shape}")


            # Attention expansion Sec 3.4
            if "W_K" in name and to_pad_shape[1][1] > 0:
                # print(f"Expainding attention W_k: {name}")
                zero_init = True

            # Hidden dimension expansion Sec 3.5
            if ("W_out" in name and to_pad_shape[-1][1] > 0) or \
                ("b_out" in name and to_pad_shape[-1][1] > 0) or \
                ("W_O" in name and to_pad_shape[1][1] > 0) or \
                ("W_pos" in name and to_pad_shape[0][1] > 0) or \
                ("W_E" in name and to_pad_shape[-1][1] > 0) or \
                ("W_U" in name and to_pad_shape[0][1] > 0):
                # print(f"Expainding hidden dimension param: {name}")
                zero_init = True

            # Key matrix scaling for attentin expansion Sec 3.4
            if "W_K" in name and to_pad_shape[1][1] > 0:
                source_tensor = source_tensor * torch.sqrt(
                    torch.tensor(target_tensor.shape[1] / source_tensor.shape[1])
                )

            # norm scaling for attentino expansion Sec 3.4
            if "w_ln" in name and to_pad_shape[-1][1] > 0:
                raise NotImplementedError("The Norms are skipped in OmniGROK check implementation!")

            if head_pad:
                # we update the source_tensor to match the requried d_head'
                # print("padding head")
                source_tensor = source_tensor.view(source_tensor.shape[0], s_n_head, -1)
                pad = torch.randn((source_tensor.shape[0], s_n_head, head_pad),
                                  dtype=source_tensor.dtype, device=source_tensor.device) * 1e-9
                source_tensor = torch.cat([source_tensor, pad], dim=2).view(source_tensor.shape[0], -1)

            if zero_init:
                # print("0 init")
                padded_tensor = F.pad(
                    source_tensor, [p for pair in to_pad_shape for p in pair],
                    "constant", value=0
                )                # Create noise tensor and mask only the padded region
                noise = torch.randn_like(padded_tensor) * 1e-8
                mask = (padded_tensor == 0) & (F.pad(source_tensor, [p for pair in to_pad_shape for p in pair], value=float('nan')).isnan())

                padded_tensor += noise * mask  # Add noise only to padded parts
            else:
                # print("No 0 init")
                # Start from target_tensor's random values
                padded_tensor = target_tensor.clone()

                # Copy source tensor
                slices = tuple(slice(0, s) for s in source_tensor.shape)
                padded_tensor[slices] = source_tensor

            final_state[name] = padded_tensor

        else: # if param not in source tensor
            # print("Not found in source state dict")s
            # Layer addition Sec 3.6
            if "W_O" in name or "W_out" in name:
                # print("0 init")
                padded_tensor = torch.randn_like(target_tensor, dtype=target_tensor.dtype, device=target_tensor.device) * 1e-9
            else:
                # print("No 0 init")
                padded_tensor =  target_tensor.clone()

            final_state[name] = padded_tensor

        # print("-----")
    return final_state
