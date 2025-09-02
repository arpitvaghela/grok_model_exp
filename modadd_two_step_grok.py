import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from tqdm.auto import tqdm
from collections import OrderedDict

import matplotlib.pyplot as plt
from dataclasses import dataclass

from datetime import datetime
import hashlib
import json

from model import params_pad_to_shape, Transformer
import itertools

@dataclass
class ModelSpec:
    d_model: int
    d_mlp: int
    n_head: int

    def __post_init__(self):
        assert self.d_model % self.n_head == 0, \
            f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})"

    @property
    def d_head(self):
        return self.d_model // self.n_head

    def __repr__(self):
        return (f"ModelSpec(d_model={self.d_model}, "
            f"d_mlp={self.d_mlp}, "
            f"n_head={self.n_head}, "
            f"d_head={self.d_head})")

def full_loss(model, data, device):
    loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False)
    # Take the final position only
    x, labels = next(iter(loader))
    x = x.to(device)
    labels = labels.to(device)
    logits = model(x)[:, -1]
    return torch.nn.functional.cross_entropy(logits, labels)

def full_accuracy(model, data, device):
    loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False)
    # Take the final position only
    x, labels = next(iter(loader))
    x = x.to(device)
    labels = labels.to(device)
    logits = model(x)[:, -1]
    predictions = torch.argmax(logits, dim=1)
    return torch.sum(predictions == labels).item() / len(labels)

def train_and_plot(sizes, train, test, t_steps=5000, exp_freq=None, log_dir="./logs/"):
    log_steps = []
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    norms = []
    grad_norms = []

    # if not exp_freq:
    #     exp_freq = t_steps // len(sizes)
    size_index = 0
    model = Transformer(num_layers=1, 
                        d_vocab=equals_token+1, 
                        d_model=sizes[0].d_model,
                        d_mlp=sizes[0].d_mlp,
                        d_k=sizes[0].d_head,
                        d_v=sizes[0].d_head,
                        num_heads=sizes[0].n_head,
                        n_ctx=3, # context length
                        act_type='ReLU', 
                        use_cache=False, 
                        use_ln=False # use LayerNorm
                    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    
    exp_thres = 0.1 # first threshold
    exp_freq1 = None
    exp_freq2 = None

    for epoch in tqdm(range(t_steps)):
        # if epoch % exp_freq == 0:
        if len(test_accuracies) and (test_accuracies[-1] > exp_thres): #  or (exp_thres == 0.1 and epoch == 1000)):
            # idx = epoch // exp_freq
            if size_index == 1:
                size_index = 2
                exp_freq2 = epoch
                exp_thres = 1.1 # no further expansion

            elif size_index == 0:
                size_index = 1 
                exp_freq1 = epoch
                exp_thres = 0.3 # second threshold

            
            if size_index > 0 and size_index < len(sizes):
                print( f"@epoch {epoch}\t{sizes[size_index -1]} -> {sizes[size_index]}")
                
                model_2 = Transformer(num_layers=1, 
                            d_vocab=equals_token+1, 
                            d_model=sizes[size_index].d_model,
                            d_mlp=sizes[size_index].d_mlp,
                            d_k=sizes[size_index].d_head,
                            d_v=sizes[size_index].d_head,
                            num_heads=sizes[size_index].n_head,
                            n_ctx=3, # context length
                            act_type='ReLU', 
                            use_cache=False, 
                            use_ln=False # use LayerNorm
                        ).to(device)
                model_2.load_state_dict(
                    params_pad_to_shape(model.state_dict(), model_2.state_dict())
                )
                model = model_2
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
            
        train_loss = full_loss(model, train, device)
        if epoch % 30 == 0:
            with torch.no_grad():
                log_steps.append(epoch)
                test_loss = full_loss(model, test, device)
                train_losses.append(train_loss.item())
                test_losses.append(test_loss.item())
                train_accuracies.append(full_accuracy(model, train, device))
                test_accuracies.append(full_accuracy(model, test, device))
                norms.append(np.sqrt(sum(param.pow(2).sum().item() for param in model.parameters())))

        train_loss.backward()

        # if epoch%30 ==0:
        #      with torch.no_grad():
        #         # Compute total gradient norm (L2)
        #         total_grad_norm = 0.0
        #         for p in model.parameters():
        #             if p.grad is not None:
        #                 param_norm = p.grad.data.norm(2)
        #                 total_grad_norm += param_norm.item() ** 2

        #         grad_norms.append(total_grad_norm)

        optimizer.step()
        optimizer.zero_grad()


    data = {
        "log_steps": log_steps,
        "train_accuracies":train_accuracies,
        "test_accuracies":test_accuracies,
        "weight_norms": norms,
        "model_sizes": [vars(s) for s in sizes],
        "exp_freq1": exp_freq1,
        "exp_freq2": exp_freq2,
        "total_steps": t_steps
    }

    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    raw_string = "\n".join(["_".join(f"{k}_{v}" for k, v in vars(s).items()) for s in sizes])
    short_hash = hashlib.md5(raw_string.encode()).hexdigest()[-6:].upper()  # e.g. 'E1234A'

    # exp_name = f"{now}__{tag}__{short_hash}"
    exp_name = " | ".join(["_".join(f"{v}" for k, v in vars(s).items()) for s in sizes]) # + f"@step{exp_freq}"
    exp_path = os.path.join(log_dir, exp_name)

    os.makedirs(exp_path, exist_ok=True)
    
    with open(os.path.join(exp_path, "config.json"), "w") as f:
        json.dump(data, f, indent=2)
    
    ax = plt.subplot(1, 1, 1)
    expansion_steps = [exp_freq1, exp_freq2]

    plt.plot(log_steps, train_accuracies, color='red', label='train')
    plt.plot(log_steps, test_accuracies, color='green', label='test')
    

    
    if any(a>=0.95 for a in test_accuracies):
        time_to_95_pct_test = log_steps[min(i for i, acc in enumerate(test_accuracies) if acc >= 0.95)]
        plt.plot([time_to_95_pct_test]*2, [0, 1], color='green', linestyle='--')
        plt.text(time_to_95_pct_test+10, 0.65, f"@{time_to_95_pct_test} test acc\nhits 95%")

    for step in expansion_steps:
        if step:
            plt.axvline(x=step, color='blue', linestyle='dotted', linewidth=1, alpha=0.2)


    plt.legend()

    plt.xlabel("Optimization Steps")
    # plt.xlim(8, 2*10**4)
    
    plt.ylim(0,1)
    ax.set_ylabel("Accuracy")
    ax2 = ax.twinx()
    ax2.set_ylabel("Weight Norm", color='purple')
    ax2.plot(log_steps, norms, color='purple', label='weight norm')
    ax2.set_ylim(27, 63)
    
    plt.xscale('log')
    plt.legend(loc=(0.015, 0.72))
    
    size_str = "dmodel,dmlp,nhead: " + " | ".join(["_".join(f"{v}" for k, v in vars(s).items()) for s in sizes])
    plt.title("1L Transformer on Modular Addition (p=113)\n"+ size_str + f"\nexp@{exp_freq} t-{t_steps}", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, exp_name, "plot.png"), dpi=300)
    plt.close()


if __name__ == "__main__":

    for seed in [6345, 346, 645]:
        p = 113
        fraction = 0.3

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        torch.set_default_dtype(torch.float64)
        print(f"Using {device}")

        equals_token = p
        x, y = torch.meshgrid(torch.arange(p), torch.arange(p), indexing='ij')
        x = x.flatten()
        y = y.flatten()
        
        # plus = torch.ones(x.shape, dtype=torch.int64) * plus_token
        equals = torch.ones(x.shape, dtype=torch.int64) * equals_token
        prompts = torch.stack([x, y, equals], dim=1).to(device)
        answers = ((x + y) % p).to(device)

        data = torch.utils.data.TensorDataset(prompts, answers)
        train, test = torch.utils.data.random_split(data, 
                                        [int(fraction * len(data)),
                                        len(data) - int(fraction * len(data))
                                        ])
        
        base_sizes = [[8,32,1]] # [[4, 16, 1], [8,32,1]]
        base2_sizes = [[32,128,4], [64, 256, 4]]
        target_sizes = [[128,512,4], [256, 1024, 4]]
        
        # exp_thresholds = [0.1, 0.25, 0.4]

        # log_dir = f"log/modadd/base/seed_{seed}"
        
        for base, base2, target in itertools.product(base_sizes,base2_sizes, target_sizes):
            sizes = [ModelSpec(base[0], base[1], base[2]), ModelSpec(base2[0], base2[1], base2[2]), ModelSpec(target[0], target[1], target[2])]
            
            # exp_name = " | ".join(["_".join(f"{v}" for k, v in vars(s).items()) for s in sizes])
            log_dir = os.path.join("log", "modadd2", "exp" ,f"seed_{seed}")
            os.makedirs(log_dir, exist_ok=True)

            print(f"Training {sizes}")
            
            train_and_plot(sizes, train, test, 10_000, -1, log_dir=log_dir)
        
        log_dir = os.path.join("log", "modadd2", "base", f"seed_{seed}")
        os.makedirs(log_dir, exist_ok=True)

        for size in base_sizes + target_sizes:
            sizes = [ModelSpec(size[0], size[1], size[2])]
            print(f"Training {sizes}")
            # exp_name = " | ".join(["_".join(f"{v}" for k, v in vars(s).items()) for s in sizes])
            train_and_plot(sizes,train, test, 100_000, -1, log_dir=log_dir)


    
    