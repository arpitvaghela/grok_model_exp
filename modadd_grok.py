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
import seaborn as sns
from dataclasses import dataclass

from datetime import datetime
import hashlib
import json

from model import params_pad_to_shape, Transformer, GradientLogger
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
    logits, attn_scores_masked = model(x)
    logits = logits[:, -1]
    return torch.nn.functional.cross_entropy(logits, labels), attn_scores_masked

def full_accuracy(model, data, device):
    loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False)
    # Take the final position only
    x, labels = next(iter(loader))
    x = x.to(device)
    labels = labels.to(device)
    logits = model(x)[0][:, -1]
    predictions = torch.argmax(logits, dim=1)
    return torch.sum(predictions == labels).item() / len(labels)


def train_and_plot(sizes,
                   train,
                   test,
                   t_steps=5000,
                   exp_freq=None,
                   exp_thres=0.2,
                   log_dir="./logs/"):
    log_steps = []
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    norms = []
    grad_norms = []
    attention_scores = []

    # if not exp_freq:
    #     exp_freq = t_steps // len(sizes)
    size_index = 0
    model = Transformer(
        num_layers=1,
        d_vocab=equals_token + 1,
        d_model=sizes[0].d_model,
        d_mlp=sizes[0].d_mlp,
        d_k=sizes[0].d_head,
        d_v=sizes[0].d_head,
        num_heads=sizes[0].n_head,
        n_ctx=3,  # context length
        act_type='ReLU',
        use_cache=False,
        use_ln=False  # use LayerNorm
    ).to(device)
    grad_logger = GradientLogger()
    grad_logger.register_hooks(model)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-3,
                                  weight_decay=1.0,
                                  betas=(0.9, 0.98))

    pbar = tqdm(range(t_steps),
                desc="Training",
                postfix={
                    "val_acc": 0,
                    "train_acc": 0,
                    "w_norm": 0,
                    "g_norm": 0
                })
    for epoch in pbar:
        # if epoch % exp_freq == 0:
        if len(test_accuracies
               ) and test_accuracies[-1] > exp_thres and size_index == 0:
            # idx = epoch // exp_freq
            size_index = 1
            exp_freq = epoch
            if size_index > 0 and size_index < len(sizes):
                print(
                    f"@epoch {epoch}\t{sizes[size_index -1]} -> {sizes[size_index]}"
                )

                model_2 = Transformer(
                    num_layers=1,
                    d_vocab=equals_token + 1,
                    d_model=sizes[size_index].d_model,
                    d_mlp=sizes[size_index].d_mlp,
                    d_k=sizes[size_index].d_head,
                    d_v=sizes[size_index].d_head,
                    num_heads=sizes[size_index].n_head,
                    n_ctx=3,  # context length
                    act_type='ReLU',
                    use_cache=False,
                    use_ln=False  # use LayerNorm
                ).to(device)
                model_2.load_state_dict(
                    params_pad_to_shape(model.state_dict(),
                                        model_2.state_dict()))
                model = model_2
                optimizer = torch.optim.AdamW(model.parameters(),
                                              lr=1e-3,
                                              weight_decay=1.0,
                                              betas=(0.9, 0.98))
                grad_logger.register_hooks(model)

        train_loss, attn_scores_masked = full_loss(model, train, device)
        if epoch % 30 == 0:
            with torch.no_grad():
                log_steps.append(epoch)
                test_loss, attn_scores_masked = full_loss(model, test, device)
                train_losses.append(train_loss.item())
                test_losses.append(test_loss.item())
                train_acc = full_accuracy(model, train, device)
                test_acc = full_accuracy(model, test, device)
                train_accuracies.append(train_acc)
                test_accuracies.append(test_acc)
                weight_norm = np.sqrt(
                    sum(
                        param.pow(2).sum().item()
                        for param in model.parameters()))
                norms.append(weight_norm)

                # Store mean attention scores across all heads and batches
                mean_attn = torch.mean(torch.stack(attn_scores_masked),
                                       dim=0)  # Average across layers
                mean_attn = mean_attn.mean(dim=0)  # Average across batches
                attention_scores.append(mean_attn.cpu().numpy())

        train_loss.backward()

        if epoch % 30 == 0:
            # Compute total gradient norm (L2)
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm += param_norm.item()**2
            grad_norm = np.sqrt(total_grad_norm)
            grad_norms.append(grad_norm)
            # Update tqdm postfix with latest metrics
            pbar.set_postfix(val_acc=f"{test_acc:.3f}",
                             train_acc=f"{train_acc:.3f}",
                             w_norm=f"{weight_norm:.3f}",
                             g_norm=f"{grad_norm:.3f}")

        optimizer.step()
        optimizer.zero_grad()

    grad_logger.save("grads.pt")
    grad_logger.plot_grad_norms()
    grad_logger.plot_grad_hist()
    
    # Pad attention scores to match largest shape before storing
    if attention_scores:
        # Find max shape across all attention matrices
        max_shape = max(score.shape for score in attention_scores)

        # Pad smaller matrices with zeros to match max shape
        padded_scores = []
        for score in attention_scores:
            if score.shape != max_shape:
                padded = np.zeros(max_shape)
                padded[:score.shape[0], :score.shape[1]] = score
                padded_scores.append(padded)
            else:
                padded_scores.append(score)

        attention_scores = padded_scores

    data = {
        "log_steps": log_steps,
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies,
        "weight_norms": norms,
        "grad_norms": grad_norms,
        "model_sizes": [vars(s) for s in sizes],
        "exp_freq": exp_freq,
        "total_steps": t_steps,
        "attention_scores": attention_scores
    }

    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    tag = "1L_modadd"
    raw_string = "\n".join(
        ["_".join(f"{k}_{v}" for k, v in vars(s).items()) for s in sizes])
    short_hash = hashlib.md5(
        raw_string.encode()).hexdigest()[-6:].upper()  # e.g. 'E1234A'

    # exp_name = f"{now}__{tag}__{short_hash}"
    exp_name = " | ".join(
        ["_".join(f"{v}" for k, v in vars(s).items())
         for s in sizes]) + f"@step{exp_freq}"
    exp_path = os.path.join(log_dir, exp_name)

    os.makedirs(exp_path, exist_ok=True)
    # Save numpy arrays using numpy's save format
    np.save(os.path.join(exp_path, "attention_scores.npy"),
            np.array(data["attention_scores"]))

    # Save rest of data as JSON, excluding the numpy arrays
    data_json = data.copy()
    del data_json["attention_scores"]

    with open(os.path.join(exp_path, "config.json"), "w") as f:
        json.dump(data_json, f, indent=2)

    # # Create figure with subplots
    # fig, (ax1, ax2) = plt.subplots(2,
    #                                1,
    #                                figsize=(10, 12),
    #                                height_ratios=[2, 1])

    # # Plot accuracies and norms
    # expansion_steps = [exp_freq * i for i in range(1, len(sizes))]

    # ax1.plot(log_steps, train_accuracies, color='red', label='train')
    # ax1.plot(log_steps, test_accuracies, color='green', label='test')

    # if any(a >= 0.95 for a in test_accuracies):
    #     time_to_95_pct_test = log_steps[min(
    #         i for i, acc in enumerate(test_accuracies) if acc >= 0.95)]
    #     ax1.plot([time_to_95_pct_test] * 2, [0, 1],
    #              color='green',
    #              linestyle='--')
    #     ax1.text(time_to_95_pct_test + 10, 0.65,
    #              f"@{time_to_95_pct_test} test acc\nhits 95%")

    # for step in expansion_steps:
    #     ax1.axvline(x=step,
    #                 color='blue',
    #                 linestyle='dotted',
    #                 linewidth=1,
    #                 alpha=0.2)

    # ax1.legend()
    # ax1.set_xlabel("Optimization Steps")
    # ax1.set_ylim(0, 1)
    # ax1.set_ylabel("Accuracy")

    # ax1_2 = ax1.twinx()
    # ax1_2.set_ylabel("Weight/Grad Norm", color='purple')
    # ax1_2.plot(log_steps, norms, color='purple', label='weight norm')
    # ax1_2.plot(log_steps, grad_norms, color='orange', label='grad norm')
    # ax1_2.set_ylim(27, 63)
    # ax1.set_xscale('log')
    # ax1_2.legend(loc=(0.015, 0.72))

    # # Plot attention heatmap
    # attention_scores = np.array(attention_scores) # shape b,i,q,h
    # sns.heatmap(attention_scores.T,
    #             ax=ax2,
    #             cmap='viridis',
    #             xticklabels=log_steps[::len(log_steps) // 10],
    #             yticklabels=[
    #                 'pos ' + str(i) for i in range(attention_scores.shape[1])
    #             ])
    # ax2.set_xlabel('Training Step')
    # ax2.set_ylabel('Position')
    # ax2.set_title('Attention Scores Over Training')

    # size_str = "dmodel,dmlp,nhead: " + " | ".join(
    #     ["_".join(f"{v}" for k, v in vars(s).items()) for s in sizes])
    # fig.suptitle("1L Transformer on Modular Addition (p=113)\n" + size_str +
    #              f"\nexp@{exp_freq} t-{t_steps}",
    #              fontsize=8)
    # plt.tight_layout()
    # plt.savefig(os.path.join(log_dir, exp_name, "plot.png"), dpi=300)
    # plt.close()


if __name__ == "__main__":

    for seed in [42]: # , 66, 94, 17, 31]:
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

        base_sizes = [[8, 32, 1]]# [[4, 16, 1], [8,32,1], [16,64,2], [32, 128, 4]]
        target_sizes = [[128, 512, 4]]# [[64, 256, 4], [128,512,4], [256, 1024, 4]]
        exp_thresholds = [0.1] # [0.1, 0.25, 0.4]

        # log_dir = f"log/modadd/base/seed_{seed}"
        log_dir = os.path.join("log", "modadd", "base", f"seed_{seed}")
        os.makedirs(log_dir, exist_ok=True)

        # for size in base_sizes + target_sizes:
        #     sizes = [ModelSpec(size[0], size[1], size[2])]
        #     print(f"Training {sizes}")
        #     # exp_name = " | ".join(["_".join(f"{v}" for k, v in vars(s).items()) for s in sizes])
        #     train_and_plot(sizes,train, test, 10_000, -1, log_dir=log_dir)

        for base, target, et in itertools.product(base_sizes, target_sizes, exp_thresholds):
            sizes = [ModelSpec(base[0], base[1], base[2]), ModelSpec(target[0], target[1], target[2])]

            # exp_name = " | ".join(["_".join(f"{v}" for k, v in vars(s).items()) for s in sizes])
            log_dir = os.path.join("log", "modadd", "exp",f"thres_{et}" ,f"seed_{seed}")
            os.makedirs(log_dir, exist_ok=True)

            print(f"Training {sizes}")

            train_and_plot(sizes, train, test, 5_000, -1, log_dir=log_dir,exp_thres=et)
