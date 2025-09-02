import os
import json
import hashlib
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
from dataclasses import dataclass

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path',
                        type=str,
                        help='Path to config.json file')
    args = parser.parse_args()

    with open(args.config_path) as f:
        data = json.load(f)

        # Create figure with 3 subplots
        plt.figure(figsize=(12, 12))
        
        # First subplot for accuracies
        ax1 = plt.subplot(3, 1, 1)
        sizes = [ModelSpec(**s) for s in data["model_sizes"]]
        expansion_steps = [data["exp_freq"]]

        log_steps = data["log_steps"]
        train_accuracies = data["train_accuracies"]
        test_accuracies = data["test_accuracies"]
        norms = data["weight_norms"]
        grad_norms = data["grad_norms"]

        plt.plot(log_steps, train_accuracies, color='red', label='train')
        plt.plot(log_steps, test_accuracies, color='green', label='test')


        for step in expansion_steps:
            plt.axvline(x=step,
                        color='blue',
                        linestyle='dotted',
                        linewidth=1,
                        alpha=0.5)

        plt.legend()
        plt.xscale('log')
        plt.ylim(0, 1)
        ax1.set_ylabel("Accuracy")

        # Second subplot for weight norm
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(log_steps, norms, color='purple', label='weight norm')
        ax2.set_ylabel("Weight Norm")
        ax2.set_xscale('log')
        ax2.legend()

        for step in expansion_steps:
            ax2.axvline(x=step,
                        color='blue',
                        linestyle='dotted',
                        linewidth=1,
                        alpha=0.5)

        # Third subplot for gradient norm
        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(log_steps, grad_norms, color='orange', label='grad norm')
        ax3.set_ylabel("Gradient Norm")
        ax3.set_xlabel("Optimization Steps")
        ax3.set_xscale('log')
        ax3.legend()

        for step in expansion_steps:
            ax3.axvline(x=step,
                        color='blue',
                        linestyle='dotted',
                        linewidth=1,
                        alpha=0.5)

        # plot the green line when test accuracy hits 95% for all plots
        if any(a >= 0.95 for a in test_accuracies):
            time_to_95_pct_test = log_steps[min(
                i for i, acc in enumerate(test_accuracies) if acc >= 0.95)]
            ax1.plot([time_to_95_pct_test] * 2, [0, 1],
                     color='green',
                     linestyle='--')
            ax1.text(time_to_95_pct_test + 10, 0.65,
                     f"@{time_to_95_pct_test} test acc\nhits 95%")
            
            # Add vertical line to weight norm plot
            ax2.plot([time_to_95_pct_test] * 2, ax2.get_ylim(),
                     color='green',
                     linestyle='--')
            
            # Add vertical line to gradient norm plot  
            ax3.plot([time_to_95_pct_test] * 2, ax3.get_ylim(),
                     color='green', 
                     linestyle='--')


        size_str = "dmodel,dmlp,nhead: " + " | ".join(
            ["_".join(f"{v}" for k, v in vars(s).items()) for s in sizes])
        plt.suptitle("1L Transformer on Modular Addition (p=113)\n" +
                     size_str + f"\nexp@{data['exp_freq']} t-{data['total_steps']}",
                     fontsize=8)
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(os.path.dirname(args.config_path), "plot.png"), dpi=300)
        plt.close()
