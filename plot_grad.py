import os
import torch
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_grads(filepath):
    grads = torch.load(filepath)

    # Plot gradient values over time for each parameter
    num_params = len(grads)
    n_rows = (num_params + 1) // 2  # 2 plots per row
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 8 * n_rows))
    axes = axes.flatten()

    for i, (param_name, grad_list) in enumerate(grads.items()):
        print(f"\nParameter: {param_name}")
        print(f"Shape of each gradient tensor: {grad_list[0].shape}")
        print(f"Shape of each gradient tensor: {grad_list[-1].shape}")

        target_shape = grad_list[-1].shape
        padded_grads = []
        for g in grad_list:
            if g.shape != target_shape:
                # Create padded tensor filled with NaN
                padded = torch.full(target_shape,
                                    float('nan'),
                                    device=g.device)
                # Copy original values
                slices = tuple(slice(0, s) for s in g.shape)
                padded[slices] = g
                padded_grads.append(padded)
            else:
                padded_grads.append(g)
        padded_grads_tensor = torch.stack(padded_grads).view(
            len(padded_grads), -1)
        min_val = torch.min(
            padded_grads_tensor[~torch.isnan(padded_grads_tensor)])
        max_val = torch.max(
            padded_grads_tensor[~torch.isnan(padded_grads_tensor)])
        bins = torch.linspace(min_val, max_val, 200)
        # Get histogram counts for each training step
        hist_counts = []
        for step in range(padded_grads_tensor.shape[0]):
            step_grads = padded_grads_tensor[step]
            # Remove NaN values before computing histogram
            valid_grads = step_grads[~torch.isnan(step_grads)]

            counts = torch.histc(valid_grads,
                                 bins=len(bins) - 1,
                                 min=min_val,
                                 max=max_val)
            # Convert counts to log scale, adding small constant to avoid log(0)
            log_counts = torch.log(counts)
            hist_counts.append(log_counts)

        # Stack all histogram counts into a single tensor
        hist_counts = torch.stack(hist_counts)

        # Create a heatmap in the corresponding subplot
        im = axes[i].imshow(hist_counts.numpy().T,
                            aspect='auto',
                            interpolation='nearest',
                            cmap='Blues')
        plt.colorbar(im, ax=axes[i], label='Log Count')
        axes[i].set_xlabel('Training Step')
        axes[i].set_ylabel('Gradient Value Bin')

        # Add bin values as y-axis ticks
        num_ticks = 10  # Reduce number of ticks for readability
        tick_indices = np.linspace(0, len(bins) - 1, num_ticks, dtype=int)
        axes[i].set_yticks(tick_indices)
        axes[i].set_yticklabels([f'{bins[i]:.2e}' for i in tick_indices])

        axes[i].set_title(f'Gradient Distribution Evolution - {param_name}')
        print(f"Shape of concatenated gradients: {padded_grads_tensor.shape}")

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # Save the plot
    output_dir = os.path.dirname(filepath)
    plot_path = os.path.join(output_dir, 'gradient_heatmaps.png')
    plt.savefig(plot_path)
    plt.close()


def get_training_info(filepath):
    # Get config.json path from parent directory
    config_path = os.path.join(os.path.dirname(filepath), 'config.json')

    import json
    with open(config_path) as f:
        data = json.load(f)
        log_steps = data["log_steps"]
        test_accuracies = data["test_accuracies"]
    return log_steps, test_accuracies


def plot_attention_scores(filepath):
    # Load attention scores
    attention_scores = np.load(filepath)

    # Get training info
    log_steps, test_accuracies = get_training_info(filepath)

    # Get dimensions
    n_steps, n_h, n_q, n_k = attention_scores.shape

    # print(f"Attention scores shape: {attention_scores.shape}")
    # print(f"Number of heads: {n_h}")
    # print(f"Number of query positions: {n_q}")
    # print(f"Number of key positions: {n_k}")
    # print(f"Number of steps: {n_steps}")

    # Create figure and subplots for each head
    n_rows = 2  # Display max 3 heads per row
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # Handle different cases for axes array
    if n_h == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.reshape(n_rows, n_cols)

    # Initialize heatmaps
    heatmaps = []
    for head in range(n_h):
        # print(f"Head {head}")
        row = head // n_cols
        col = head % n_cols
        ax = axes[row, col]
        ax.set_title(f'Head {head}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')

        # Create initial heatmap
        heatmap = ax.imshow(attention_scores[0, head],
                            vmin=0,
                            vmax=1,
                            cmap='viridis',
                            aspect='auto')
        plt.colorbar(heatmap, ax=ax)
        heatmaps.append(heatmap)

    # Animation update function
    def update(frame):
        for head in range(n_h):
            heatmaps[head].set_array(attention_scores[frame, head])
        # Use actual training step from log_steps
        training_step = log_steps[frame] if frame < len(log_steps) else frame

        fig.suptitle(
            f'Training Step {training_step}, Test Accuracy {test_accuracies[frame]:.2f}'
        )
        return heatmaps

    # Create animation
    anim = animation.FuncAnimation(fig,
                                   update,
                                   frames=tqdm(range(n_steps)),
                                   interval=500,
                                   blit=True)

    # Save animation
    plt.tight_layout()
    output_dir = os.path.dirname(filepath)
    output_path = os.path.join(output_dir, 'attention_animation.mp4')
    anim.save(output_path, writer='ffmpeg')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir', type=str, help='Path to grads.pt file')
    args = parser.parse_args()
    plot_grads(args.filepath)


if __name__ == "__main__":
    main()
