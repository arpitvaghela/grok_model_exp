import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from tqdm import tqdm

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type=str, help='Path to attention_scores.npy file')
    args = parser.parse_args()

    plot_attention_scores(args.filepath)
