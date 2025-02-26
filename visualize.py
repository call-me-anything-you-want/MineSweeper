import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from MineSweepingPolicy import MineSweepingPolicy
from TrainSweepAndBury import generate_mines, sweep_mine
from MineSweepingEnv import random_mine_layout, MineSweepingEnv
import os
import imageio
from PIL import Image
import re
import math

def visualize_mines(board, filename='mines.png'):
    """
    Visualize a 10x10 minesweeper mines represented by a torch.tensor and save it as an image.
    
    Parameters:
    - board: torch.tensor, a 10x10 tensor where 1 represents an mine cell.
    - filename: str, the name of the file to save the image as.
    """
    board = board.view(10, 10)
    
    fig, ax = plt.subplots()
    
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            cell_value = board[i, j].item()
            rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
            if cell_value == 1:
                ax.add_patch(patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='r', facecolor='black'))
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def visualize_board(board, visualize_q=False, q_values=None, filename='minesweeper_with_q_board.png'):
    """
    Visualize a 10x10 minesweeper board represented by a torch.tensor with Q values as a heatmap.
    
    Parameters:
    - board: torch.tensor, a 10x10 tensor where -1 represents an unopened cell,
             and other numbers represent opened cells with the number of mines
             in the surrounding cells.
    - q_values: torch.tensor, a 10x10 tensor of Q values corresponding to each cell.
    - filename: str, the name of the file to save the image as.
    """

    board = board.view(10, 10)
    if visualize_q:
        assert q_values is not None, "Q values are required to visualize Q."
        q_values = q_values.view(10, 10)
    
    fig, ax = plt.subplots()
    
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            cell_value = board[i, j].item()
            rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
            if cell_value == -1:
                if visualize_q:
                    q_value = q_values[i, j].item()
                    normalized_q_value = (q_value - torch.min(q_values)) / (torch.max(q_values) - torch.min(q_values))
                    color = plt.cm.coolwarm(normalized_q_value)
                else:
                    color = 'black'
                ax.add_patch(patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='r', facecolor=color))
            else:
                ax.text(j + 0.5, i + 0.5, str(int(cell_value)), ha='center', va='center', fontsize=10)
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    if visualize_q:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=torch.min(q_values).item(), vmax=torch.max(q_values).item()))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar_ax.set_ylabel('Q Value', rotation=270, labelpad=15)
    
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def load_policy(path):
    policy = MineSweepingPolicy(torch.tensor((10, 10)))
    policy.load_state_dict(torch.load(path))
    return policy

def play_and_visualize(sweep_policy, bury_policy=None, mines=None, path=None):
    if not os.path.exists(path):
        os.makedirs(path)
    if bury_policy is not None:
        mines, mine_history = generate_mines(10, bury_policy, epsilon=0)
    elif mines is None:
        mines = random_mine_layout((10, 10), 10)
    if bury_policy is not None:
        for i, mine in enumerate(mine_history):
            mine = mine[0][0]
            visualize_mines(mine, f"{path}/mine_{i}.png")
    visualize_mines(mines, f"{path}/mine.png")
    data, _ = sweep_mine(mines, sweep_policy, epsilon=0)
    for i, transition in enumerate(data):
        transition = transition[0]
        obs = transition[0]
        visualize_board(
            obs,
            visualize_q=True,
            q_values=sweep_policy(obs).detach(),
            filename=f"{path}/{i+1}_board_and_q.png"
        )
        if i == len(data) - 1:
            final_obs = transition[3]
            if (final_obs == -1).sum() == 10:
                visualize_board(
                    final_obs,
                    visualize_q=False,
                    filename=f"{path}/{i+2}_board_and_q.png"
                )
            else:
                final_obs = torch.ones_like(final_obs)
                visualize_mines(
                    final_obs,
                    filename=f"{path}/{i+2}_board_and_q.png"
                )

def create_gif_from_pngs(png_paths, gif_path):
    """
    Create a GIF file from a list of PNG file paths.
    
    Parameters:
    - png_paths: list of str, a list containing the file paths of PNG images.
    - gif_path: str, the file path where the GIF will be saved.
    """

    images = []
    max_width = 0
    max_height = 0
    for png_path in png_paths:
        image = Image.open(png_path)
        max_width = max(max_width, image.width)
        max_height = max(max_height, image.height)

    for png_path in png_paths:
        image = Image.open(png_path)
        new_image = Image.new('RGBA', (max_width, max_height), (255, 255, 255, 255))
        new_image.paste(image, (0, 0))
        images.append(new_image)

    images = [np.array(image) for image in images]

    imageio.mimsave(gif_path, images, 'GIF', fps=2)

def plot_win_rates(folder_path, output_image_path):
    file_names = os.listdir(folder_path)
    
    data = {i: [] for i in range(1, 11)}
    min_iters = {i: float('inf') for i in range(1, 11)}
    
    pattern = re.compile(r'(sweep_)?(\d+)_(\d\.\d+)_(\d+)\.pt')
    
    for file_name in file_names:
        match = pattern.match(file_name)
        if match:
            _, num, win_rate, iter = match.groups()
            num, iter = int(num), int(iter)
            data[num].append((iter, float(win_rate)))
            min_iters[num] = min(min_iters[num], iter)
    
    plt.figure(figsize=(10, 6))

    win_rates = []
    for num in sorted(data.keys()):
        win_rates.extend(data[num])
    win_rates = sorted(win_rates, key=lambda x: x[0])
    processed_win_rates = []
    for i in range(len(win_rates)):
        processed_win_rates.append((win_rates[i][0], win_rates[i][1]))
        if i != len(win_rates) - 1:
            processed_win_rates.append((win_rates[i+1][0], win_rates[i][1]))
    # plt.plot(*zip(*processed_win_rates), marker='o')
    plt.plot(*zip(*processed_win_rates))
    
    for num, min_iter in min_iters.items():
        if min_iter != float('inf'):
            plt.axvline(x=min_iter, color='gray', linestyle='--', linewidth=1)
    
    plt.xlabel('ln(iter)')
    plt.ylabel('win rate')
    plt.xscale('log')
    plt.title('Win Rates by Iteration')
    # plt.grid(True, axis="y")
    
    plt.savefig(output_image_path)
    plt.close()

def export_onnx(sweep_policy, observation, path):
    torch.onnx.export(
        sweep_policy,
        observation,
        f"{path}/sweep.onnx",
        export_params=True,
    )

if __name__ == "__main__":
    sweep_policy = load_policy("./two_policy_saved_models/sweep_10_0.79_9635.pt")
    bury_policy = load_policy("./two_policy_saved_models/bury_10_0.79_9635.pt")
    play_and_visualize(sweep_policy, bury_policy, path="./visualize")
    files = os.listdir("./visualize")
    files = [f for f in files if f.endswith(".png")]
    files = sorted(
        [f for f in files if f.startswith(tuple(f"{i}" for i in range(10)))],
        key=lambda f: int(f.split("_")[0])
    )
    files = [f"./visualize/{f}" for f in files]
    create_gif_from_pngs(files, "./visualize/game.gif")
    # plot_win_rates("./two_policy_saved_models", "./visualize/win_rate.png")
