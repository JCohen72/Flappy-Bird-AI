import os
import random
import numpy as np
import torch
import psutil
import numpy as np
import matplotlib.pyplot as plt

# Set a fixed seed for reproducibility across all modules
SEED = 0
random.seed(SEED)  # Seed for Python's random module
np.random.seed(SEED)  # Seed for NumPy random module
torch.manual_seed(SEED)  # Seed for PyTorch random module

# Ensure deterministic behavior in PyTorch's cuDNN backend
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Get the current process for memory and CPU usage tracking
process = psutil.Process(os.getpid())

# Import custom modules for game-playing and training
import HumanPlay as human
import DoubleDQN as ddqn
import NEAT as neat

def plot_synchronized_comparison(ddqn_metrics_path="ddqn_metrics.npz", neat_metrics_path="neat_metrics.npz"):
    """
    Generate comparative plots between Double DQN and NEAT metrics.

    Args:
        ddqn_metrics_path (str): Path to the Double DQN metrics file.
        neat_metrics_path (str): Path to the NEAT metrics file.

    Outputs:
        Saves and displays comparison plots for reward, memory usage, and more.
    """
    # Load metrics data from the provided file paths
    ddqn_data = np.load(ddqn_metrics_path)
    neat_data = np.load(neat_metrics_path)

    # Extract relevant logs and metrics for plotting
    ddqn_frame_log = ddqn_data['frame_log']
    ddqn_reward_log = ddqn_data['reward_log']
    ddqn_score_log = ddqn_data['score_log']
    ddqn_exploitation_scores = ddqn_data['exploitation_scores']
    ddqn_memory_log = ddqn_data['memory_usage_log']
    ddqn_time_log = ddqn_data['wall_clock_time_log']

    neat_generation_log = neat_data['generation_log']
    neat_best_fitness_log = neat_data['best_fitness_log']
    neat_avg_fitness_log = neat_data['avg_fitness_log']
    neat_eval_scores = neat_data['eval_scores']
    neat_time_per_generation = neat_data['time_per_generation']
    neat_memory_log = neat_data['memory_usage_log']

    # Create a 3x2 grid of subplots for comparative analysis
    fig, axs = plt.subplots(3, 2, figsize=(14, 14))
    axs = axs.flatten()

    # Plot DQN reward against frames
    axs[0].plot(ddqn_frame_log, ddqn_reward_log, label='DQN Reward', color='blue')
    axs[0].set_title("DQN Episode Reward vs Frames")
    axs[0].set_xlabel("Frames")
    axs[0].set_ylabel("Reward")
    axs[0].legend()

    # Plot NEAT best fitness against generations
    axs[1].plot(neat_generation_log, neat_best_fitness_log, label='NEAT Best Fitness', color='orange')
    axs[1].set_title("NEAT Best Fitness vs Generations")
    axs[1].set_xlabel("Generation")
    axs[1].set_ylabel("Score")
    axs[1].legend()

    # Compare exploitation score distributions
    axs[2].hist(ddqn_exploitation_scores, bins=10, alpha=0.7, label='DQN Exploit Scores', color='blue')
    axs[2].hist(neat_eval_scores, bins=10, alpha=0.7, label='NEAT Exploit Scores', color='orange')
    axs[2].set_title("Exploitation Score Distribution Comparison")
    axs[2].set_xlabel("Score")
    axs[2].set_ylabel("Count")
    axs[2].legend()

    # Plot memory usage over time for DQN and NEAT
    axs[3].plot(range(len(ddqn_memory_log)), ddqn_memory_log, label='DQN Memory (MB)', color='blue')
    axs[3].plot(range(len(neat_memory_log)), neat_memory_log, label='NEAT Memory (MB)', color='orange')
    axs[3].set_title("Memory Usage Comparison")
    axs[3].set_xlabel("Sample Index")
    axs[3].set_ylabel("Memory (MB)")
    axs[3].legend()

    # Compare wall-clock times for DQN and NEAT
    axs[4].plot(range(len(ddqn_time_log)), ddqn_time_log, label='DQN Time (s)', color='blue')
    axs[4].plot(neat_generation_log, neat_time_per_generation, label='NEAT Time (s)', color='orange')
    axs[4].set_title("Wall-clock Time Comparison")
    axs[4].set_xlabel("Samples (DQN) vs Generation (NEAT)")
    axs[4].set_ylabel("Time (s)")
    axs[4].legend()

    # Placeholder subplot for future metrics
    axs[5].set_title("Placeholder: Additional Metric")
    axs[5].axis('off')

    # Adjust layout and save/display the plot
    plt.tight_layout()
    plt.savefig("comparison_plots.png")
    plt.show()

def main():
    """
    Main function to control the mode of execution.
    Modes include:
    - HUMAN: Play the game manually.
    - DQN: Train or play with Double DQN.
    - NEAT: Train or play with NEAT.
    - PLOT: Generate comparison plots.
    """
    mode = input("Enter mode (HUMAN / DQN / NEAT / PLOT): ").strip().upper()

    if mode == "HUMAN":
        # Launch the manual game-playing mode
        human.play_flappy_human()

    elif mode == "DQN":
        # Ask the user whether to train a new model or watch a trained one
        train_or_play = input("Do you want to train DQN or watch a trained model? (train / watch): ").strip().lower()

        if train_or_play == "train":
            # Train Double DQN with the specified parameters
            ddqn.train_double_dqn(num_frames=500_000, render=False, seed=SEED)
        else:
            # Play the game using a pre-trained Double DQN model
            ddqn.play_flappy_with_ddqn(model_path="best_DQN_model.pth", render=True, num_games=1)

    elif mode == "NEAT":
        # Ask the user whether to train a new genome or watch a trained one
        train_or_play = input("Do you want to train NEAT or watch a trained genome? (train / watch): ").strip().lower()

        if train_or_play == "train":
            # Train NEAT with the specified parameters
            neat.train_neat(num_generations=46, render=False, eval_episodes=10, seed=SEED)
        else:
            # Play the game using a pre-trained NEAT genome
            neat.play_flappy_with_neat(genome_path="best_NEAT_genome.pickle", render=True, num_games=1)
    elif mode == "PLOT":
        # Generate comparison plots for NEAT and Double DQN metrics
        plot_synchronized_comparison()

    else:
        # Handle invalid mode selections
        print("Invalid mode selected. Please choose from HUMAN, DQN, or NEAT.")

if __name__ == "__main__":
    main()
