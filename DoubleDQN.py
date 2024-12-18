import os
import sys
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import psutil
import time
import pygame

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import FlappyBirdGame as game

process = psutil.Process(os.getpid())

class ReplayBuffer:
    """Replay buffer to store and sample transitions for training the agent."""
    def __init__(self, capacity: int):
        self.capacity = capacity  # Maximum number of transitions to store
        self.buffer = deque(maxlen=capacity)  # Deque automatically handles overflow

    def store_transition(self, state, action, reward, next_state, done):
        """Stores a transition tuple in the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Samples a batch of transitions for training."""
        minibatch = random.sample(self.buffer, batch_size)  # Random sampling
        states, actions, rewards, next_states, dones = zip(*minibatch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=bool))

    def __len__(self):
        """Returns the current size of the buffer."""
        return len(self.buffer)


class QNetwork(nn.Module):
    """Neural network for approximating the Q-value function."""
    def __init__(self, state_dim=4, num_actions=2):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 64)  # Second hidden layer
        self.fc3 = nn.Linear(64, num_actions)  # Output layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the network."""
        x = F.relu(self.fc1(x))  # ReLU activation for the first layer
        x = F.relu(self.fc2(x))  # ReLU activation for the second layer
        return self.fc3(x)  # Linear output for Q-values


def train_double_dqn(num_frames=300_000, render=False, seed=0):
    """Trains a Double DQN agent for the Flappy Bird environment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # Set random seed for reproducibility

    env = game.FlappyBirdEnv(render=render)  # Initialize environment
    num_actions = 2  # Number of possible actions

    # Initialize policy and target networks
    policy_network = QNetwork().to(game.device)
    target_network = QNetwork().to(game.device)
    target_network.load_state_dict(policy_network.state_dict())  # Sync target with policy
    target_network.eval()  # Set target network to evaluation mode

    optimizer = optim.Adam(policy_network.parameters(), lr=1e-4)  # Optimizer
    replay_buffer = ReplayBuffer(capacity=50_000)  # Experience replay buffer

    # Hyperparameters
    gamma = 0.99  # Discount factor
    batch_size = 64
    epsilon = 1.0  # Initial exploration rate
    final_epsilon = 0.01  # Minimum exploration rate
    observe_frames = 5_000  # Frames to observe before training
    explore_frames = 100_000  # Frames to anneal epsilon
    target_network_update_freq = 2_000  # Frequency to update target network
    multiple_updates_per_step = 4  # Number of updates per environment step

    current_state = env.reset()  # Reset environment
    global_frame_count = 0

    current_episode_reward = 0.0  # Accumulated reward for the episode
    episode_count = 0  # Total number of episodes
    best_score = 0  # Track the best score

    # Logs for tracking training metrics
    frame_log = []
    reward_log = []
    score_log = []
    loss_log = []
    avg_reward_window = deque(maxlen=100)  # Rolling average reward
    avg_score_window = deque(maxlen=100)  # Rolling average score

    memory_usage_log = []  # Memory usage tracking
    wall_clock_time_log = []  # Time tracking
    start_time = time.time()

    while global_frame_count < num_frames:
        global_frame_count += 1  # Increment frame count

        # Action selection: epsilon-greedy strategy
        if global_frame_count < observe_frames:
            action = random.randint(0, num_actions - 1)  # Random action during observation
        else:
            epsilon_decay = (1.0 - final_epsilon) / explore_frames  # Linear decay of epsilon
            epsilon = max(final_epsilon, epsilon - epsilon_decay)  # Update epsilon
            if random.random() < epsilon:
                action = random.randint(0, num_actions - 1)  # Random action
            else:
                state_tensor = torch.from_numpy(current_state).unsqueeze(0).to(game.device)
                with torch.no_grad():
                    q_values = policy_network(state_tensor)
                    action = q_values.argmax(dim=1).item()  # Greedy action

        # Perform the action in the environment
        next_state, reward, done = env.step(action)
        replay_buffer.store_transition(current_state, action, reward, next_state, done)  # Store transition

        current_episode_reward += reward
        current_state = next_state  # Move to the next state

        # Training step if buffer has enough samples and observation phase is over
        if len(replay_buffer) > batch_size and global_frame_count > observe_frames:
            for _ in range(multiple_updates_per_step):
                # Sample a batch of transitions
                s_batch, a_batch, r_batch, s2_batch, d_batch = replay_buffer.sample(batch_size)

                # Convert numpy arrays to torch tensors
                state_batch = torch.from_numpy(s_batch).to(game.device)
                action_batch = torch.from_numpy(a_batch).to(game.device)
                reward_batch = torch.from_numpy(r_batch).to(game.device)
                next_state_batch = torch.from_numpy(s2_batch).to(game.device)
                done_batch = torch.from_numpy(d_batch).to(game.device)

                # Compute current Q-values
                all_q_values = policy_network(state_batch)
                current_q_values = all_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

                # Compute target Q-values using Double DQN
                with torch.no_grad():
                    next_q_policy = policy_network(next_state_batch)  # Q-values from policy network
                    best_next_actions = next_q_policy.argmax(dim=1, keepdim=True)  # Best actions
                    next_q_target = target_network(next_state_batch)  # Q-values from target network
                    best_next_q_values = next_q_target.gather(1, best_next_actions).squeeze(1)
                    target_q_values = reward_batch + gamma * best_next_q_values * (~done_batch)

                # Compute loss and backpropagate
                loss = F.mse_loss(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_log.append(loss.item())  # Log the loss

        # Update target network periodically
        if global_frame_count % target_network_update_freq == 0:
            target_network.load_state_dict(policy_network.state_dict())  # Sync networks
            print(f"[Frame {global_frame_count}] Target network updated. Epsilon={epsilon:.3f}")

        # Handle end of episode
        if done:
            episode_count += 1
            avg_reward_window.append(current_episode_reward)  # Update rolling reward
            avg_score_window.append(env.score)  # Update rolling score

            # Save the model if the score is the best
            if env.score > best_score:
                best_score = env.score
                torch.save(policy_network.state_dict(), "best_DQN_model.pth")
                print(f"New best score {best_score}! Model checkpoint saved.")

            print(f"[Frame {global_frame_count}] Episode {episode_count} done. "
                  f"Reward={current_episode_reward:.2f}, Score={env.score}")

            # Log episode metrics
            frame_log.append(global_frame_count)
            reward_log.append(current_episode_reward)
            score_log.append(env.score)
            current_episode_reward = 0.0  # Reset reward
            current_state = env.reset()  # Reset environment

        # Render environment if enabled
        if render:
            env.render()
            env.clock.tick(game.FPS)

        # Log memory usage and time periodically
        if global_frame_count % 1000 == 0:
            memory_usage_log.append(process.memory_info().rss / 1024**2)  # Log memory usage in MB
            wall_clock_time_log.append(time.time() - start_time)  # Log elapsed time

    env.close()  # Close the environment

    # Final evaluation and results saving
    total_steps_dqn = env.get_global_steps()
    print(f"[DQN] Total environment steps: {total_steps_dqn}")

    exploitation_scores = evaluate_policy(policy_network, env, num_episodes=20)  # Evaluate policy
    print("Exploitation run average score:", np.mean(exploitation_scores))

    ddqn_data = {
        'frame_log': frame_log,
        'reward_log': reward_log,
        'score_log': score_log,
        'loss_log': loss_log,
        'exploitation_scores': exploitation_scores,
        'memory_usage_log': memory_usage_log,
        'wall_clock_time_log': wall_clock_time_log
    }
    np.savez("ddqn_metrics.npz", **ddqn_data)  # Save training metrics

    # Plot training results
    plot_training_results(frame_log, reward_log, score_log, loss_log, exploitation_scores,
                          memory_usage_log, wall_clock_time_log)


def evaluate_policy(policy_network: nn.Module, env: game.FlappyBirdEnv, num_episodes=10):
    """Evaluates the policy by running it for a fixed number of episodes."""
    scores = []
    for episode_idx in range(num_episodes):
        state = env.reset()  # Reset environment
        done = False
        while not done:
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(game.device)
            with torch.no_grad():
                q_values = policy_network(state_tensor)
                action = q_values.argmax(dim=1).item()  # Choose greedy action

            next_state, _, done = env.step(action)  # Step in the environment
            state = next_state
        scores.append(env.score)  # Record the score for the episode
    return scores


def plot_training_results(frame_log, reward_log, score_log, loss_log,
                          exploitation_scores, memory_usage_log, wall_clock_time_log):
    """Plots various training metrics to visualize performance."""
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    axs = axs.flatten()

    # Plot episode rewards
    axs[0].plot(frame_log, reward_log, color='blue', label='Episode Reward')
    axs[0].set_title("Episode Reward vs Frames (DQN)")
    axs[0].set_xlabel("Frames")
    axs[0].set_ylabel("Reward")
    axs[0].legend()

    # Plot episode scores
    axs[1].plot(frame_log, score_log, color='green', label='Episode Score')
    axs[1].set_title("Episode Score vs Frames (DQN)")
    axs[1].set_xlabel("Frames")
    axs[1].set_ylabel("Score")
    axs[1].legend()

    # Plot training loss
    axs[2].plot(loss_log, color='red', label='Training Loss')
    axs[2].set_title("Training Loss (DQN)")
    axs[2].set_xlabel("Training Iterations")
    axs[2].set_ylabel("Loss")
    axs[2].legend()

    # Histogram of exploitation scores
    axs[3].hist(exploitation_scores, bins=10, color='purple')
    axs[3].set_title("Exploitation Score Distribution (DQN)")
    axs[3].set_xlabel("Score")
    axs[3].set_ylabel("Count")

    # Plot memory usage
    axs[4].plot(memory_usage_log, color='orange', label='Memory Usage (MB)')
    axs[4].set_title("Memory Usage Over Time (DQN) [Sampled every 1000 frames]")
    axs[4].set_xlabel("Samples (# per 1000 frames)")
    axs[4].set_ylabel("Memory (MB)")
    axs[4].legend()

    # Plot wall-clock time
    axs[5].plot(wall_clock_time_log, color='cyan', label='Wall-clock Time (s)')
    axs[5].set_title("Wall-clock Time (DQN) [Sampled every 1000 frames]")
    axs[5].set_xlabel("Samples (# per 1000 frames)")
    axs[5].set_ylabel("Time (s)")
    axs[5].legend()

    plt.tight_layout()
    plt.savefig("ddqn_training_plots.png")  # Save the plot as an image
    plt.show()


def play_flappy_with_ddqn(model_path="best_DQN_model.pth", render=True, num_games=1):
    """Loads a trained model and lets it play Flappy Bird."""
    env = game.FlappyBirdEnv(render=render)  # Initialize environment
    policy_network = QNetwork().to(game.device)  # Load trained network
    policy_network.load_state_dict(torch.load(model_path, map_location=game.device))
    policy_network.eval()  # Set network to evaluation mode

    for game_idx in range(num_games):
        state = env.reset()  # Reset environment for new game
        done = False
        while not done:
            # Handle user quitting during play
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    sys.exit()

            if render:
                env.render()  # Render the environment
                env.clock.tick(game.FPS if game.FPS else 30)

            state_tensor = torch.from_numpy(state).unsqueeze(0).to(game.device)
            with torch.no_grad():
                q_values = policy_network(state_tensor)
                action = q_values.argmax(dim=1).item()  # Choose action

            state, _, done = env.step(action)  # Perform action in environment

        print(f"Game {game_idx + 1} ended with score: {env.score}")

    env.close()
    pygame.quit()
