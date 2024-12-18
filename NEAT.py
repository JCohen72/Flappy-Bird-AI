import os
import sys
import time
import pickle
import random
import numpy as np
import pygame
import neat
import matplotlib.pyplot as plt
import psutil
from collections import deque

import FlappyBirdGame as game

process = psutil.Process(os.getpid())

class NEATTrainer:
    def __init__(self, config, num_generations=500, render=False, eval_episodes=10, seed=0):
        """Initialize the NEAT Trainer with given configuration and parameters."""
        self.config = config
        self.num_generations = num_generations
        self.render = render
        self.eval_episodes = eval_episodes
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        # Initialize the game environment
        self.env = game.FlappyBirdEnv(render=self.render)

        # Logs and metrics for analysis
        self.generation_log = []
        self.best_fitness_log = []
        self.avg_fitness_log = []
        self.eval_scores = []
        self.time_per_generation = []
        self.memory_usage_log = []

        # Best genome tracking
        self.best_genome = None
        self.best_genome_fitness = float('-inf')
        self.start_time = None

    def eval_genomes(self, genomes, config):
        """Evaluate genomes by playing the game and assigning fitness based on performance."""
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)  # Create neural network for genome

            state = self.env.reset()  # Reset environment to initial state
            done = False
            while not done:
                # Normalize state inputs for the neural network
                inputs = state / np.array([512.0, 12.0, 300.0, 512.0], dtype=np.float32)
                outputs = net.activate(inputs.tolist())  # Generate outputs from the neural network
                action = np.argmax(outputs)  # Choose action with highest output value

                # Perform the action and update the state
                next_state, _, done = self.env.step(action)
                state = next_state

                if self.render:
                    self.env.render()  # Render the environment if enabled
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.env.close()
                            pygame.quit()
                            sys.exit()

            # Assign fitness score to the genome
            genome.fitness = self.env.score
            if genome.fitness > self.best_genome_fitness:
                self.best_genome_fitness = genome.fitness
                self.best_genome = genome

    def run_training(self):
        """Run the NEAT training process for the specified number of generations."""
        population = neat.Population(self.config)  # Create NEAT population
        population.config.fitness_threshold = float('inf')

        # Add NEAT reporters for logging and debugging
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)

        self.start_time = time.time()  # Track start time for performance analysis

        # Run NEAT for the configured number of generations
        winner = population.run(self.eval_genomes, self.num_generations)
        print(f"\n[NEAT] Training Complete. Best Genome Fitness (Score): {self.best_genome_fitness}")

        # Save the best genome to a file
        with open("best_NEAT_genome.pickle", "wb") as f:
            pickle.dump(winner, f)
        print("[NEAT] Best genome saved to best_NEAT_genome.pickle")

        # Log training statistics
        gen_idx_list = list(range(len(stats.most_fit_genomes)))
        self.generation_log = gen_idx_list
        self.best_fitness_log = [g.fitness for g in stats.most_fit_genomes]
        self.avg_fitness_log = stats.get_fitness_mean()

        # Calculate total training time
        total_training_time = time.time() - self.start_time
        self.time_per_generation = np.linspace(0, total_training_time, len(self.generation_log))

        # Track memory usage per generation
        self.memory_usage_log = []
        for g_idx in gen_idx_list:
            self.memory_usage_log.append(process.memory_info().rss / 1024**2)  # Convert memory to MB

        # Evaluate the final best genome
        self.eval_scores = self._final_exploitation_eval(winner, self.eval_episodes)
        self.plot_training_results()  # Plot training metrics

        # Save metrics to a file for later analysis
        neat_data = {
            'generation_log': self.generation_log,
            'best_fitness_log': self.best_fitness_log,
            'avg_fitness_log': self.avg_fitness_log,
            'eval_scores': self.eval_scores,
            'time_per_generation': self.time_per_generation,
            'memory_usage_log': self.memory_usage_log
        }
        np.savez("neat_metrics.npz", **neat_data)

    def _final_exploitation_eval(self, best_genome, episodes=10):
        """Evaluate the best genome over multiple episodes to determine its average performance."""
        print(f"\n[NEAT] Performing final exploitation evaluation ({episodes} episodes)...")
        net = neat.nn.FeedForwardNetwork.create(best_genome, self.config)
        scores = []

        for ep_i in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                # Normalize state inputs for the neural network
                inputs = state / np.array([512.0, 12.0, 300.0, 512.0], dtype=np.float32)
                outputs = net.activate(inputs.tolist())  # Generate outputs from the neural network
                action = np.argmax(outputs)  # Choose action with highest output value
                next_state, _, done = self.env.step(action)
                state = next_state
            scores.append(self.env.score)  # Log the score for this episode
            print(f"Episode {ep_i+1} ended with score: {self.env.score}")

        avg_score = np.mean(scores)  # Calculate average score over all episodes
        print(f"[NEAT] Final exploitation run average score: {avg_score:.2f}")
        return scores

    def plot_training_results(self):
        """Generate and save plots to visualize NEAT training performance and metrics."""
        fig, axs = plt.subplots(3, 2, figsize=(14, 12))
        axs = axs.flatten()

        # Plot best genome fitness over generations
        axs[0].plot(self.generation_log, self.best_fitness_log, color='blue', label='Best Fitness (Score)')
        axs[0].set_title("Best Genome Score over Generations (NEAT)")
        axs[0].set_xlabel("Generation")
        axs[0].set_ylabel("Score")
        axs[0].legend()

        # Plot average genome fitness over generations
        axs[1].plot(self.generation_log, self.avg_fitness_log, color='green', label='Average Score')
        axs[1].set_title("Average Genome Score over Generations (NEAT)")
        axs[1].set_xlabel("Generation")
        axs[1].set_ylabel("Score")
        axs[1].legend()

        # Plot histogram of evaluation scores
        axs[2].hist(self.eval_scores, bins=10, color='purple')
        axs[2].set_title("Exploitation Score Distribution (NEAT)")
        axs[2].set_xlabel("Score")
        axs[2].set_ylabel("Count")

        # Plot runtime per generation
        axs[3].plot(self.generation_log, self.time_per_generation, color='red', label='Runtime')
        axs[3].set_title("NEAT Wall-clock Time Over Generations")
        axs[3].set_xlabel("Generation")
        axs[3].set_ylabel("Time (s)")
        axs[3].legend()

        # Plot memory usage per generation
        axs[4].plot(self.generation_log, self.memory_usage_log, color='orange', label='Memory Usage (MB)')
        axs[4].set_title("NEAT Memory Usage (sampled once per generation)")
        axs[4].set_xlabel("Generation")
        axs[4].set_ylabel("Memory (MB)")
        axs[4].legend()

        # Placeholder for additional metrics
        axs[5].set_title("Placeholder for additional metric (NEAT)")
        axs[5].axis('off')

        plt.tight_layout()
        plt.savefig("neat_training_plots.png")
        plt.show()

    def close_env(self):
        """Close the game environment to free resources."""
        self.env.close()


def train_neat(num_generations=500, render=False, eval_episodes=10, seed=0):
    """Main function to configure and train NEAT for Flappy Bird."""
    config_path = os.path.join(os.path.dirname(__file__), "config-neat-flappy.txt")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    trainer = NEATTrainer(config, num_generations=num_generations, render=render,
                          eval_episodes=eval_episodes, seed=seed)
    trainer.run_training()

    # Log total steps taken during training
    total_steps_neat = trainer.env.get_global_steps()
    print(f"[NEAT] Total environment steps: {total_steps_neat}")

    trainer.close_env()


def play_flappy_with_neat(genome_path="best_NEAT_genome.pickle", render=True, num_games=1):
    """Play Flappy Bird using a trained genome from NEAT."""
    config_path = os.path.join(os.path.dirname(__file__), "config-neat-flappy.txt")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Load the trained genome from a file
    with open(genome_path, "rb") as f:
        winner_genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(winner_genome, config)  # Create neural network for the genome
    env = game.FlappyBirdEnv(render=render)  # Initialize game environment

    for game_idx in range(num_games):
        state = env.reset()
        done = False
        while not done:
            if render:
                env.render()
                env.clock.tick(game.FPS if hasattr(env, 'FPS') else 30)  # Set FPS for rendering
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        pygame.quit()
                        sys.exit()

            # Normalize state inputs and predict action
            inputs = state / np.array([512.0, 12.0, 300.0, 512.0], dtype=np.float32)
            outputs = net.activate(inputs.tolist())
            action = np.argmax(outputs)
            next_state, _, done = env.step(action)
            state = next_state

        print(f"Game {game_idx + 1} ended with score: {env.score}")

    env.close()
    pygame.quit()
