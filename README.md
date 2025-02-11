# README

## Overview
This project applies three approaches (Human play, Double Deep Q-Network (DQN), and NEAT evolutionary algorithms) to the Flappy Bird game. The code provides options to:

1. **HUMAN**: Play the game manually.
2. **DQN**: Train or watch a pre-trained Double DQN agent.
3. **NEAT**: Train or watch a pre-trained NEAT-based agent.
4. **PLOT**: Generate comparison plots of DQN and NEAT metrics.

The code logs training metrics, saves trained models, and produces visualizations to evaluate performance and resource usage (memory/time). This README will guide you through setup, running the code, and understanding the outputs.

<img width="286" alt="Screenshot 2025-02-11 at 1 08 34 PM" src="https://github.com/user-attachments/assets/197efffd-b127-4893-87c9-39ce7bc177f5" />

## Contents of This Directory
- **`main.py`**: The main entry point to run the code. It imports modules like `HumanPlay`, `DoubleDQN`, and `NEAT` and provides a command-line interface to select a mode.
- **`HumanPlay.py`**: Allows a human player to control the Flappy Bird agent.
- **`DoubleDQN.py`**: Contains functions and classes for training and evaluating a Double DQN agent.
- **`NEAT.py`**: Contains functions and classes for training and evaluating a NEAT-based agent.
- **`FlappyBirdGame.py`**: Defines the Flappy Bird environment with a `FlappyBirdEnv` class. This handles state representation, stepping through the environment, rendering, and scoring.
- **`config-neat-flappy.txt`**: Configuration file for the NEAT algorithm (population size, mutation rates, etc.).
- **`best_DQN_model.pth`** (generated after training): The best-performing Double DQN model weights.
- **`best_NEAT_genome.pickle`** (generated after training): The best-performing NEAT genome.
- **`ddqn_metrics.npz`**: Logged metrics from DQN training (rewards, scores, memory usage, etc.).
- **`neat_metrics.npz`**: Logged metrics from NEAT training.
- **`ddqn_training_plots.png`**: Plots of DQN training metrics.
- **`neat_training_plots.png`**: Plots of NEAT training metrics.
- **`comparison_plots.png`**: Comparison plots synchronizing DQN and NEAT metrics side-by-side.

## Prerequisites
- **Python 3.x**
- **PyTorch** for DQN training and inference.
- **NEAT-Python** library for the NEAT algorithm.
- **Pygame** for rendering the Flappy Bird environment.
- **Matplotlib**, **NumPy**, **psutil** for plotting and resource logging.

## How to Run
To start the program, run:
python main.py

You will be prompted to choose a mode. The available modes are:

### 1. HUMAN Mode
**Description:** Lets you play Flappy Bird using spacebar to flap.

**Steps:**
- Run the code and enter `HUMAN` when prompted.
- Press `SPACE` to make the bird jump.
- The game ends when the bird collides with an obstacle or the ground.
- The final score prints on the console.

### 2. DQN Mode
**Description:** Train a Double DQN agent or watch a pre-trained model play.

**Training Steps:**
- Choose `DQN` at the mode prompt.
- Enter `train` to start training.
- The code will:
  - Run for a specified number of frames (`num_frames`).
  - Periodically save the best-performing model as `best_DQN_model.pth`.
  - Log metrics (rewards, scores, memory usage, wall-clock time).
  - Save metrics in `ddqn_metrics.npz`.

After training, you can visualize results from `ddqn_training_plots.png`.

**Watching a Trained Model:**
- Choose `DQN` at the mode prompt.
- Enter `watch`.
- Ensure `best_DQN_model.pth` is present.

### 3. NEAT Mode
**Description:** Train a population of NEAT genomes to navigate Flappy Bird or watch the best genome.

**Training Steps:**
- Choose `NEAT` at the mode prompt.
- Enter `train`.
- The code will:
  - Use the NEAT configuration `config-neat-flappy.txt`.
  - Run evolutionary training for the specified number of generations.
  - Save the best-performing genome to `best_NEAT_genome.pickle`.
  - Log metrics (best fitness, average fitness, memory usage, wall-clock time).
  - Save metrics in `neat_metrics.npz`.

After training, you can visualize results from `neat_training_plots.png`.

**Watching a Trained Genome:**
- Choose `NEAT` at the mode prompt.
- Enter `watch`.
- Make sure `best_NEAT_genome.pickle` is available.

### 4. PLOT Mode
**Description:** Plot a synchronized comparison between DQN and NEAT metrics if both `ddqn_metrics.npz` and `neat_metrics.npz` are available.

**Steps:**
- Choose `PLOT` at the mode prompt.
- `comparison_plots.png` will be generated, showing side-by-side comparisons of:
  - DQN Episode Rewards vs. Frames
  - NEAT Best Fitness vs. Generations
  - Exploitation score distributions
  - Memory usage comparisons
  - Wall-clock time comparisons

## Understanding Output Files
- **`ddqn_metrics.npz`**: Contains arrays for frames, rewards, scores, losses, exploitation scores, memory logs, and time logs during DQN training.
- **`neat_metrics.npz`**: Similar metrics for NEAT (generations, best fitness, average fitness, exploitation scores, memory, and time).
- **Trained models**:
  - `best_DQN_model.pth`: The best DQN weights.
  - `best_NEAT_genome.pickle`: The best NEAT genome.

**Plots**:
- **`ddqn_training_plots.png`**: DQN training progress plots.
- **`neat_training_plots.png`**: NEAT training progress plots.
- **`comparison_plots.png`**: Side-by-side comparison of DQN and NEAT metrics.

## License and Credits
- **Flappy Bird** is a trademark of its creator, Dong Nguyen. This is a research/learning implementation
