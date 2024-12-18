import os
import random
import numpy as np
import pygame
import torch

# Global Constants
SCREEN_WIDTH = 288  # Width of the game screen
SCREEN_HEIGHT = 512  # Height of the game screen
FLOOR_POSITION = 450  # Y-coordinate of the floor/base

PIPE_VELOCITY = 4  # Speed at which pipes move
GRAVITY = 1.2  # Gravity effect on the bird
MAX_FALL_SPEED = 12  # Maximum falling speed for the bird
JUMP_VELOCITY = -9  # Velocity when the bird jumps

FPS = 0  # Frame rate (0 or None for unlimited frame rate)

# Rewards
REWARD_DEATH = -5.0  # Penalty for colliding or falling
REWARD_PASS_PIPE = 1.0  # Reward for passing a pipe
REWARD_ALIVE = 0.1  # Reward for staying alive

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Initialize Pygame and load assets
pygame.init()
BACKGROUND_IMAGE = pygame.image.load(os.path.join("assets/bg.png"))
BASE_IMAGE = pygame.image.load(os.path.join("assets/base-1.png"))
PIPE_IMAGE = pygame.image.load(os.path.join("assets/pipe.png"))
BIRD_IMAGES = [
    pygame.image.load(os.path.join("assets/bird1.png")),
    pygame.image.load(os.path.join("assets/bird2.png")),
    pygame.image.load(os.path.join("assets/bird3.png"))
]

class Bird:
    """ Represents the Flappy Bird character. """
    def __init__(self, x: float, y: float):
        self.x = x  # X-coordinate of the bird
        self.y = y  # Y-coordinate of the bird
        self.velocity = 0  # Current velocity of the bird
        self.image = BIRD_IMAGES[0]  # Initial bird image

    def jump(self):
        """ Makes the bird jump by setting its velocity. """
        self.velocity = JUMP_VELOCITY

    def move(self):
        """ Updates the bird's position based on its velocity and gravity. """
        self.velocity += GRAVITY  # Apply gravity
        if self.velocity > MAX_FALL_SPEED:  # Limit falling speed
            self.velocity = MAX_FALL_SPEED
        self.y += self.velocity  # Update vertical position

class Pipe:
    """ Represents the pipe obstacle for the Flappy Bird game. """
    GAP_SIZE = 160  # Gap size between the top and bottom pipes

    def __init__(self, x: float):
        self.x = x  # X-coordinate of the pipe
        self.pipe_top_image = pygame.transform.flip(PIPE_IMAGE, False, True)  # Flipped image for the top pipe
        self.pipe_bottom_image = PIPE_IMAGE  # Image for the bottom pipe

        self.height = 0  # Height of the top pipe
        self.top_y = 0  # Y-coordinate of the top pipe
        self.bottom_y = 0  # Y-coordinate of the bottom pipe
        self.passed = False  # Whether the bird has passed this pipe

        self._set_random_height()  # Randomize the height of the pipe

    def _set_random_height(self):
        """ Sets a random height for the pipe. """
        self.height = random.randrange(50, 280)
        self.top_y = self.height - self.pipe_top_image.get_height()
        self.bottom_y = self.height + self.GAP_SIZE

    def move(self):
        """ Moves the pipe to the left. """
        self.x -= PIPE_VELOCITY

    def has_collided(self, bird: Bird) -> bool:
        """ Checks if the bird has collided with this pipe. """
        bird_rect = pygame.Rect(
            bird.x, bird.y, BIRD_IMAGES[0].get_width(), BIRD_IMAGES[0].get_height()
        )
        top_rect = pygame.Rect(
            self.x, self.top_y, self.pipe_top_image.get_width(), self.pipe_top_image.get_height()
        )
        bottom_rect = pygame.Rect(
            self.x, self.bottom_y, self.pipe_bottom_image.get_width(), self.pipe_bottom_image.get_height()
        )
        return bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect)

class Base:
    """ Represents the moving base (floor). """
    BASE_WIDTH = BASE_IMAGE.get_width()  # Width of the base image

    def __init__(self, y: float):
        self.y = y  # Y-coordinate of the base
        self.x1 = 0  # X-coordinate of the first base segment
        self.x2 = self.BASE_WIDTH  # X-coordinate of the second base segment

    def move(self):
        """ Moves the base segments to the left to create a continuous scrolling effect. """
        self.x1 -= PIPE_VELOCITY
        self.x2 -= PIPE_VELOCITY

        # Loop the base segments when they go off-screen
        if self.x1 + self.BASE_WIDTH < 0:
            self.x1 = self.x2 + self.BASE_WIDTH
        if self.x2 + self.BASE_WIDTH < 0:
            self.x2 = self.x1 + self.BASE_WIDTH

class FlappyBirdEnv:
    """
    A simplified Flappy Bird environment that provides a feature vector:
    [bird.y, bird.velocity, distance_to_next_pipe, pipe_gap_center].
    """
    def __init__(self, render: bool = False):
        self.render_mode = render  # Whether to render the environment
        if render:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        else:
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        pygame.display.set_caption("AI Flappy Bird")
        self.clock = pygame.time.Clock()

        self.bird = None  # The bird character
        self.base = None  # The moving base
        self.pipe_list = []  # List of pipes on the screen
        self.score = 0  # Player's score
        self.done = False  # Whether the game is over

        self.global_step = 0  # Global step counter

        self.reset()  # Initialize the game

    def reset(self):
        """ Resets the environment to the initial state. """
        self.bird = Bird(50, 200)  # Initialize the bird
        self.base = Base(FLOOR_POSITION)  # Initialize the base
        self.pipe_list = [Pipe(300)]  # Add the first pipe
        self.score = 0
        self.done = False
        return self._get_state()

    def step(self, action: int):
        """ Advances the environment by one step based on the given action. """
        self.global_step += 1

        if action == 1:  # If the action is to jump
            self.bird.jump()

        self.bird.move()  # Update the bird's position
        self.base.move()  # Move the base

        reward = REWARD_ALIVE  # Default reward for being alive
        add_pipe = False  # Flag to add a new pipe
        pipes_to_remove = []  # List of pipes to remove

        for pipe in self.pipe_list:
            pipe.move()  # Move the pipe
            if pipe.has_collided(self.bird):  # Check for collision
                self.done = True  # End the game
                reward = REWARD_DEATH  # Apply death penalty

            if pipe.x + pipe.pipe_top_image.get_width() < 0:  # If the pipe is off-screen
                pipes_to_remove.append(pipe)

            if not pipe.passed and pipe.x < self.bird.x:  # If the bird passes the pipe
                pipe.passed = True
                add_pipe = True

        if add_pipe:
            self.score += 1  # Increase the score
            reward += REWARD_PASS_PIPE  # Add reward for passing a pipe
            self.pipe_list.append(Pipe(SCREEN_WIDTH + 10))  # Add a new pipe

        for pipe_to_remove in pipes_to_remove:
            self.pipe_list.remove(pipe_to_remove)  # Remove off-screen pipes

        # Check if the bird hits the floor or flies too high
        if ((self.bird.y + BIRD_IMAGES[0].get_height()) >= FLOOR_POSITION) or (self.bird.y < -50):
            self.done = True
            reward = REWARD_DEATH

        next_state = self._get_state()  # Get the next state
        return next_state, reward, self.done

    def _get_state(self):
        """ Constructs the state representation for the current environment. """
        next_pipe = None
        for pipe in self.pipe_list:
            if pipe.x + PIPE_IMAGE.get_width() > self.bird.x - 50:  # Find the next pipe
                next_pipe = pipe
                break
        if not next_pipe:
            next_pipe = self.pipe_list[0]

        distance_to_pipe_x = next_pipe.x - self.bird.x  # Horizontal distance to the next pipe
        gap_center = (next_pipe.bottom_y + next_pipe.top_y + next_pipe.pipe_top_image.get_height()) / 2  # Center of the pipe gap

        return np.array([self.bird.y, self.bird.velocity, distance_to_pipe_x, gap_center], dtype=np.float32)

    def get_global_steps(self):
        """ Returns the total number of steps taken in the environment. """
        return self.global_step
    
    def render(self):
        """ Renders the game environment. """
        if not self.render_mode:
            return

        self.screen.blit(BACKGROUND_IMAGE, (0, 0))  # Draw the background
        for pipe in self.pipe_list:
            top_pipe_image = pygame.transform.flip(PIPE_IMAGE, False, True)
            self.screen.blit(top_pipe_image, (pipe.x, pipe.top_y))  # Draw the top pipe
            self.screen.blit(PIPE_IMAGE, (pipe.x, pipe.bottom_y))  # Draw the bottom pipe

        self.screen.blit(BASE_IMAGE, (self.base.x1, self.base.y))  # Draw the first base segment
        self.screen.blit(BASE_IMAGE, (self.base.x2, self.base.y))  # Draw the second base segment
        self.screen.blit(BIRD_IMAGES[0], (self.bird.x, self.bird.y))  # Draw the bird

        font = pygame.font.SysFont("comicsans", 40)
        score_text = font.render(f"{self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, 20))  # Display the score

        pygame.display.update()  # Update the display

    def close(self):
        """ Closes the game and quits Pygame. """
        pygame.quit()
