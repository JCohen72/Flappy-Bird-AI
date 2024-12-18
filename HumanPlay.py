import pygame

import FlappyBirdGame as game

def play_flappy_human():
    """
    Runs the Flappy Bird game for a human player.
    The player controls the bird using the spacebar, and the game renders in real-time.
    """
    # Initialize the Flappy Bird environment with rendering enabled
    env = game.FlappyBirdEnv(render=True)
    
    # Create a clock object to control the game's frame rate
    clock = pygame.time.Clock()
    
    # Game loop control flag
    running = True
    
    # Main game loop
    while running:
        # Limit the game to 30 frames per second
        clock.tick(30)

        # Process all events in the event queue
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Exit the game loop if the user closes the window
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Make the bird jump when the spacebar is pressed
                    env.bird.jump()

        # Perform a game step with no additional action (action=0)
        _, _, done = env.step(action=0)
        
        # Check if the game is over
        if done:
            print(f"Game Over! Final Score: {env.score}")
            # Reset the game environment for a new round
            env.reset()

        # Render the current state of the game
        env.render()

    # Clean up resources and close the environment after exiting the loop
    env.close()
