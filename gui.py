from environment import SoccerEnvironment, CatchEnvironment
import pygame
import threading
import time

class SoccerGui():
    def __init__(self,env: SoccerEnvironment,mac):
        self.env = env
        self.mac = mac
        pass

    def run(self):
        if self.mac:
            self._main_loop()
        else:
            """
            Starts the GUI in a separate thread.
            """
            self.running = True
            threading.Thread(target=self._main_loop, daemon=True).start()


    def _main_loop(self):
        """
        GUI loop.
        """
        pygame.init()
        # Set up the display
        screen_width, screen_height = 600, 400   # Keep this ratio so players stay a a circle
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Simulation GUI")
        clock = pygame.time.Clock()
        # Main loop
        running = True
        # Initialize font
        pygame.font.init()
        font = pygame.font.Font(None, 36)  # Default font with size 36
        if self.mac:
            # Record the start time
            start_time = time.time()
        
        while running:

            # Check if 1 second has passed
            if self.mac and time.time() - start_time > 1:
                running = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                ## actions if we are using the mock value function and policy
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        self.env.mock_actions[1] = "move_up"
                    if event.key == pygame.K_s:
                        self.env.mock_actions[1] = "move_down"
                    if event.key == pygame.K_a:
                        self.env.mock_actions[1] = "move_left"
                    if event.key == pygame.K_d:
                        self.env.mock_actions[1] = "move_right"
                    if event.key == pygame.K_UP:
                        self.env.mock_actions[0] = "move_up"
                    if event.key == pygame.K_DOWN:
                        self.env.mock_actions[0] = "move_down"
                    if event.key == pygame.K_LEFT:
                        self.env.mock_actions[0] = "move_left"
                    if event.key == pygame.K_RIGHT:
                        self.env.mock_actions[0] = "move_right"

            # Clear the screen
            screen.fill((255, 255, 255))  # White background
            # Draw the environment
            field_x, field_y = screen_width // 12, 0
            square_width, square_height = screen_width // 6, screen_height // 4
            field_color = (81, 219, 132)
            for row in range(4):
                for col in range(5):
                    rect = pygame.Rect(field_x + col * square_width, field_y + row * square_height, square_width, square_height)
                    pygame.draw.rect(screen, field_color, rect, 1)
            # print(self.env.state)
            playerA_color = (0, 0, 255)
            playerB_color = (0, 0, 255)
            if self.env.state[2] == 0:
                playerA_color = (255, 0, 0)  # Red for player A with the ball
            elif self.env.state[2] == 1:
                playerB_color = (255, 0, 0)
            playerA_pos = self.env.state[0]
            playerB_pos = self.env.state[1]
            ##
            playerA_center = (
                field_x + playerA_pos[0] * square_width + square_width // 2,
                screen_height - playerA_pos[1] * square_height - square_height // 2
            )
            playerB_center = (
                field_x + playerB_pos[0] * square_width + square_width // 2,
                screen_height - playerB_pos[1] * square_height - square_height // 2
            )
            ##
            pygame.draw.circle(screen, playerA_color, playerA_center, square_width // 3)
            pygame.draw.circle(screen, playerB_color, playerB_center, square_width // 3)

            # Render letters
            textA = font.render("A", True, (255, 255, 255))  # White text for Player A
            textB = font.render("B", True, (255, 255, 255))  # White text for Player B

            # Center the text on the circles
            textA_rect = textA.get_rect(center=playerA_center)
            textB_rect = textB.get_rect(center=playerB_center)

            # Blit the text onto the screen
            screen.blit(textA, textA_rect)
            screen.blit(textB, textB_rect)


            pygame.display.flip()
            clock.tick(60)  # Limit to 60 FPS

        pygame.quit()

class CatchGui():
    def __init__(self,env: CatchEnvironment,mac):
        self.env = env
        self.mac = mac
        pass

    def run(self):
        if self.mac:
            self._main_loop()
        else:
            """
            Starts the GUI in a separate thread.
            """
            self.running = True
            threading.Thread(target=self._main_loop, daemon=True).start()


    def _main_loop(self):
        """
        GUI loop.
        """
        pygame.init()
        # Set up the display
        screen_width, screen_height = 600, 400   # Keep this ratio so players stay a a circle
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Simulation GUI")
        clock = pygame.time.Clock()
        # Main loop
        running = True
        # Initialize font
        pygame.font.init()
        font = pygame.font.Font(None, 36)  # Default font with size 36
        if self.mac:
            # Record the start time
            start_time = time.time()
        
        while running:

            # Check if 1 second has passed
            if self.mac and time.time() - start_time > 1:
                running = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                ## actions if we are using the mock value function and policy
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        self.env.mock_actions[1] = "move_up"
                    if event.key == pygame.K_s:
                        self.env.mock_actions[1] = "move_down"
                    if event.key == pygame.K_a:
                        self.env.mock_actions[1] = "move_left"
                    if event.key == pygame.K_d:
                        self.env.mock_actions[1] = "move_right"
                    if event.key == pygame.K_UP:
                        self.env.mock_actions[0] = "move_up"
                    if event.key == pygame.K_DOWN:
                        self.env.mock_actions[0] = "move_down"
                    if event.key == pygame.K_LEFT:
                        self.env.mock_actions[0] = "move_left"
                    if event.key == pygame.K_RIGHT:
                        self.env.mock_actions[0] = "move_right"

            # Clear the screen
            screen.fill((255, 255, 255))  # White background
            # Draw the environment
            field_x, field_y = screen_width // 9, 0
            square_width, square_height = screen_width // 6, screen_height // 4
            field_color = (81, 219, 132)
            for row in range(4):
                for col in range(4):
                    rect = pygame.Rect(field_x + col * square_width, field_y + row * square_height, square_width, square_height)
                    pygame.draw.rect(screen, field_color, rect, 1)
            # print(self.env.state)
            playerA_color = (0, 0, 255)
            playerB_color = (0, 0, 255)
            if self.env.state[2] == 0:
                playerA_color = (255, 0, 0)  # Red for player A with the ball
            elif self.env.state[2] == 1:
                playerB_color = (255, 0, 0)
            playerA_pos = self.env.state[0]
            playerB_pos = self.env.state[1]
            ##
            playerA_center = (
                field_x + playerA_pos[0] * square_width + square_width // 2,
                screen_height - playerA_pos[1] * square_height - square_height // 2
            )
            playerB_center = (
                field_x + playerB_pos[0] * square_width + square_width // 2,
                screen_height - playerB_pos[1] * square_height - square_height // 2
            )
            ##
            pygame.draw.circle(screen, playerA_color, playerA_center, square_width // 3)
            pygame.draw.circle(screen, playerB_color, playerB_center, square_width // 3)

            # Render letters
            textA = font.render("A", True, (255, 255, 255))  # White text for Player A
            textB = font.render("B", True, (255, 255, 255))  # White text for Player B

            # Center the text on the circles
            textA_rect = textA.get_rect(center=playerA_center)
            textB_rect = textB.get_rect(center=playerB_center)

            # Blit the text onto the screen
            screen.blit(textA, textA_rect)
            screen.blit(textB, textB_rect)


            pygame.display.flip()
            clock.tick(60)  # Limit to 60 FPS

        pygame.quit()


    

    
