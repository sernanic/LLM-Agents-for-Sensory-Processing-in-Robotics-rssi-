import pygame
import sys
# import numpy as np
import math
import json
# from rssi_navigation import RSSINavigator
from genetic_navigation import GeneticNavigator, Robot
from create_llm_dataset import create_training_dataset
# import time
import random

# Initialize Pygame
pygame.init()
pygame.font.init()

def get_target_generation():
    # Create a small window for input
    input_screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("Enter Target Generation")
    font = pygame.font.Font(None, 36)
    input_text = ""
    input_rect = pygame.Rect(50, 80, 300, 40)
    active = True
    
    while active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    try:
                        value = int(input_text)
                        if value > 0:
                            return value
                    except ValueError:
                        input_text = ""
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                else:
                    if event.unicode.isnumeric():
                        input_text += event.unicode
        
        input_screen.fill((255, 255, 255))
        # Draw input box
        pygame.draw.rect(input_screen, (0, 0, 0), input_rect, 2)
        # Render prompt text
        prompt_surface = font.render("Enter target generation:", True, (0, 0, 0))
        input_screen.blit(prompt_surface, (50, 40))
        # Render input text
        text_surface = font.render(input_text, True, (0, 0, 0))
        input_screen.blit(text_surface, (input_rect.x + 5, input_rect.y + 5))
        pygame.display.flip()

def get_spawn_points_count():
    # Create a small window for input
    input_screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("Enter Number of Spawn Points")
    font = pygame.font.Font(None, 36)
    input_text = ""
    input_rect = pygame.Rect(50, 80, 300, 40)
    active = True
    
    while active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    try:
                        value = int(input_text)
                        if value > 0:
                            return value
                    except ValueError:
                        input_text = ""
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                else:
                    if event.unicode.isnumeric():
                        input_text += event.unicode
        
        input_screen.fill((255, 255, 255))
        # Draw input box
        pygame.draw.rect(input_screen, (0, 0, 0), input_rect, 2)
        # Render prompt text
        prompt_surface = font.render("Enter number of spawn points:", True, (0, 0, 0))
        input_screen.blit(prompt_surface, (50, 40))
        # Render input text
        text_surface = font.render(input_text, True, (0, 0, 0))
        input_screen.blit(text_surface, (input_rect.x + 5, input_rect.y + 5))
        pygame.display.flip()

def ask_view_best_robot():
    # Create a small window for input
    input_screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("View Best Robot?")
    font = pygame.font.Font(None, 36)
    active = True
    
    while active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_y:
                    return True
                elif event.key == pygame.K_n:
                    return False
        
        input_screen.fill((255, 255, 255))
        # Render prompt text
        prompt_surface = font.render("View best robot? (y/n)", True, (0, 0, 0))
        input_screen.blit(prompt_surface, (50, 80))
        pygame.display.flip()

# Constants
CELL_SIZE = 50
GRID_SIZE = 11
INFO_PANEL_WIDTH = 300
WINDOW_HEIGHT = CELL_SIZE * GRID_SIZE
GRID_WIDTH = CELL_SIZE * GRID_SIZE
TOTAL_WIDTH = GRID_WIDTH + INFO_PANEL_WIDTH
MOVEMENT_DELAY = 500  # Delay between movements in milliseconds

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
PURPLE = (128, 0, 128)  # Color for genetic algorithm robots

# RSSI values
rssi_values = [
    [-8.738930154245397, -7.325638626280702, -6.186205103212336, -5.728630197605584, -6.186205103212336, -7.325638626280702, -8.738930154245397, -10.16560518993271, -11.500994273634886, -12.718330240965772, -13.820485038841705],
    [-7.325638626280702, -5.21710497313177, -3.1759051465725197, -2.2068050164919573, -3.1759051465725197, -5.21710497313177, -7.325638626280702, -9.196505059852145, -10.810185082201892, -12.20680501649196, -13.428963799220222],
    [-6.186205103212336, -3.1759051465725197, 0.0, 0.0, 0.0, -3.1759051465725197, -6.186205103212336, -8.490694316995073, -10.335938582920514, -11.868222343882286, -13.175905146572523],
    [-5.728630197605584, -2.2068050164919573, 0.0, 0.0, 0.0, -2.2068050164919573, -5.728630197605584, -8.227404929771582, -10.16560518993271, -11.74923011088521, -13.08816590349747],
    [-6.186205103212336, -3.1759051465725197, 0.0, 0.0, 0.0, -3.1759051465725197, -6.186205103212336, -8.490694316995073, -10.335938582920514, -11.868222343882286, -13.175905146572523],
    [-7.325638626280702, -5.21710497313177, -3.1759051465725197, -2.2068050164919573, -3.1759051465725197, -5.21710497313177, -7.325638626280702, -9.196505059852145, -10.810185082201892, -12.20680501649196, -13.428963799220222],
    [-8.738930154245397, -7.325638626280702, -6.186205103212336, -5.728630197605584, -6.186205103212336, -7.325638626280702, -8.738930154245397, -10.16560518993271, -11.500994273634886, -12.718330240965772, -13.820485038841705],
    [-10.16560518993271, -9.196505059852145, -8.490694316995073, -8.227404929771582, -8.490694316995073, -9.196505059852145, -10.16560518993271, -11.237704886411395, -12.31404367040969, -13.346238539560327, -14.315338669640889],
    [-11.500994273634886, -10.810185082201892, -10.335938582920514, -10.16560518993271, -10.335938582920514, -10.810185082201892, -11.500994273634886, -12.31404367040969, -13.175905146572523, -14.039503453320007, -14.878522300522095],
    [-12.718330240965772, -12.20680501649196, -11.868222343882286, -11.74923011088521, -11.868222343882286, -12.20680501649196, -12.718330240965772, -13.346238539560327, -14.039503453320007, -14.759530067525015, -15.48039436035526],
    [-13.820485038841705, -13.428963799220222, -13.175905146572523, -13.08816590349747, -13.175905146572523, -13.428963799220222, -13.820485038841705, -14.315338669640889, -14.878522300522095, -15.48039436035526, -16.09846586013728],
]

# Get spawn points count from user
spawn_points_count = get_spawn_points_count()

# Get target generation
target_generation = get_target_generation()

# Ask if user wants to see best robot
view_best = ask_view_best_robot()

# Generate random spawn points (excluding target positions where RSSI = 0)
spawn_points = []
while len(spawn_points) < spawn_points_count:
    x = random.randint(0, GRID_SIZE - 1)
    y = random.randint(0, GRID_SIZE - 1)
    pos = [x, y]
    # Check if position is valid (not a target position)
    if pos not in spawn_points and rssi_values[x][y] != 0.0:
        spawn_points.append(pos)

print(f"Generated {spawn_points_count} spawn points: {spawn_points}")

# Process each spawn point
all_training_data = []
for spawn_index, spawn_pos in enumerate(spawn_points):
    print(f"\nProcessing spawn point {spawn_index + 1}/{spawn_points_count}: {spawn_pos}")
    
    # Create dataset at target generation for this spawn point
    print(f"Creating dataset for spawn point {spawn_index + 1}...")
    dataset_navigator = GeneticNavigator(population_size=100, gene_length=50)
    dataset_navigator.start_position = spawn_pos
    dataset_navigator.max_generations = target_generation
    
    # Evolve for dataset creation
    dataset_robots = dataset_navigator.initialize_population()
    dataset_navigator.current_population = dataset_robots
    for _ in range(target_generation):
        dataset_navigator.evolve()
    
    # Get top 12 robots for dataset
    top_robots = sorted(dataset_navigator.current_population, key=lambda x: x.fitness, reverse=True)[:12]
    spawn_data = create_training_dataset(dataset_navigator, top_robots)
    all_training_data.extend(spawn_data)
    print(f"Data collected for spawn point {spawn_index + 1}")

# Write all data to a single file
filename = f'llm_training_data_gen_{target_generation}.jsonl'
with open(filename, 'w') as f:
    for example in all_training_data:
        f.write(json.dumps(example) + '\n')
print(f"\nAll spawn points processed! Dataset has been created: {filename}")

if view_best:
    # Initialize a new genetic navigator for visualization
    genetic_navigator = GeneticNavigator(population_size=100, gene_length=50)
    genetic_navigator.start_position = spawn_points[0]
    genetic_navigator.max_generations = target_generation
    genetic_robots = genetic_navigator.initialize_population()
    genetic_navigator.current_population = genetic_robots
    
    # Evolve to target generation
    best_robot = None
    for _ in range(target_generation):
        genetic_navigator.evolve()
    best_robot = genetic_navigator.best_robot
    print(f"Best robot fitness at generation {target_generation}: {best_robot.fitness}")
    
    # Reset positions for visualization
    genetic_navigator.reset_robot_positions()
    
    # Set up display for visualization
    screen = pygame.display.set_mode((TOTAL_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(f"RSSI Navigation - Spawn 1/{spawn_points_count} - Gen {target_generation}")
    
    def get_color(rssi_value):
        min_rssi = min(min(row) for row in rssi_values)
        max_rssi = max(max(row) for row in rssi_values)
        normalized = (rssi_value - min_rssi) / (max_rssi - min_rssi)
        return (0, 0, int(255 * normalized))

    def get_orientation_from_action(action: str) -> int:
        """Convert navigation action to orientation angle."""
        return {
            'left': 180,
            'right': 0,
            'straight': 270,
            'back': 90
        }.get(action, 0)

    def move_robot(action: str) -> bool:
        """Move the robot based on the selected action."""
        global robot_pos, robot_orientation
        
        # Get new position
        x, y = robot_pos
        
        # Check if we're already at the target
        if rssi_values[x][y] == 0:
            return False
        
        if action == 'left' and y > 0:
            robot_pos[1] -= 1
            robot_orientation = 180
            return True
        elif action == 'right' and y < GRID_SIZE - 1:
            robot_pos[1] += 1
            robot_orientation = 0
            return True
        elif action == 'straight' and x > 0:
            robot_pos[0] -= 1
            robot_orientation = 270
            return True
        elif action == 'back' and x < GRID_SIZE - 1:
            robot_pos[0] += 1
            robot_orientation = 90
            return True
        return False

    def draw_robot(screen, pos, orientation):
        robot_x = pos[1] * CELL_SIZE + (CELL_SIZE - 50) // 2
        robot_y = pos[0] * CELL_SIZE + (CELL_SIZE - 50) // 2
        
        # Draw robot body
        pygame.draw.rect(screen, RED, (robot_x, robot_y, 50, 50))
        
        # Draw direction indicator
        center_x = robot_x + 50 // 2
        center_y = robot_y + 50 // 2
        indicator_length = 50 // 2
        
        angle = math.radians(orientation)
        point1 = (center_x + indicator_length * math.cos(angle),
                  center_y + indicator_length * math.sin(angle))
        point2 = (center_x + indicator_length * math.cos(angle + 2.6),
                  center_y + indicator_length * math.sin(angle + 2.6))
        point3 = (center_x + indicator_length * math.cos(angle - 2.6),
                  center_y + indicator_length * math.sin(angle - 2.6))
        
        pygame.draw.polygon(screen, YELLOW, [point1, point2, point3])

    def draw_genetic_robots(screen):
        """Draw all robots from the genetic algorithm."""
        for robot in genetic_navigator.current_population:
            robot_x = robot.position[1] * CELL_SIZE + (CELL_SIZE - 50) // 2
            robot_y = robot.position[0] * CELL_SIZE + (CELL_SIZE - 50) // 2
            pygame.draw.rect(screen, PURPLE, (robot_x, robot_y, 50, 50))

    def update_genetic_robots():
        global running, genetic_navigator, target_generation
        
        # Check if we've reached max generations
        if genetic_navigator.generation > target_generation:
            running = False
            print(f"Visualization complete for generation {target_generation}")
            return
        
        # Move all robots one step
        genetic_navigator.update_robot_positions()
        
        # Check if all robots have finished their current sequence
        all_finished = all(robot.current_gene_index >= len(robot.genes) or 
                          robot.steps_taken >= 25 or 
                          robot.reached_target 
                          for robot in genetic_navigator.current_population)
        
        if all_finished:
            genetic_navigator.reset_robot_positions()  # Reset positions for new visualization

    def draw_info_panel(screen, robot_pos, path_history, rssi_values, action, confidence):
        """Draw information panel with navigation details."""
        panel_x = GRID_WIDTH
        font = pygame.font.SysFont('Arial', 16)
        y_offset = 10

        # Clear panel area
        pygame.draw.rect(screen, WHITE, (panel_x, 0, INFO_PANEL_WIDTH, WINDOW_HEIGHT))
        pygame.draw.line(screen, BLACK, (panel_x, 0), (panel_x, WINDOW_HEIGHT))
        
        # Display robot information
        texts = [
            f"Robot Position: ({robot_pos[0]}, {robot_pos[1]})",
            f"Current RSSI: {rssi_values[robot_pos[0]][robot_pos[1]]:.2f}",
            f"Last Action: {action}",
            f"Confidence: {confidence:.2f}",
            "",
            "Genetic Algorithm Stats:",
            f"Generation: {genetic_navigator.generation}",
            f"Population Size: {len(genetic_navigator.current_population)}"
        ]
        
        if genetic_navigator.best_robot:
            texts.extend([
                f"Best Fitness: {genetic_navigator.best_robot.fitness:.4f}",
                f"Best Steps: {genetic_navigator.best_robot.steps_taken}",
                f"Reached Target: {genetic_navigator.best_robot.reached_target}",
                f"Robots at Target: {sum(1 for robot in genetic_navigator.current_population if robot.reached_target)}"
            ])

        for text in texts:
            text_surface = font.render(text, True, BLACK)
            screen.blit(text_surface, (panel_x + 10, y_offset))
            y_offset += 25

    # Global variables for robot state
    robot_pos = None
    robot_orientation = 0

    # Main visualization loop for this spawn point
    running = True
    last_move_time = pygame.time.get_ticks()
    action_index = 0
    
    # Initialize robot position at spawn point
    robot_pos = list(spawn_points[0])
    robot_orientation = 0

    if view_best and best_robot:
        # Trim the robot's genes to only include actions up to reaching target
        test_pos = list(spawn_points[0])
        trimmed_genes = []
        position_history = [test_pos.copy()]
        
        for gene in best_robot.genes:
            trimmed_genes.append(gene)
            # Simulate the move
            new_pos = test_pos.copy()
            if gene == 'left' and new_pos[1] > 0:
                new_pos[1] -= 1
            elif gene == 'right' and new_pos[1] < GRID_SIZE - 1:
                new_pos[1] += 1
            elif gene == 'straight' and new_pos[0] > 0:
                new_pos[0] -= 1
            elif gene == 'back' and new_pos[0] < GRID_SIZE - 1:
                new_pos[0] += 1
            
            # Check if we're stuck
            position_history.append(new_pos.copy())
            if len(position_history) >= 10:
                last_positions = position_history[-10:]
                position_set = set(tuple(pos) for pos in last_positions)
                if len(position_set) <= 3:  # Robot is revisiting same 2-3 positions
                    print("Robot is stuck in a loop!")
                    break
            
            test_pos = new_pos
            # Check if we've reached target
            if rssi_values[test_pos[0]][test_pos[1]] == 0:
                break
        
        robot_actions = trimmed_genes
        # Reset best robot to spawn position
        best_robot.position = list(spawn_points[0])
        best_robot.position_history = []
    else:
        robot_actions = None

    while running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:  # Add space to skip to next spawn point
                    running = False
        
        # Move robots
        if current_time - last_move_time >= MOVEMENT_DELAY:
            if view_best and best_robot:
                if action_index < len(robot_actions):
                    action = robot_actions[action_index]
                    if move_robot(action):  # Only increment if move was successful
                        best_robot.position = list(robot_pos)  # Update best robot position
                        best_robot.position_history.append(best_robot.position.copy())
                        action_index += 1
                        
                        # Check if we're stuck
                        if len(best_robot.position_history) >= 10:
                            last_positions = best_robot.position_history[-10:]
                            position_set = set(tuple(pos) for pos in last_positions)
                            if len(position_set) <= 3:  # Robot is revisiting same 2-3 positions
                                print("Robot is stuck in a loop!")
                                pygame.time.wait(1000)  # Wait a second to show stuck message
                                running = False
                        # Check if we've reached the target
                        elif rssi_values[robot_pos[0]][robot_pos[1]] == 0:
                            print("Target reached!")
                            pygame.time.wait(1000)  # Wait a second to show we reached the target
                            running = False
                else:
                    running = False
            else:
                update_genetic_robots()
            last_move_time = current_time
        
        # Clear screen
        screen.fill(WHITE)
        
        # Draw grid and RSSI values
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                cell_color = get_color(rssi_values[i][j])
                pygame.draw.rect(screen, cell_color, 
                               (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE), 0)
                pygame.draw.rect(screen, GRAY, 
                               (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
        
        # Draw current spawn point marker
        spawn_x = spawn_points[0][1] * CELL_SIZE + CELL_SIZE // 2
        spawn_y = spawn_points[0][0] * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, RED, (spawn_x, spawn_y), 8)
        
        # Draw genetic algorithm robots
        draw_genetic_robots(screen)
        
        # Draw robot
        draw_robot(screen, robot_pos, robot_orientation)
        
        # Draw information panel
        draw_info_panel(screen, robot_pos, [], rssi_values, "", 0.0)
        
        # Update display
        pygame.display.flip()
        
        # Add small delay to control simulation speed
        pygame.time.delay(50)

pygame.quit()
sys.exit()
