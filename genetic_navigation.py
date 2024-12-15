import numpy as np
from typing import List, Tuple, Optional
import random
from dataclasses import dataclass
from rssi_navigation import RSSINavigator

@dataclass
class Robot:
    position: List[int]
    genes: List[str]  # Sequence of actions
    fitness: float = 0.0
    steps_taken: int = 0
    reached_target: bool = False
    current_gene_index: int = 0  # Track current action being executed
    position_history: List[List[int]] = None  # Track visited positions

    def __post_init__(self):
        self.position_history = []

    def is_stuck(self) -> bool:
        """Check if robot is stuck in a loop or not making progress"""
        if len(self.position_history) < 10:  # Need some history to detect loops
            return False
            
        # Check last 10 positions for repeating patterns
        last_positions = self.position_history[-10:]
        position_set = set(tuple(pos) for pos in last_positions)
        if len(position_set) <= 3:  # Robot is revisiting same 2-3 positions
            return True
            
        return False

class GeneticNavigator:
    def __init__(self, population_size: int = 100, gene_length: int = 50):
        self.population_size = population_size
        self.gene_length = gene_length
        self.navigator = RSSINavigator()
        self.actions = ['left', 'right', 'straight', 'back']
        self.grid_size = 11
        self.target_rssi = 0.0
        self.current_population = []  # Add current population storage
        self.start_position = [self.grid_size - 1, self.grid_size - 1]
        self.movement_step = 0  # Track current movement step
        self.generation = 0  # Add generation counter
        self.best_robot = None  # Track best robot
        self.max_generations = 300  # Maximum number of generations
        # Initialize RSSI values
        self.rssi_values = [
            [-8.73, -7.32, -6.18, -5.72, -6.18, -7.32, -8.73, -10.16, -11.50, -12.71, -13.82],
            [-7.32, -5.21, -3.17, -2.20, -3.17, -5.21, -7.32, -9.19, -10.81, -12.20, -13.42],
            [-6.18, -3.17, 0.0, 0.0, 0.0, -3.17, -6.18, -8.49, -10.33, -11.86, -13.17],
            [-5.72, -2.20, 0.0, 0.0, 0.0, -2.20, -5.72, -8.22, -10.16, -11.74, -13.08],
            [-6.18, -3.17, 0.0, 0.0, 0.0, -3.17, -6.18, -8.49, -10.33, -11.86, -13.17],
            [-7.32, -5.21, -3.17, -2.20, -3.17, -5.21, -7.32, -9.19, -10.81, -12.20, -13.42],
            [-8.73, -7.32, -6.18, -5.72, -6.18, -7.32, -8.73, -10.16, -11.50, -12.71, -13.82],
            [-10.16, -9.19, -8.49, -8.22, -8.49, -9.19, -10.16, -11.23, -12.31, -13.34, -14.31],
            [-11.50, -10.81, -10.33, -10.16, -10.33, -10.81, -11.50, -12.31, -13.17, -14.03, -14.87],
            [-12.71, -12.20, -11.86, -11.74, -11.86, -12.20, -12.71, -13.34, -14.03, -14.75, -15.48],
            [-13.82, -13.42, -13.17, -13.08, -13.17, -13.42, -13.82, -14.31, -14.87, -15.48, -16.09]
        ]

    def generate_random_position(self) -> List[int]:
        """Generate a random position on the grid, excluding target positions."""
        while True:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            if self.rssi_values[x][y] != 0.0:
                return [x, y]

    def initialize_population(self) -> List[Robot]:
        """Create initial population with random genes."""
        population = []
        for _ in range(self.population_size):
            genes = random.choices(self.actions, k=self.gene_length)
            population.append(Robot(
                position=self.start_position.copy(),
                genes=genes,
                current_gene_index=0
            ))
        return population

    def is_valid_move(self, pos: List[int], action: str) -> bool:
        """Check if a move is valid within the grid."""
        x, y = pos
        if action == 'left' and y > 0:
            return True
        elif action == 'right' and y < self.grid_size - 1:
            return True
        elif action == 'straight' and x > 0:
            return True
        elif action == 'back' and x < self.grid_size - 1:
            return True
        return False

    def apply_move(self, pos: List[int], action: str) -> List[int]:
        """Apply the move and return new position."""
        new_pos = pos.copy()
        if action == 'left':
            new_pos[1] -= 1
        elif action == 'right':
            new_pos[1] += 1
        elif action == 'straight':
            new_pos[0] -= 1
        elif action == 'back':
            new_pos[0] += 1
        return new_pos

    def update_robot_positions(self) -> None:
        """Move all robots one step according to their genes."""
        for robot in self.current_population:
            # Skip if robot has already reached target or exceeded steps/genes
            if robot.reached_target or robot.current_gene_index >= len(robot.genes) or robot.steps_taken >= 25:
                continue

            action = robot.genes[robot.current_gene_index]
            if self.is_valid_move(robot.position, action):
                new_position = self.apply_move(robot.position, action)
                
                # Check if position has been visited before
                if any(new_position == pos for pos in robot.position_history):
                    robot.current_gene_index += 1
                    continue
                    
                robot.position = new_position
                robot.position_history.append(robot.position.copy())
                robot.steps_taken += 1
                
                # Check if target reached
                current_rssi = self.rssi_values[robot.position[0]][robot.position[1]]
                if current_rssi == 0.0:
                    robot.reached_target = True
                    robot.fitness = 1.0 / robot.steps_taken
                    continue

            robot.current_gene_index += 1

    def reset_robot_positions(self) -> None:
        """Reset all robots to starting position for new generation."""
        for robot in self.current_population:
            robot.position = self.start_position.copy()
            robot.current_gene_index = 0
            robot.steps_taken = 0
            robot.reached_target = False
            robot.fitness = 0.0
            robot.position_history = []

    def calculate_fitness(self, robot: Robot) -> float:
        """Calculate fitness based on path length and final RSSI value."""
        current_pos = robot.position.copy()
        steps = 0
        robot.position_history = [current_pos.copy()]  # Reset position history
        
        for action in robot.genes:
            if steps >= 25 or robot.is_stuck():  # Kill robots that take too many steps or are stuck
                robot.fitness = 0
                robot.steps_taken = steps
                return 0
                
            if not self.is_valid_move(current_pos, action):
                continue
                
            new_pos = self.apply_move(current_pos, action)
            
            # Skip if position has been visited before
            if any(new_pos == pos for pos in robot.position_history):
                continue
                
            current_pos = new_pos
            robot.position_history.append(current_pos.copy())
            steps += 1
            
            current_rssi = self.rssi_values[current_pos[0]][current_pos[1]]
            if current_rssi == 0.0:  # Target reached
                robot.reached_target = True
                robot.steps_taken = steps
                # Fitness is inversely proportional to steps taken
                robot.fitness = 1.0 / steps
                return robot.fitness
        
        # If target not reached, fitness is based on final RSSI value
        final_rssi = self.rssi_values[current_pos[0]][current_pos[1]]
        robot.steps_taken = steps
        robot.fitness = 1.0 / (abs(final_rssi) + 1)  # Closer to 0.0 is better
        return robot.fitness

    def select_parents(self, population: List[Robot], num_parents: int) -> List[Robot]:
        """Select parents using tournament selection."""
        parents = []
        for _ in range(num_parents):
            tournament = random.sample(population, k=5)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        return parents

    def crossover(self, parent1: Robot, parent2: Robot) -> Tuple[List[str], List[str]]:
        """Perform crossover between two parents."""
        crossover_point = random.randint(0, len(parent1.genes) - 1)
        child1_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
        child2_genes = parent2.genes[:crossover_point] + parent1.genes[crossover_point:]
        return child1_genes, child2_genes

    def mutate(self, genes: List[str], mutation_rate: float = 0.1) -> List[str]:
        """Randomly mutate genes."""
        mutated_genes = genes.copy()
        for i in range(len(mutated_genes)):
            if random.random() < mutation_rate:
                mutated_genes[i] = random.choice(self.actions)
        return mutated_genes

    def evolve(self, generations: int = 1) -> Tuple[Robot, List[float]]:
        """Evolve the population for a specified number of generations."""
        fitness_history = []
        best_robot = None
        
        for _ in range(generations):
            # Calculate fitness for all robots
            for robot in self.current_population:
                if robot.fitness == 0:  # Only calculate if not already calculated
                    robot.fitness = self.calculate_fitness(robot)
            
            # Find best robot
            self.best_robot = max(self.current_population, key=lambda x: x.fitness)
            fitness_history.append(self.best_robot.fitness)
            
            # Create new population through selection and crossover
            new_population = []
            while len(new_population) < self.population_size:
                parent1 = self.select_parents(self.current_population, 1)[0]
                parent2 = self.select_parents(self.current_population, 1)[0]
                child1_genes, child2_genes = self.crossover(parent1, parent2)
                child1_genes = self.mutate(child1_genes)
                child2_genes = self.mutate(child2_genes)
                new_population.extend([
                    Robot(position=self.start_position.copy(), genes=child1_genes, current_gene_index=0),
                    Robot(position=self.start_position.copy(), genes=child2_genes, current_gene_index=0)
                ])
            
            # Ensure population size stays constant
            new_population = new_population[:self.population_size]
            
            self.current_population = new_population
            self.generation += 1  # Increment generation counter
        
        return self.best_robot, fitness_history

def main():
    # Initialize and run genetic algorithm
    navigator = GeneticNavigator(population_size=100, gene_length=50)
    navigator.current_population = navigator.initialize_population()
    best_robot, fitness_history = navigator.evolve(generations=50)
    
    print(f"\nBest Robot Results:")
    print(f"Starting Position: {best_robot.position}")
    print(f"Steps Taken: {best_robot.steps_taken}")
    print(f"Reached Target: {best_robot.reached_target}")
    print(f"Fitness Score: {best_robot.fitness:.4f}")
    print(f"Optimal Path: {best_robot.genes[:best_robot.steps_taken]}")

if __name__ == "__main__":
    main()
