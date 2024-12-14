import json
import random
from genetic_navigation import GeneticNavigator
from rssi_grid_manager import RSSIGridManager
from create_llm_dataset import create_training_dataset

def get_random_spawn_point(grid_size=11):
    return [random.randint(0, grid_size-1), random.randint(0, grid_size-1)]

def create_datasets_for_variations(num_spawn_points=10):
    # Initialize the RSSI grid manager
    grid_manager = RSSIGridManager()
    
    # Process each grid variation
    for variation_name, grid in grid_manager.grids.items():
        print(f"\nProcessing {variation_name} variation...")
        all_training_data = []
        
        # Generate data for multiple spawn points
        for spawn_idx in range(num_spawn_points):
            print(f"Processing spawn point {spawn_idx + 1}/{num_spawn_points}...")
            
            # Get a random spawn point that's not a target (RSSI = 0)
            while True:
                spawn_point = get_random_spawn_point()
                if grid[spawn_point[0]][spawn_point[1]] != 0.0:
                    break
            
            print(f"Spawn point selected: {spawn_point}")
            
            # Initialize the navigator with the current grid variation and spawn point
            navigator = GeneticNavigator(population_size=100, gene_length=50)
            navigator.rssi_values = grid
            navigator.start_position = spawn_point
            
            # Evolve to get the best robots for this spawn point
            navigator.current_population = navigator.initialize_population()
            navigator.evolve(generations=300)
            
            # Get the top 12 robots by fitness
            top_robots = sorted(navigator.current_population, key=lambda x: x.fitness, reverse=True)[:12]
            
            # Create the training dataset using all top robots
            spawn_data = create_training_dataset(navigator, top_robots)
            all_training_data.extend(spawn_data)
        
        # Write to a variation-specific file in JSONL format
        filename = f'llm_training_data_gen_500_{variation_name}.jsonl'
        with open(filename, 'w') as f:
            for example in all_training_data:
                f.write(json.dumps(example) + '\n')
        
        print(f"Created dataset for {variation_name} variation: {filename}")

if __name__ == "__main__":
    create_datasets_for_variations()
