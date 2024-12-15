import json
import random
from genetic_navigation import GeneticNavigator
from rssi_grid_manager import RSSIGridManager
from create_llm_dataset import create_training_dataset

def get_random_spawn_point(grid_size=11):
    return [random.randint(0, grid_size-1), random.randint(0, grid_size-1)]

def create_datasets_for_variations(num_spawn_points=10, generations=300):
    # Initialize the RSSI grid manager
    grid_manager = RSSIGridManager()
    
    # Create a single file for all training data
    all_variations_data = []
    
    # Process each grid variation
    for variation_name, grid in grid_manager.grids.items():
        print(f"\nProcessing {variation_name} variation...")
        
        # Keep track of used spawn points for this variation
        used_spawn_points = set()
        
        # Generate data for multiple spawn points
        for spawn_idx in range(num_spawn_points):
            print(f"Processing spawn point {spawn_idx + 1}/{num_spawn_points}...")
            
            # Get a random spawn point that's not a target (RSSI = 0) and hasn't been used
            while True:
                spawn_point = get_random_spawn_point()
                spawn_point_tuple = tuple(spawn_point)  # Convert to tuple for set membership
                if grid[spawn_point[0]][spawn_point[1]] != 0.0 and spawn_point_tuple not in used_spawn_points:
                    used_spawn_points.add(spawn_point_tuple)
                    break
            
            print(f"Spawn point selected: {spawn_point}")
            
            # Initialize the navigator with the current grid variation and spawn point
            navigator = GeneticNavigator(population_size=100, gene_length=50)
            navigator.rssi_values = grid
            navigator.start_position = spawn_point
            
            # Evolve to get the best robots for this spawn point
            navigator.current_population = navigator.initialize_population()
            navigator.evolve(generations=generations)
            
            # Get the top 12 robots by fitness
            top_robots = sorted(navigator.current_population, key=lambda x: x.fitness, reverse=True)[:12]
            
            # Create the training dataset using all top robots
            spawn_data = create_training_dataset(navigator, top_robots)
            
            # Add variation information to each data point
            for data_point in spawn_data:
                data_point['variation'] = variation_name
            
            all_variations_data.extend(spawn_data)
        
        print(f"Completed processing {variation_name} variation")
    
    # Write all data to a single file
    filename = f'llm_training_data_gen_{generations}_all_variations.jsonl'
    with open(filename, 'w') as f:
        for example in all_variations_data:
            f.write(json.dumps(example) + '\n')
    
    print(f"\nCreated combined dataset with all variations: {filename}")

if __name__ == "__main__":
    create_datasets_for_variations()
