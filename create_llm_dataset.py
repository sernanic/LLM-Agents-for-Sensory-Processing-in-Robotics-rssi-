import json
from genetic_navigation import GeneticNavigator
from collections import deque

def create_training_dataset(navigator, robots):
    training_data = []
    
    for i, robot in enumerate(robots):
        # Add a separator comment for each robot
        training_data.append({
            "comment": f"### Robot {i+1} - Fitness: {robot.fitness:.4f} ###"
        })
        
        rssi_window = deque([-100] * 5, maxlen=5)  # Start with -100 for first 5 steps
        current_pos = navigator.start_position.copy()
        
        # Process each action the robot took
        for i, action in enumerate(robot.genes):
            if i >= 25:  # Maximum steps limit
                break
                
            if not navigator.is_valid_move(current_pos, action):
                continue
                
            current_rssi = navigator.rssi_values[current_pos[0]][current_pos[1]]
            
            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "an action machine based on last 5 rssi values"
                    },
                    {
                        "role": "user",
                        "content": f"{list(rssi_window)}"
                    },
                    {
                        "role": "assistant",
                        "content": action
                    }
                ]
            }
            # 
            training_data.append(example)
            
            current_pos = navigator.apply_move(current_pos, action)
            rssi_window.append(current_rssi)
            
            if navigator.rssi_values[current_pos[0]][current_pos[1]] == 0.0:
                break
    
    return training_data

def main():
    # Initialize the navigator and evolve to get the best robots
    navigator = GeneticNavigator(population_size=100, gene_length=50)
    navigator.current_population = navigator.initialize_population()
    navigator.evolve(generations=300)
    
    # Get the top 12 robots by fitness
    top_robots = sorted(navigator.current_population, key=lambda x: x.fitness, reverse=True)[:12]
    
    # Create the training dataset using all top robots
    training_data = create_training_dataset(navigator, top_robots)
    
    # Write to file in JSONL format
    with open('llm_training_data.jsonl', 'w') as f:
        for example in training_data:
            f.write(json.dumps(example) + '\n')

if __name__ == "__main__":
    main()
