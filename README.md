# RSSI Navigation Visualization

A Python-based project that visualizes robot navigation using RSSI (Received Signal Strength Indicator) signals and genetic algorithms. The project demonstrates how robots can learn to navigate towards a signal source using evolutionary strategies.

## Features

- Interactive visualization of robot navigation using Pygame
- Genetic algorithm implementation for optimizing navigation paths
- RSSI signal strength visualization
- Machine Learning Dataset Creation
  - Generates structured training data from successful robot navigation paths
  - Uses a sliding window approach of last 5 RSSI readings
  - Creates examples in chat-completion format for ML training
  - Records successful navigation strategies from genetic algorithm
  - Each training example includes:
    - System prompt defining the task
    - User input: List of 5 consecutive RSSI readings
    - Assistant response: Optimal navigation action
  - Filters out invalid moves and limits maximum steps
  - Automatically stops data collection when target is reached
  - Includes fitness scores and robot performance metrics
- Multiple spawn points for testing navigation strategies

## Components

### 1. RSSI Visualizer (`rssi_visualizer.py`)
- Main visualization interface using Pygame
- Displays robots, RSSI heatmap, and navigation paths
- Interactive controls for simulation parameters
- Real-time visualization of genetic algorithm generations

### 2. Genetic Navigation (`genetic_navigation.py`)
- Implementation of genetic algorithms for robot navigation
- Robot class with genes representing movement sequences
- Fitness evaluation based on RSSI values
- Population management and evolution strategies

### 3. Dataset Creation (`create_llm_dataset.py`)
- Creates training datasets from successful navigation paths
- Formats data for machine learning models
- Records RSSI values and corresponding optimal actions
- Supports window-based observation patterns

## Requirements

```
pygame
numpy
scikit-learn
torch
```

## Installation

1. Clone the repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the main visualization:
   ```bash
   python rssi_visualizer.py
   ```

2. Create training datasets:
   ```bash
   python create_llm_dataset.py
   ```

## Controls

- Enter the target generation number when prompted
- Watch as robots evolve their navigation strategies
- Monitor the information panel for real-time statistics

## Technical Details

- Grid Size: 11x11
- RSSI Values: Range from 0.0 (strongest) to -16.09 (weakest)
- Population Size: 100 robots per generation
- Gene Length: 50 actions per robot
- Available Actions: left, right, straight, back

## Contributing

Feel free to submit issues and enhancement requests!
