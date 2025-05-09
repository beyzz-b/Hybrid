# Hybrid Pathfinding System

## Project Overview

This project implements a hybrid pathfinding system that combines classical graph search algorithms (A* or Dijkstra) with reinforcement learning (PPO from Stable Baselines3). The system navigates through a map, visiting all required waypoints before reaching a final goal.

### Key Features

1. **Global Path Planning**: Uses A* to compute optimal paths between key points
2. **Local Navigation**: Employs PPO (Proximal Policy Optimization) to learn adaptive navigation
3. **Real Map Support**: Works with OpenStreetMap data via OSMnx
4. **Waypoint Navigation**: Finds efficient routes through multiple waypoints
5. **Visualization**: Includes tools to visualize paths and agent behavior

## Components

The system consists of several key components:

### 1. A* Algorithm Implementation (`AStar` class)
- Efficient implementation of the A* graph search algorithm
- Calculates optimal paths between nodes
- Supports custom heuristics for different map types

### 2. Custom Reinforcement Learning Environment (`HybridPathfindingEnv` class)
- Gym-compatible environment for agent training
- Observation space includes current position, distances to waypoints/goal, etc.
- Reward system encourages waypoint collection and efficient movement

### 3. Hybrid Pathfinding System (`HybridPathfindingSystem` class)
- Main class that integrates A* and PPO
- Manages environment setup, agent training, and evaluation
- Includes tools for comparing pure A* with the hybrid approach

### 4. Example Implementation (`example_real_map.py`)
- Demonstrates the system on real OpenStreetMap data
- Provides a complete workflow from map loading to agent training and evaluation

## Requirements

- Python 3.7+
- networkx
- numpy
- matplotlib
- gym
- stable-baselines3
- osmnx (for real map data)

## Installation

1. Install the required packages:

```bash
pip install networkx numpy matplotlib gym stable-baselines3
pip install osmnx  # Only needed for real map features
```

2. Clone this repository or download the source files.

## Usage

### Basic Usage

```python
from hybrid_pathfinding_system import HybridPathfindingSystem

# Create system with synthetic map
system = HybridPathfindingSystem()

# Set up environment with random nodes
env = system.setup_environment()

# Train the agent
system.train_agent(total_timesteps=20000)

# Run and visualize an episode
system.run_episode(render=True)

# Compare with pure A*
system.compare_with_pure_astar()
```

### Using Real Map Data

```python
# Load a real map
system = HybridPathfindingSystem(
    use_real_map=True,
    location="Manhattan, New York, USA"
)

# Continue as above...
```

### Running the Complete Example

```bash
python example_real_map.py
```

This will guide you through running a complete demonstration on real map data.

## How It Works

### 1. Global Path Planning with A*

A* algorithm computes optimal paths between key points (start → waypoints → goal), creating a high-level sequence of waypoints to visit.

```python
path, cost = astar.find_path(start_node, goal_node)
```

### 2. Local Navigation with PPO

The PPO agent learns to follow the global path while handling local navigation decisions and adapting to the environment.

```python
# Training the agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)

# Using the trained agent
obs = env.reset()
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
```

### 3. Hybrid Decision-Making

The system combines the strengths of both approaches:
- A* provides an optimal global path
- PPO handles local navigation decisions and adapts to the environment
- The agent receives rewards for following the global path and reaching waypoints

## Visualization

The system includes visualization tools to display:
- The map and graph structure
- Start node, waypoints, and goal
- Global path planned by A*
- Agent's position and movement
- Visited and unvisited waypoints

## Customization

You can customize various aspects of the system:

1. **Graph Creation**:
   - Use synthetic graphs for testing
   - Load real map data for any location

2. **Environment Parameters**:
   - Adjust reward functions
   - Modify observation space

3. **Training Parameters**:
   - Change PPO hyperparameters
   - Adjust training duration

## Performance Comparison

The system includes tools to compare the performance of:
- Pure A* approach (optimal but inflexible)
- Hybrid A*+PPO approach (adaptable but may be suboptimal)

Metrics include:
- Path length
- Computation time
- Success rate
- Rewards earned

## Future Improvements

Potential enhancements to the system:
1. Support for dynamic obstacles
2. Multi-agent pathfinding
3. Integration with more complex RL algorithms
4. Online learning capabilities
5. Better visualization and real-time monitoring
