import os
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
from stable_baselines3 import PPO
import time
import random

# Import our hybrid pathfinding system
from hybrid_pathfinding_system import HybridPathfindingSystem, AStar, HybridPathfindingEnv

def run_real_map_example(location="Central Park, New York, USA", 
                        network_type="drive", 
                        num_waypoints=5,
                        training_steps=30000):
    """
    Run a complete example of the hybrid pathfinding system on a real map.
    
    Args:
        location: String describing the location to use
        network_type: Type of network to extract ('drive', 'walk', 'bike', etc.)
        num_waypoints: Number of waypoints to generate
        training_steps: Number of training steps for PPO
    """
    print(f"Running hybrid pathfinding example on real map: {location}")
    
    # Step 1: Load the map data
    print("Loading map data...")
    try:
        # Download map data
        G = ox.graph_from_place(location, network_type=network_type)
        
        # Project the graph to use proper distance metrics
        G = ox.project_graph(G)
        
        # Add edge weights based on length
        for u, v, data in G.edges(data=True):
            data['weight'] = data.get('length', 1.0)
        
        # Add coordinates to nodes
        for node, data in G.nodes(data=True):
            data['x'] = data.get('x', 0.0)
            data['y'] = data.get('y', 0.0)
        
        print(f"Map loaded with {len(G.nodes())} nodes and {len(G.edges())} edges")
    except Exception as e:
        print(f"Error loading map: {e}")
        print("Please check your internet connection and the location name.")
        return
    
    # Step 2: Initialize the hybrid pathfinding system
    system = HybridPathfindingSystem(map_data=G)
    
    # Step 3: Select important nodes (start, waypoints, goal)
    all_nodes = list(G.nodes())
    
    if len(all_nodes) < num_waypoints + 2:
        print(f"Not enough nodes in the map. Adjusting waypoints to {len(all_nodes) - 2}")
        num_waypoints = max(1, len(all_nodes) - 2)
    
    # Select random nodes
    selected_nodes = random.sample(all_nodes, num_waypoints + 2)
    start_node = selected_nodes[0]
    waypoints = selected_nodes[1:-1]
    goal_node = selected_nodes[-1]
    
    # Step 4: Set up environment
    print("Setting up environment...")
    env = system.setup_environment(
        start_node=start_node,
        waypoints=waypoints,
        goal_node=goal_node
    )
    
    # Step 5: Visualize the initial setup
    print("Initial map setup:")
    system.env.render()
    plt.savefig("initial_map_setup.png")
    print("Initial setup saved as 'initial_map_setup.png'")
    
    # Step 6: Calculate pure A* solution
    print("Calculating pure A* solution...")
    astar = AStar(G)
    
    current = start_node
    total_path = []
    total_cost = 0
    remaining = waypoints.copy()
    
    # Visit all waypoints using A*
    while remaining:
        # Find closest waypoint
        closest = None
        min_cost = float('inf')
        closest_path = []
        
        for waypoint in remaining:
            path, cost = astar.find_path(current, waypoint)
            if cost < min_cost:
                min_cost = cost
                closest = waypoint
                closest_path = path
        
        # Add path to total path
        if len(total_path) > 0 and len(closest_path) > 0 and total_path[-1] == closest_path[0]:
            total_path.extend(closest_path[1:])
        else:
            total_path.extend(closest_path)
        
        total_cost += min_cost
        current = closest
        remaining.remove(closest)
    
    # Add path to goal
    path_to_goal, cost_to_goal = astar.find_path(current, goal_node)
    
    if len(total_path) > 0 and len(path_to_goal) > 0 and total_path[-1] == path_to_goal[0]:
        total_path.extend(path_to_goal[1:])
    else:
        total_path.extend(path_to_goal)
    
    total_cost += cost_to_goal
    
    print(f"A* solution found:")
    print(f"  Path length: {len(total_path)}")
    print(f"  Total cost: {total_cost:.2f}")
    
    # Step 7: Train the PPO agent
    print("\nTraining PPO agent...")
    print(f"  Training for {training_steps} steps...")
    start_time = time.time()
    
    system.model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )
    
    system.model.learn(total_timesteps=training_steps)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Step 8: Run and visualize the trained agent
    print("\nRunning trained agent...")
    
    # Create a figure to store frames
    frames = []
    
    # Prepare for visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Run episode
    obs = system.env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        # Predict action
        action, _ = system.model.predict(obs, deterministic=True)
        
        # Take action
        obs, reward, done, info = system.env.step(action)
        total_reward += reward
        steps += 1
        
        # Render and save frame
        system.env.render(mode='human')
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        frames.append(frame)
        
        # Clear axis for next frame
        ax.clear()
        
        # Don't run forever
        if steps > 1000:
            break
    
    print(f"Episode completed:")
    print(f"  Steps: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Visited waypoints: {len(system.env.visited_waypoints)}/{len(waypoints)}")
    
    # Step 9: Compare results
    print("\nComparison - A* vs. Hybrid approach:")
    print("-" * 50)
    print(f"Pure A*:")
    print(f"  Path length: {len(total_path)}")
    print(f"  Total cost: {total_cost:.2f}")
    print(f"Hybrid approach:")
    print(f"  Steps taken: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Success rate: {len(system.env.visited_waypoints)/len(waypoints):.2f}")
    
    # Final visualization - save the last frame
    plt.savefig("final_path_visualization.png")
    print("Final path visualization saved as 'final_path_visualization.png'")
    
    return {
        "system": system,
        "astar_path": total_path,
        "astar_cost": total_cost,
        "hybrid_steps": steps,
        "hybrid_reward": total_reward,
        "frames": frames
    }

def main():
    """
    Main function to run the example.
    """
    print("Hybrid Pathfinding on Real Map Example")
    print("-------------------------------------")
    
    # Get location from user
    try:
        location = input("Enter location (default: 'Central Park, New York, USA'): ").strip()
        if not location:
            location = "Central Park, New York, USA"
        
        network_type = input("Enter network type (drive/walk/bike, default: drive): ").strip().lower()
        if network_type not in ['drive', 'walk', 'bike']:
            network_type = 'drive'
        
        num_waypoints_input = input("Enter number of waypoints (default: 5): ").strip()
        num_waypoints = int(num_waypoints_input) if num_waypoints_input else 5
        
        training_steps_input = input("Enter training steps (default: 20000): ").strip()
        training_steps = int(training_steps_input) if training_steps_input else 20000
        
    except:
        # Fallback values
        location = "Central Park, New York, USA"
        network_type = "drive"
        num_waypoints = 5
        training_steps = 20000
    
    # Run example
    results = run_real_map_example(
        location=location,
        network_type=network_type,
        num_waypoints=num_waypoints,
        training_steps=training_steps
    )
    
    print("\nExample completed!")
    
    return results

if __name__ == "__main__":
    main()
