import numpy as np
import heapq
import matplotlib.pyplot as plt
import gym
from gym import spaces
import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import osmnx as ox
import random
from typing import List, Tuple, Dict, Set, Optional
import time
import math

# -------------------------------------------------------------------------
# Part 1: A* Algorithm Implementation for Global Path Planning
# -------------------------------------------------------------------------

class AStar:
    """A* algorithm implementation for pathfinding on graphs."""
    
    def __init__(self, graph):
        """
        Initialize the A* pathfinder.
        
        Args:
            graph: A NetworkX graph or a custom graph representation
        """
        self.graph = graph
    
    def heuristic(self, a, b):
        """
        Calculate the heuristic (estimated distance) between nodes a and b.
        For geographic data, this could be Haversine or Euclidean distance.
        
        Args:
            a: Start node
            b: Goal node
            
        Returns:
            Estimated distance between a and b
        """
        # For geographic coordinates
        if hasattr(self.graph, 'nodes') and 'x' in self.graph.nodes[a] and 'y' in self.graph.nodes[a]:
            x1, y1 = self.graph.nodes[a]['x'], self.graph.nodes[a]['y']
            x2, y2 = self.graph.nodes[b]['x'], self.graph.nodes[b]['y']
            return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        # Default Euclidean distance for non-geographic graphs
        else:
            return 1  # Can be customized based on the graph structure
    
    def find_path(self, start, goal):
        """
        Find the shortest path between start and goal using A* algorithm.
        
        Args:
            start: Start node
            goal: Goal node
            
        Returns:
            path: List of nodes forming the path from start to goal
            cost: Total cost of the path
        """
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current_cost, current = heapq.heappop(frontier)
            
            if current == goal:
                break
                
            # For each neighbor of the current node
            if hasattr(self.graph, 'neighbors'):  # NetworkX-style graph
                neighbors = list(self.graph.neighbors(current))
            else:  # Custom graph representation
                neighbors = self.graph.get_neighbors(current)
                
            for next_node in neighbors:
                # Calculate the cost to the next node
                if hasattr(self.graph, 'get_edge_data'):  # NetworkX-style graph
                    edge_data = self.graph.get_edge_data(current, next_node)
                    if edge_data and 'weight' in edge_data:
                        new_cost = cost_so_far[current] + edge_data['weight']
                    else:
                        new_cost = cost_so_far[current] + 1
                else:  # Custom graph representation
                    new_cost = cost_so_far[current] + self.graph.get_edge_cost(current, next_node)
                
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self.heuristic(next_node, goal)
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current
        
        # Reconstruct the path
        path = []
        if goal in came_from:
            current = goal
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, cost_so_far[goal]
        else:
            return [], float('inf')  # No path found

# -------------------------------------------------------------------------
# Part 2: Custom Environment for Reinforcement Learning
# -------------------------------------------------------------------------

class HybridPathfindingEnv(gym.Env):
    """Custom Environment for hybrid pathfinding that follows gym interface."""
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, graph, start_node, waypoints, goal_node, 
                 global_planner=None, observation_size=10):
        """
        Initialize the environment.
        
        Args:
            graph: The underlying graph representation of the environment
            start_node: Starting node ID
            waypoints: List of waypoint node IDs that must be visited
            goal_node: Final goal node ID
            global_planner: Global pathfinding algorithm (default: A*)
            observation_size: Size of the observation space around the agent
        """
        super(HybridPathfindingEnv, self).__init__()
        
        self.graph = graph
        self.start_node = start_node
        self.original_waypoints = waypoints.copy()
        self.waypoints = waypoints.copy()  # List of waypoints to visit
        self.goal_node = goal_node
        self.global_planner = global_planner if global_planner else AStar(graph)
        
        # Current state
        self.current_node = None
        self.current_waypoint_target = None
        self.global_path = []
        self.visited_waypoints = set()
        self.steps_taken = 0
        self.max_steps = 1000  # Maximum steps before termination
        
        # Extract graph dimensions for observation space
        if hasattr(graph, 'nodes'):
            self.nodes_data = list(graph.nodes(data=True))
            if self.nodes_data and 'x' in self.nodes_data[0][1] and 'y' in self.nodes_data[0][1]:
                x_coords = [data['x'] for _, data in self.nodes_data]
                y_coords = [data['y'] for _, data in self.nodes_data]
                self.min_x, self.max_x = min(x_coords), max(x_coords)
                self.min_y, self.max_y = min(y_coords), max(y_coords)
            else:
                # Default dimensions if coordinates are not available
                self.min_x, self.max_x = 0, 100
                self.min_y, self.max_y = 0, 100
        
        # Define action and observation space
        # Action space: Movement directions (e.g., to neighbors)
        if hasattr(graph, 'nodes'):
            max_degree = max(dict(graph.degree()).values())
            self.action_space = spaces.Discrete(max_degree + 1)  # +1 for "stay" action
        else:
            self.action_space = spaces.Discrete(9)  # Default: 8 directions + stay
        
        # Observation space:
        # - Current position (normalized)
        # - Relative position to next waypoint
        # - Relative position to final goal
        # - Local graph structure encoded in some way
        self.observation_size = observation_size
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(observation_size,),
            dtype=np.float32
        )
        
        # Graph visualization
        self.fig = None
        self.ax = None
    
    def reset(self):
        """Reset the environment to initial state."""
        self.current_node = self.start_node
        self.waypoints = self.original_waypoints.copy()
        self.visited_waypoints = set()
        self.steps_taken = 0
        
        # Determine the next waypoint to visit
        self._update_current_waypoint_target()
        
        # Calculate initial global path
        self._update_global_path()
        
        return self._get_observation()
    
    def _update_current_waypoint_target(self):
        """Update the current waypoint target based on visited waypoints."""
        if not self.waypoints:
            # All waypoints visited, target the goal
            self.current_waypoint_target = self.goal_node
        else:
            # Find the closest unvisited waypoint
            min_dist = float('inf')
            closest_waypoint = None
            
            for waypoint in self.waypoints:
                path, cost = self.global_planner.find_path(self.current_node, waypoint)
                if cost < min_dist:
                    min_dist = cost
                    closest_waypoint = waypoint
            
            self.current_waypoint_target = closest_waypoint
    
    def _update_global_path(self):
        """Update the global path to the current waypoint target."""
        if self.current_waypoint_target:
            self.global_path, _ = self.global_planner.find_path(
                self.current_node, self.current_waypoint_target)
        else:
            self.global_path = []
    
    def _get_observation(self):
        """
        Get the current observation of the environment.
        
        Returns:
            observation: A numpy array representing the state
        """
        observation = np.zeros(self.observation_size, dtype=np.float32)
        
        # Current position (normalized)
        if hasattr(self.graph, 'nodes'):
            x = self.graph.nodes[self.current_node].get('x', 0)
            y = self.graph.nodes[self.current_node].get('y', 0)
            
            # Normalize coordinates
            norm_x = (x - self.min_x) / (self.max_x - self.min_x)
            norm_y = (y - self.min_y) / (self.max_y - self.min_y)
            
            observation[0] = norm_x
            observation[1] = norm_y
            
            # Distance to current waypoint target (normalized)
            if self.current_waypoint_target:
                target_x = self.graph.nodes[self.current_waypoint_target].get('x', 0)
                target_y = self.graph.nodes[self.current_waypoint_target].get('y', 0)
                
                # Relative position to target
                rel_x = target_x - x
                rel_y = target_y - y
                distance = math.sqrt(rel_x**2 + rel_y**2)
                
                # Normalize by maximum possible distance
                max_distance = math.sqrt((self.max_x - self.min_x)**2 + (self.max_y - self.min_y)**2)
                norm_distance = distance / max_distance
                
                observation[2] = norm_distance
                
                # Direction to target (normalized)
                if distance > 0:
                    observation[3] = rel_x / distance  # Normalized x direction
                    observation[4] = rel_y / distance  # Normalized y direction
                else:
                    observation[3] = 0
                    observation[4] = 0
            
            # Distance to goal (normalized)
            goal_x = self.graph.nodes[self.goal_node].get('x', 0)
            goal_y = self.graph.nodes[self.goal_node].get('y', 0)
            
            rel_x_goal = goal_x - x
            rel_y_goal = goal_y - y
            distance_goal = math.sqrt(rel_x_goal**2 + rel_y_goal**2)
            
            norm_distance_goal = distance_goal / max_distance
            
            observation[5] = norm_distance_goal
            
            # Direction to goal (normalized)
            if distance_goal > 0:
                observation[6] = rel_x_goal / distance_goal
                observation[7] = rel_y_goal / distance_goal
            else:
                observation[6] = 0
                observation[7] = 0
            
            # Progress indicator (visited waypoints / total waypoints)
            observation[8] = len(self.visited_waypoints) / max(1, len(self.original_waypoints))
            
            # Global path suggestion (if available)
            if len(self.global_path) > 1:
                next_node = self.global_path[1]  # Next node in the global path
                next_x = self.graph.nodes[next_node].get('x', 0)
                next_y = self.graph.nodes[next_node].get('y', 0)
                
                # Direction to next node in global path
                rel_x_next = next_x - x
                rel_y_next = next_y - y
                distance_next = math.sqrt(rel_x_next**2 + rel_y_next**2)
                
                if distance_next > 0:
                    observation[9] = rel_x_next / distance_next
                else:
                    observation[9] = 0
            else:
                observation[9] = 0
        
        return observation
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            observation: The new observation
            reward: The reward for taking the action
            done: Whether the episode is finished
            info: Additional information
        """
        self.steps_taken += 1
        
        # Get valid neighbors
        if hasattr(self.graph, 'neighbors'):
            neighbors = list(self.graph.neighbors(self.current_node))
        else:
            neighbors = self.graph.get_neighbors(self.current_node)
        
        # Map action to neighbor
        if action == 0 or action >= len(neighbors) + 1:
            # Stay at current node (or invalid action)
            next_node = self.current_node
            movement_reward = -0.1  # Small penalty for staying still
        else:
            # Move to a neighbor
            next_node = neighbors[action - 1]
            movement_reward = -0.05  # Small cost for movement
        
        # Update current node
        self.current_node = next_node
        
        # Check if we reached the current waypoint target
        reached_waypoint = False
        if self.current_node == self.current_waypoint_target and self.current_waypoint_target in self.waypoints:
            self.waypoints.remove(self.current_waypoint_target)
            self.visited_waypoints.add(self.current_waypoint_target)
            reached_waypoint = True
            self._update_current_waypoint_target()
        
        # Update global path
        self._update_global_path()
        
        # Calculate reward
        reward = movement_reward
        
        # Big reward for reaching a waypoint
        if reached_waypoint:
            reward += 10.0
        
        # Check if the agent follows the global path
        if len(self.global_path) > 1 and self.current_node == self.global_path[1]:
            reward += 0.5  # Reward for following the global path
        
        # Check if we reached the goal
        done = False
        if self.current_node == self.goal_node and not self.waypoints:
            # All waypoints visited and reached the goal
            reward += 100.0
            done = True
        
        # Check if we exceeded max steps
        if self.steps_taken >= self.max_steps:
            done = True
            reward -= 10.0  # Penalty for timeout
        
        # Get new observation
        observation = self._get_observation()
        
        info = {
            'current_node': self.current_node,
            'remaining_waypoints': len(self.waypoints),
            'visited_waypoints': len(self.visited_waypoints),
            'steps_taken': self.steps_taken
        }
        
        return observation, reward, done, info
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: The rendering mode
            
        Returns:
            Visual representation of the environment
        """
        if not self.fig:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        self.ax.clear()
        
        # Draw the graph
        if hasattr(self.graph, 'nodes'):
            # Draw nodes
            node_positions = {}
            for node in self.graph.nodes():
                if 'x' in self.graph.nodes[node] and 'y' in self.graph.nodes[node]:
                    node_positions[node] = (
                        self.graph.nodes[node]['x'],
                        self.graph.nodes[node]['y']
                    )
            
            # Draw edges
            for u, v in self.graph.edges():
                if u in node_positions and v in node_positions:
                    x1, y1 = node_positions[u]
                    x2, y2 = node_positions[v]
                    self.ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=0.5)
            
            # Draw start node
            if self.start_node in node_positions:
                x, y = node_positions[self.start_node]
                self.ax.plot(x, y, 'go', markersize=10, label='Start')
            
            # Draw waypoints
            for wp in self.original_waypoints:
                if wp in node_positions:
                    x, y = node_positions[wp]
                    if wp in self.visited_waypoints:
                        self.ax.plot(x, y, 'bx', markersize=8, label='Visited Waypoint')
                    else:
                        self.ax.plot(x, y, 'bo', markersize=8, label='Waypoint')
            
            # Draw goal node
            if self.goal_node in node_positions:
                x, y = node_positions[self.goal_node]
                self.ax.plot(x, y, 'ro', markersize=10, label='Goal')
            
            # Draw current node
            if self.current_node in node_positions:
                x, y = node_positions[self.current_node]
                self.ax.plot(x, y, 'mo', markersize=8, label='Current')
            
            # Draw global path
            if self.global_path:
                path_x = []
                path_y = []
                for node in self.global_path:
                    if node in node_positions:
                        x, y = node_positions[node]
                        path_x.append(x)
                        path_y.append(y)
                self.ax.plot(path_x, path_y, 'g--', linewidth=2, alpha=0.7, label='Global Path')
        
        # Set plot limits
        self.ax.set_xlim(self.min_x - 0.1, self.max_x + 0.1)
        self.ax.set_ylim(self.min_y - 0.1, self.max_y + 0.1)
        
        # Set title and labels
        self.ax.set_title('Hybrid Pathfinding Environment')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        
        # Create legend without duplicates
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        plt.tight_layout()
        
        if mode == 'human':
            plt.pause(0.1)
            return self.fig
        elif mode == 'rgb_array':
            self.fig.canvas.draw()
            img = np.array(self.fig.canvas.renderer.buffer_rgba())
            return img

# -------------------------------------------------------------------------
# Part 3: Hybrid Pathfinding System Integration
# -------------------------------------------------------------------------

class HybridPathfindingSystem:
    """
    Main class that integrates A*/Dijkstra for global planning and
    PPO for local navigation.
    """
    
    def __init__(self, map_data=None, use_real_map=False, location=None):
        """
        Initialize the hybrid pathfinding system.
        
        Args:
            map_data: Graph data or None to create a synthetic graph
            use_real_map: Whether to use real map data from OSMnx
            location: Location name for OSMnx (if use_real_map is True)
        """
        if use_real_map and location:
            # Load real map data
            print(f"Loading real map data for {location}...")
            self.graph = self._load_real_map(location)
        elif map_data:
            # Use provided map data
            self.graph = map_data
        else:
            # Create synthetic graph
            print("Creating synthetic graph...")
            self.graph = self._create_synthetic_graph()
        
        # Initialize global planner
        self.global_planner = AStar(self.graph)
        
        # Environment and RL components will be initialized later
        self.env = None
        self.model = None
    
    def _load_real_map(self, location, network_type='drive'):
        """
        Load real map data using OSMnx.
        
        Args:
            location: Name of the location (e.g., 'Manhattan, New York, USA')
            network_type: Type of network to download
            
        Returns:
            NetworkX graph of the map
        """
        try:
            # Download map data
            G = ox.graph_from_place(location, network_type=network_type)
            
            # Project graph to use meaningful distances
            G = ox.project_graph(G)
            
            # Add edge weights based on length
            for u, v, data in G.edges(data=True):
                data['weight'] = data.get('length', 1.0)
            
            return G
        except Exception as e:
            print(f"Error loading real map: {e}")
            print("Falling back to synthetic graph...")
            return self._create_synthetic_graph()
    
    def _create_synthetic_graph(self, grid_size=10):
        """
        Create a synthetic grid graph for testing.
        
        Args:
            grid_size: Size of the grid
            
        Returns:
            NetworkX graph
        """
        # Create grid graph
        G = nx.grid_2d_graph(grid_size, grid_size)
        
        # Convert to node IDs and add coordinates
        mapping = {}
        for i, node in enumerate(G.nodes()):
            mapping[node] = i
            G.nodes[node]['x'] = node[0]
            G.nodes[node]['y'] = node[1]
        
        G = nx.relabel_nodes(G, mapping)
        
        # Add edge weights (all 1 by default)
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0
        
        return G
    
    def setup_environment(self, start_node=None, waypoints=None, goal_node=None):
        """
        Set up the environment with specified nodes.
        
        Args:
            start_node: Starting node (or random if None)
            waypoints: List of waypoint nodes (or random if None)
            goal_node: Goal node (or random if None)
            
        Returns:
            The configured environment
        """
        # Select random nodes if not specified
        all_nodes = list(self.graph.nodes())
        
        if start_node is None:
            start_node = random.choice(all_nodes)
        
        if waypoints is None:
            # Select random waypoints (different from start and goal)
            num_waypoints = min(5, len(all_nodes) - 2)  # At most 5 waypoints
            waypoints = []
            available_nodes = [n for n in all_nodes if n != start_node]
            
            for _ in range(num_waypoints):
                if not available_nodes:
                    break
                wp = random.choice(available_nodes)
                waypoints.append(wp)
                available_nodes.remove(wp)
        
        if goal_node is None:
            # Select random goal (different from start and waypoints)
            available_nodes = [n for n in all_nodes if n != start_node and n not in waypoints]
            if available_nodes:
                goal_node = random.choice(available_nodes)
            else:
                # Fallback if no nodes are available
                goal_node = random.choice(all_nodes)
        
        print(f"Environment setup:")
        print(f"  Start node: {start_node}")
        print(f"  Waypoints: {waypoints}")
        print(f"  Goal node: {goal_node}")
        
        # Create environment
        self.env = HybridPathfindingEnv(
            graph=self.graph,
            start_node=start_node,
            waypoints=waypoints,
            goal_node=goal_node,
            global_planner=self.global_planner
        )
        
        # Wrap environment for Stable Baselines
        return DummyVecEnv([lambda: self.env])
    
    def train_agent(self, env=None, total_timesteps=50000):
        """
        Train the PPO agent.
        
        Args:
            env: Environment to train in (or use self.env if None)
            total_timesteps: Number of timesteps to train for
            
        Returns:
            Trained PPO model
        """
        if env is None:
            if self.env is None:
                env = self.setup_environment()
            else:
                env = DummyVecEnv([lambda: self.env])
        
        print("Training PPO agent...")
        self.model = PPO(
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
        
        self.model.learn(total_timesteps=total_timesteps)
        print("Training completed!")
        
        return self.model
    
    def evaluate_agent(self, num_episodes=5):
        """
        Evaluate the trained agent over multiple episodes.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Average reward
        """
        if self.model is None:
            print("Error: Agent not trained yet!")
            return None
        
        print(f"Evaluating agent over {num_episodes} episodes...")
        mean_reward, _ = evaluate_policy(
            self.model, 
            self.env, 
            n_eval_episodes=num_episodes, 
            deterministic=True
        )
        
        print(f"Mean reward: {mean_reward:.2f}")
        return mean_reward
    
    def run_episode(self, render=True):
        """
        Run a complete episode with the trained agent.
        
        Args:
            render: Whether to render the environment
            
        Returns:
            Total reward, steps taken, and success status
        """
        if self.model is None:
            print("Error: Agent not trained yet!")
            return None
        
        obs = self.env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            steps += 1
            
            if render:
                self.env.render()
                time.sleep(0.1)  # Add delay for better visualization
        
        success = self.env.current_node == self.env.goal_node and not self.env.waypoints
        
        print(f"Episode finished after {steps} steps")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Success: {success}")
        print(f"Visited {len(self.env.visited_waypoints)}/{len(self.env.original_waypoints)} waypoints")
        
        return total_reward, steps, success
    
    def compare_with_pure_astar(self):
        """
        Compare the hybrid approach with pure A* for visiting all waypoints.
        
        Returns:
            Comparison results
        """
        if self.env is None:
            print("Error: Environment not configured yet!")
            return None
        
        print("Comparing hybrid approach with pure A*...")
        
        # Pure A* approach
        start_time = time.time()
        
        current = self.env.start_node
        total_path = []
        total_cost = 0
        
        # Visit all waypoints in optimal order using A*
        remaining = self.env.original_waypoints.copy()
        
        while remaining:
            # Find closest waypoint
            closest = None
            min_cost = float('inf')
            closest_path = []
            
            for waypoint in remaining:
                path, cost = self.global_planner.find_path(current, waypoint)
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
        path_to_goal, cost_to_goal = self.global_planner.find_path(current, self.env.goal_node)
        
        if len(total_path) > 0 and len(path_to_goal) > 0 and total_path[-1] == path_to_goal[0]:
            total_path.extend(path_to_goal[1:])
        else:
            total_path.extend(path_to_goal)
        
        total_cost += cost_to_goal
        astar_time = time.time() - start_time
        
        # Run hybrid approach
        start_time = time.time()
        reward, steps, success = self.run_episode(render=False)
        hybrid_time = time.time() - start_time
        
        # Compare results
        print("\nComparison Results:")
        print("-------------------")
        print(f"Pure A* path length: {len(total_path)}")
        print(f"Pure A* cost: {total_cost:.2f}")
        print(f"Pure A* computation time: {astar_time:.4f} seconds")
        print("\n")
        print(f"Hybrid approach steps: {steps}")
        print(f"Hybrid approach reward: {reward:.2f}")
        print(f"Hybrid approach success: {success}")
        print(f"Hybrid approach computation time: {hybrid_time:.4f} seconds")
        
        return {
            "astar": {
                "path_length": len(total_path),
                "cost": total_cost,
                "time": astar_time
            },
            "hybrid": {
                "steps": steps,
                "reward": reward,
                "success": success,
                "time": hybrid_time
            }
        }

# -------------------------------------------------------------------------
# Part 4: Demonstration and Visualization
# -------------------------------------------------------------------------

def main():
    """
    Main demonstration function.
    """
    print("Hybrid Pathfinding System: A* + PPO")
    print("-----------------------------------")
    
    # Ask user if they want to use real map data
    use_real_map = False
    location = None
    
    try:
        use_real_map_input = input("Use real map data? (y/n, default: n): ").strip().lower()
        use_real_map = use_real_map_input == 'y'
        
        if use_real_map:
            location = input("Enter location (e.g., 'Manhattan, New York, USA'): ").strip()
            if not location:
                print("No location provided, falling back to synthetic graph.")
                use_real_map = False
    except:
        # Fallback in case of no input available (e.g., in notebook)
        print("Using synthetic graph as default...")
    
    # Create hybrid pathfinding system
    system = HybridPathfindingSystem(use_real_map=use_real_map, location=location)
    
    # Set up environment
    vec_env = system.setup_environment()
    
    # Training parameters
    try:
        timesteps_input = input("Enter training timesteps (default: 20000): ").strip()
        total_timesteps = int(timesteps_input) if timesteps_input else 20000
    except:
        total_timesteps = 20000
    
    # Train agent
    system.train_agent(total_timesteps=total_timesteps)
    
    # Evaluate agent
    system.evaluate_agent(num_episodes=3)
    
    # Run and visualize episode
    system.run_episode(render=True)
    
    # Compare with pure A*
    system.compare_with_pure_astar()
    
    print("\nDemonstration completed!")

if __name__ == "__main__":
    main()