# slbfs_controller.py
import numpy as np
from collections import deque
import logging
import math

class SLBFSController:
    """
    A baseline MAPF controller using Step-wise Local BFS within agent observations.
    """
    def __init__(self, obs_radius):
        self.obs_radius = obs_radius
        self.obs_H, self.obs_W = (obs_radius * 2 + 1, obs_radius * 2 + 1)
        self.center_r, self.center_c = self.obs_H // 2, self.obs_W // 2

        # Action mapping: 0:Stay, 1:Up, 2:Down, 3:Left, 4:Right
        self.action_delta_map = {
            0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)
        }
        self.delta_action_map = {v: k for k, v in self.action_delta_map.items()}
        logging.info("SLBFSController initialized.")

    def _local_bfs(self, local_obstacles, local_agents, start_rc, target_rc):
        """
        Performs BFS within the local observation window.

        Args:
            local_obstacles (np.ndarray): Obstacle map (HxW, 1=obstacle).
            local_agents (np.ndarray): Other agents map (HxW, 1=agent).
            start_rc (tuple): Starting (row, col) within the window (agent's center).
            target_rc (tuple): Target (row, col) within the window.

        Returns:
            int or None: The first action (0-4) on the shortest path, or None if target is unreachable.
        """
        q = deque([(start_rc, [])]) # Queue stores ((r, c), path_list)
        visited = {start_rc}
        parent = {} # Store parent to reconstruct path: child_coords -> parent_coords

        while q:
            (r, c), path = q.popleft()

            if (r, c) == target_rc:
                # Found target, reconstruct first step
                if not path: # Started at target
                    return 0 # Stay
                # Backtrack to find the step taken from the start node
                curr = target_rc
                while parent.get(curr) != start_rc and parent.get(curr) is not None:
                    curr = parent[curr]
                # Now curr is the first step taken from start_rc
                dr = curr[0] - start_rc[0]
                dc = curr[1] - start_rc[1]
                return self.delta_action_map.get((dr, dc), 0) # Get action, default Stay

            # Explore neighbors
            for action_idx, (dr, dc) in self.action_delta_map.items():
                if action_idx == 0: continue # Don't explore staying

                nr, nc = r + dr, c + dc

                # Check bounds, obstacles, agents, and visited
                if (0 <= nr < self.obs_H and 0 <= nc < self.obs_W and
                        local_obstacles[nr, nc] == 0 and
                        local_agents[nr, nc] == 0 and # Treat other agents as obstacles
                        (nr, nc) not in visited):

                    visited.add((nr, nc))
                    parent[(nr, nc)] = (r, c) # Store parent
                    new_path = path + [(nr, nc)] # Append coordinate to path list
                    q.append(((nr, nc), new_path))

        return None # Target not reachable within the window

    def _heuristic_move(self, local_obstacles, local_agents, start_rc, global_pos, global_goal):
        """
        Fallback heuristic when target is not visible/valid in local BFS.
        Moves to the valid neighbor that minimizes Euclidean distance to the global goal.
        """
        min_dist_sq = float('inf')
        best_action = 0 # Default Stay

        start_r, start_c = start_rc # Local center coordinates

        for action_idx, (dr, dc) in self.action_delta_map.items():
            if action_idx == 0: continue # Only consider actual moves

            nr_local, nc_local = start_r + dr, start_c + dc

            # Check local validity (bounds, obstacle, agent)
            if not (0 <= nr_local < self.obs_H and 0 <= nc_local < self.obs_W and
                    local_obstacles[nr_local, nc_local] == 0 and
                    local_agents[nr_local, nc_local] == 0):
                continue # Skip invalid neighbors

            # Calculate global position of this valid neighbor
            neighbor_global_r = global_pos[0] + dr
            neighbor_global_c = global_pos[1] + dc

            # Calculate squared Euclidean distance to global goal
            dist_sq = (neighbor_global_r - global_goal[0])**2 + (neighbor_global_c - global_goal[1])**2

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_action = action_idx

        return best_action


    def compute_actions(self, observations, global_positions_xy, global_goals_xy):
        """
        Computes actions for all agents based on local BFS or heuristics.

        Args:
            observations (list): List of observations for each agent. Can be dict or ndarray.
            global_positions_xy (list): List of current global (row, col) positions.
            global_goals_xy (list): List of global (row, col) goals.

        Returns:
            list: List of intended actions (int) for each agent.
        """
        num_agents = len(observations)
        intended_actions = [0] * num_agents # Default Stay

        for i in range(num_agents):
            agent_obs_data = observations[i]
            agent_pos = global_positions_xy[i]
            agent_goal = global_goals_xy[i]

            # --- Extract Obstacle and Agent Channels ---
            local_obstacles = np.zeros((self.obs_H, self.obs_W), dtype=int)
            local_agents = np.zeros((self.obs_H, self.obs_W), dtype=int)

            if isinstance(agent_obs_data, dict):
                obs_ch = agent_obs_data.get("obstacles")
                agt_ch = agent_obs_data.get("agents")
                if obs_ch is not None: local_obstacles = (obs_ch > 0.5).astype(int)
                if agt_ch is not None: local_agents = (agt_ch > 0.5).astype(int)
            elif isinstance(agent_obs_data, np.ndarray):
                if agent_obs_data.shape[0] >= 1:
                     local_obstacles = (agent_obs_data[0] > 0.5).astype(int)
                if agent_obs_data.shape[0] >= 2:
                     local_agents = (agent_obs_data[1] > 0.5).astype(int)
            else:
                logging.warning(f"SL-BFS: Agent {i}: Unexpected observation type {type(agent_obs_data)}. Agent will STAY.")
                intended_actions[i] = 0
                continue

             # Ensure agent's own position is not marked as an agent obstacle
            local_agents[self.center_r, self.center_c] = 0


            # --- Determine Local Target Coords & Validity ---
            relative_target_r = agent_goal[0] - agent_pos[0]
            relative_target_c = agent_goal[1] - agent_pos[1]
            target_idx_r = self.center_r + relative_target_r
            target_idx_c = self.center_c + relative_target_c

            is_target_visible = (0 <= target_idx_r < self.obs_H and 0 <= target_idx_c < self.obs_W)
            is_target_on_obstacle = is_target_visible and local_obstacles[target_idx_r, target_idx_c] == 1
            is_target_valid_for_bfs = is_target_visible and not is_target_on_obstacle

            # --- Decide Action ---
            action = 0 # Default Stay
            if is_target_valid_for_bfs:
                # Try local BFS
                bfs_action = self._local_bfs(local_obstacles, local_agents,
                                             (self.center_r, self.center_c),
                                             (target_idx_r, target_idx_c))
                if bfs_action is not None:
                    action = bfs_action
                    # logging.debug(f"Agent {i}: Local BFS action = {action}")
                else:
                    # Target visible but unreachable locally, use heuristic
                    # logging.debug(f"Agent {i}: Target visible but unreachable locally, using heuristic.")
                    action = self._heuristic_move(local_obstacles, local_agents,
                                                  (self.center_r, self.center_c),
                                                  agent_pos, agent_goal)
            else:
                # Target not visible or invalid, use heuristic
                # logging.debug(f"Agent {i}: Target not visible or invalid, using heuristic.")
                action = self._heuristic_move(local_obstacles, local_agents,
                                              (self.center_r, self.center_c),
                                              agent_pos, agent_goal)

            intended_actions[i] = action

        return intended_actions