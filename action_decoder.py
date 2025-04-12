# action_decoder.py
import numpy as np
import logging
import random # Needed for probabilistic sampling if multiple max probabilities

class ActionDecoder:
    """
    Decodes actions for agents based on predicted potential field patches.
    """
    def __init__(self, strategy='gradient', temperature=1.0): # Added temperature
        """
        Args:
            strategy (str): 'gradient' or 'probabilistic'.
            temperature (float): Scaling factor for probabilistic strategy.
                                 Higher temp -> more randomness. Must be > 0.
        """
        if strategy not in ['gradient', 'probabilistic']:
             raise ValueError("Strategy must be 'gradient' or 'probabilistic'")
        if temperature <= 0:
             logging.warning("Temperature should be > 0 for probabilistic strategy. Setting to 1.0.")
             temperature = 1.0

        self.strategy = strategy
        self.temperature = temperature
        # POGEMA Action mapping: 0:Stay, 1:Up, 2:Down, 3:Left, 4:Right
        # Delta mapping: (dr, dc) for each action index
        self.action_delta_map = {
            0: (0, 0),  # Stay
            1: (-1, 0), # Up
            2: (1, 0),  # Down
            3: (0, -1), # Left
            4: (0, 1)   # Right
        }
        self.delta_action_map = {v: k for k, v in self.action_delta_map.items()}
        logging.info(f"ActionDecoder initialized with strategy: {self.strategy}, Temperature: {self.temperature if self.strategy == 'probabilistic' else 'N/A'}")

    def _get_potential_at_coord(self, potential_patch, r, c):
        """Safely get potential value from patch, handles out-of-bounds."""
        H, W = potential_patch.shape
        if 0 <= r < H and 0 <= c < W:
            return potential_patch[r, c]
        else:
            return 1e9 # Assign high potential to out-of-bounds

    def _get_valid_neighbors(self, obs_ch, agents_ch, r_center, c_center, patch_shape):
        """ Identifies valid neighboring cells (including current cell for Stay). """
        valid_actions = {} # action_idx -> (nr, nc)
        H, W = patch_shape
        for action_idx, (dr, dc) in self.action_delta_map.items():
            nr, nc = r_center + dr, c_center + dc

            # Check bounds
            if not (0 <= nr < H and 0 <= nc < W):
                continue

            # Check obstacles/agents (allow staying on current spot even if agent channel is 1)
            is_obstacle = obs_ch[nr, nc] > 0.5
            # Only check agent collision if moving to a new cell
            is_agent = agents_ch[nr, nc] > 0.5 and (dr != 0 or dc != 0)

            if not is_obstacle and not is_agent:
                valid_actions[action_idx] = (nr, nc)
        return valid_actions


    def decode_actions(self, predicted_potentials, current_observations):
        """
        Decodes intended actions for a batch of agents.

        Args:
            predicted_potentials (list or np.ndarray): List or array of predicted potential field patches
                                                     (each shape HxW) for each agent.
            current_observations (list or np.ndarray): List or array of corresponding agent observations
                                                      (input tensors, shape CxHxW) used for the prediction.

        Returns:
            list: A list of intended actions (int) for each agent.
        """
        num_agents = len(predicted_potentials)
        intended_actions = []
        if num_agents == 0: return intended_actions

        obs_H, obs_W = predicted_potentials[0].shape # Assume all patches have same shape
        center_r, center_c = obs_H // 2, obs_W // 2 # Agent's position in the patch

        for i in range(num_agents):
            potential_patch = predicted_potentials[i]
            # Assuming input channels are available as passed
            # Ch 0: Obstacles, Ch 1: Other Agents
            obstacle_channel = current_observations[i][0]
            agents_channel = current_observations[i][1]
            patch_shape = (obs_H, obs_W)

            # Find valid moves first
            valid_moves = self._get_valid_neighbors(obstacle_channel, agents_channel, center_r, center_c, patch_shape)

            if not valid_moves: # Agent is trapped
                action = 0 # Force Stay
            elif self.strategy == 'gradient':
                action = self._decode_gradient(potential_patch, valid_moves, center_r, center_c)
            elif self.strategy == 'probabilistic':
                action = self._decode_probabilistic(potential_patch, valid_moves, center_r, center_c)
            else:
                 # Should not happen due to init check
                 action = 0

            intended_actions.append(action)

        return intended_actions

    def _decode_gradient(self, potential_patch, valid_moves, r_center, c_center):
        """Find the valid action leading to the lowest potential."""
        best_action = 0  # Default: Stay (should usually be in valid_moves if not trapped)
        # Initialize min potential with current cell potential, assuming Stay (0) is valid
        min_potential = self._get_potential_at_coord(potential_patch, r_center, c_center)
        if 0 not in valid_moves: # If somehow staying is invalid (e.g., on an obstacle - shouldn't happen)
             min_potential = float('inf') # Ensure first valid move is chosen
             best_action = random.choice(list(valid_moves.keys())) if valid_moves else 0 # Pick random valid if Stay invalid


        for action_idx, (nr, nc) in valid_moves.items():
            neighbor_potential = self._get_potential_at_coord(potential_patch, nr, nc)
            if neighbor_potential < min_potential:
                min_potential = neighbor_potential
                best_action = action_idx
            # Tie-breaking: If potentials are equal, prefer moving over staying, otherwise keep first best found
            elif neighbor_potential == min_potential and best_action == 0 and action_idx != 0:
                 best_action = action_idx


        return best_action

    def _decode_probabilistic(self, potential_patch, valid_moves, r_center, c_center):
        """Sample action based on potential values (lower potential = higher prob)."""
        if not valid_moves: return 0 # Trapped

        actions = list(valid_moves.keys())
        potentials = np.array([self._get_potential_at_coord(potential_patch, *valid_moves[a]) for a in actions])

        # Convert potentials to probabilities using softmax
        # Lower potential should have higher probability -> use negative potential
        # Avoid division by zero if temperature is somehow <= 0
        temp = max(self.temperature, 1e-6)
        scaled_neg_potentials = -potentials / temp

        # Subtract max for numerical stability before exponentiation
        scaled_neg_potentials -= np.max(scaled_neg_potentials)
        exp_potentials = np.exp(scaled_neg_potentials)
        probabilities = exp_potentials / np.sum(exp_potentials)

        # Handle potential NaN probabilities if all exp_potentials are zero (very large potentials)
        if np.isnan(probabilities).any():
             # Fallback: uniform probability over valid moves, or choose gradient action
             logging.warning("NaN encountered in probabilistic sampling, falling back to uniform random choice among valid actions.")
             chosen_action = random.choice(actions)
             # Or fallback to gradient:
             # chosen_action = self._decode_gradient(potential_patch, valid_moves, r_center, c_center)
        else:
            # Sample action based on calculated probabilities
            try:
                 chosen_action = np.random.choice(actions, p=probabilities)
            except ValueError as e:
                 logging.warning(f"ValueError during np.random.choice (maybe probabilities don't sum to 1? Sum={np.sum(probabilities)}): {e}. Falling back to random.")
                 # Fallback to uniform random choice if sampling fails
                 chosen_action = random.choice(actions)


        return chosen_action