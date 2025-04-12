# conflict_resolver.py
import logging
import random

class ConflictResolver:
    """
    Resolves conflicts between intended agent actions.
    Operates on GLOBAL coordinates.
    """
    def __init__(self, strategy='priority'):
        """
        Args:
            strategy (str): 'priority', 'stay', 'random'.
        """
        if strategy not in ['priority', 'stay', 'random']:
            raise ValueError("Strategy must be 'priority', 'stay', or 'random'")
        self.strategy = strategy
        # Action mapping (same as decoder)
        self.action_delta_map = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        logging.info(f"ConflictResolver initialized with strategy: {self.strategy}")

    def resolve_conflicts(self, intended_actions, current_positions_xy):
        """
        Resolves conflicts based on intended actions and current global positions.

        Args:
            intended_actions (list): List of intended actions from ActionDecoder.
            current_positions_xy (list): List of current (row, col) global positions.

        Returns:
            list: List of final, conflict-free actions.
        """
        num_agents = len(intended_actions)
        if num_agents <= 1:
            return intended_actions # No conflicts possible

        # Use a copy to modify actions during resolution
        final_actions = list(intended_actions)

        # --- Resolution Loop ---
        # Loop multiple times might be needed for cascading conflicts, but simple strategies often converge fast.
        # A single pass might suffice for priority/random/stay, but let's do a fixed number of passes or until no changes.
        max_passes = num_agents # Heuristic limit
        for _pass in range(max_passes):
            conflicts_resolved_this_pass = 0

            # 1. Calculate intended next positions based on *current* final_actions
            intended_next_pos = {} # agent_idx -> next_global_pos
            for i in range(num_agents):
                action = final_actions[i] # Use potentially modified action
                dr, dc = self.action_delta_map[action]
                cr, cc = current_positions_xy[i]
                intended_next_pos[i] = (cr + dr, cc + dc)

            # --- Resolve Vertex Conflicts ---
            vertex_targets = {} # target_pos -> list of agent indices wanting to move there
            for i in range(num_agents):
                target = intended_next_pos[i]
                # Agents staying at their current spot don't cause vertex conflict *at that spot*
                # unless another agent tries to move there. We only care about multiple agents targeting the *same new spot*.
                if target == current_positions_xy[i]: continue

                if target not in vertex_targets:
                    vertex_targets[target] = []
                vertex_targets[target].append(i)

            for target_pos, agents in vertex_targets.items():
                if len(agents) > 1: # Vertex conflict detected!
                    #logging.debug(f"Pass {_pass + 1}: Vertex conflict at {target_pos} for agents: {agents}")
                    if self.strategy == 'priority':
                        agents.sort() # Sort by index (lower index = higher priority)
                        winner = agents[0]
                        for loser in agents[1:]:
                            if final_actions[loser] != 0:
                                final_actions[loser] = 0 # Stay
                                conflicts_resolved_this_pass += 1
                    elif self.strategy == 'random':
                        winner = random.choice(agents)
                        for loser in agents:
                            if loser != winner and final_actions[loser] != 0:
                                final_actions[loser] = 0 # Stay
                                conflicts_resolved_this_pass += 1
                    elif self.strategy == 'stay': # Default 'stay'
                        for agent_idx in agents:
                            if final_actions[agent_idx] != 0:
                                final_actions[agent_idx] = 0 # Stay
                                conflicts_resolved_this_pass += 1

            # --- Resolve Edge (Swap) Conflicts ---
            # Recalculate intended positions based on actions potentially modified by vertex resolution
            for i in range(num_agents):
                action = final_actions[i]
                dr, dc = self.action_delta_map[action]
                cr, cc = current_positions_xy[i]
                intended_next_pos[i] = (cr + dr, cc + dc)

            # Check pairs for swaps
            resolved_swap_pair = set() # Track pairs to avoid resolving twice
            for i in range(num_agents):
                # Skip if agent i is staying or the pair was already handled
                if final_actions[i] == 0 or i in resolved_swap_pair: continue

                for j in range(i + 1, num_agents):
                     # Skip if agent j is staying or the pair was already handled
                     if final_actions[j] == 0 or j in resolved_swap_pair: continue

                     # Check for swap: i wants j's spot AND j wants i's spot
                     if intended_next_pos[i] == current_positions_xy[j] and \
                        intended_next_pos[j] == current_positions_xy[i]:

                         #logging.debug(f"Pass {_pass + 1}: Edge conflict between Agent {i} at {current_positions_xy[i]} and Agent {j} at {current_positions_xy[j]}")
                         conflicts_resolved_this_pass += 1 # Count resolution attempt
                         resolved_swap_pair.add(i) # Mark both as handled for this pass
                         resolved_swap_pair.add(j)

                         if self.strategy == 'priority':
                             # Higher priority (i) moves, lower priority (j) stays
                             final_actions[j] = 0
                         elif self.strategy == 'random':
                             # Randomly pick one to stay
                             loser = random.choice([i, j])
                             final_actions[loser] = 0
                         elif self.strategy == 'stay': # Default 'stay'
                             # Both stay
                             final_actions[i] = 0
                             final_actions[j] = 0
                         break # Move to next i once a swap involving i is handled


            # If no conflicts were resolved in this pass, the state is stable
            if conflicts_resolved_this_pass == 0:
                # logging.debug(f"Resolution converged after pass {_pass + 1}")
                break
        # End resolution loop

        # Final sanity check (optional but recommended) - can be removed if performance critical
        final_pos_check = {}
        for i in range(num_agents):
            action = final_actions[i]
            dr, dc = self.action_delta_map[action]
            cr, cc = current_positions_xy[i]
            final_pos = (cr + dr, cc + dc)
            if final_pos in final_pos_check:
                 original_agent = final_pos_check[final_pos]
                 logging.error(f"Conflict resolution failed! Agent {i} and Agent {original_agent} both ended at {final_pos} after resolution. Forcing agent {i} to STAY.")
                 final_actions[i] = 0 # Force stay as last resort
            else:
                 final_pos_check[final_pos] = i


        return final_actions