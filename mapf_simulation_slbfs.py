# mapf_simulation.py
import torch
import numpy as np
import time
import logging
import argparse
from pathlib import Path
import yaml

# Import environment, model, helpers
try:
    from pogema_toolbox.create_env import Environment
    from pogema_toolbox.registry import ToolboxRegistry # Needed if using registered maps defined in yamls elsewhere
    from create_env import create_eval_env # Your env creation helper
    # Assuming the helper function for loading maps is available if needed by create_eval_env
    # from potential_dataset import load_and_register_maps # If using registered maps
except ImportError as e:
    logging.error(f"Failed to import POGEMA or related modules: {e}")
    exit(1)
mappath="../MAPF-GPT/dataset_configs/10-medium-mazes"
mappath = "../MAPF-GPT/eval_configs/04-movingai"
for maps_file in Path(mappath).rglob('maps.yaml'):
    with open(maps_file, 'r') as f:
        maps = yaml.safe_load(f)
    ToolboxRegistry.register_maps(maps)
    
# Import your project modules
from unet_model import PotentialFieldUNet
from action_decoder import ActionDecoder # Assuming saved in action_decoder.py
from conflict_resolver import ConflictResolver # Assuming saved in conflict_resolver.py

from slbfs_controller import SLBFSController


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    
def create_dynamic_input_tensor(agent_obs_channels, relative_target_xy, obs_window_size):
    """
    Creates the multi-channel input tensor (6 channels) for a single agent during simulation.
    Includes distant goal direction encoding.

    Args:
        agent_obs_channels (np.ndarray): Agent observation (Channels, H, W).
                                         Assumes Channels 0=Obstacles, 1=Agents.
        relative_target_xy (tuple): (row, col) offset of the target relative to the agent.
        obs_window_size (tuple): (H, W) of the observation window.

    Returns:
        np.ndarray: Input tensor (6, H, W).
          - Ch 0: Obstacles (0/1)
          - Ch 1: Other Agents (0/1)
          - Ch 2: Target Location (one-hot if visible in non-obstacle, else 0)
          - Ch 3: Self Location (one-hot at center)
          - Ch 4: Normalized Target Direction Y (filled if target not visible)
          - Ch 5: Normalized Target Direction X (filled if target not visible)
    """
    obs_H, obs_W = obs_window_size
    # MODIFICATION: Increased channels to 6
    num_input_channels = 6
    input_tensor = np.zeros((num_input_channels, obs_H, obs_W), dtype=np.float32)

    # Channels 0 & 1: Obstacles and Other Agents (as before)
    if agent_obs_channels.shape[0] >= 2:
        input_tensor[0] = agent_obs_channels[0].astype(np.float32)
        input_tensor[1] = agent_obs_channels[1].astype(np.float32)
    elif agent_obs_channels.shape[0] == 1:
        input_tensor[0] = agent_obs_channels[0].astype(np.float32)
        # logging.warning("Observation only has 1 channel (Obstacles?). Other agents channel will be empty.") # Reduced logging noise
    # else: logging.warning("Observation has 0 channels. Obstacle and Other agents channels will be empty.")

    # Channel 3: Self Position (Center) - Place before target check
    center_r, center_c = obs_H // 2, obs_W // 2
    input_tensor[3, center_r, center_c] = 1.0

    # Channels 2, 4, 5: Target Information
    target_r_rel, target_c_rel = relative_target_xy
    target_idx_r = center_r + target_r_rel
    target_idx_c = center_c + target_c_rel

    # Check if target is within observation boundaries
    is_target_visible = (0 <= target_idx_r < obs_H and 0 <= target_idx_c < obs_W)

    if is_target_visible:
        # Target is visible: Use Channel 2 for one-hot encoding if not blocked
        if input_tensor[0, target_idx_r, target_idx_c] == 0:  # Check obstacle channel
            input_tensor[2, target_idx_r, target_idx_c] = 1.0
        # Channels 4 and 5 remain zero when target is visible
    else:
        # Target is NOT visible: Use Channels 4 and 5 for direction vector
        # Channel 2 remains zero
        if target_r_rel == 0 and target_c_rel == 0:
             # Agent is likely already at goal, or goal is invalid? Direction is zero.
             norm_dy, norm_dx = 0.0, 0.0
        else:
            # Calculate normalized direction vector
            magnitude = np.sqrt(target_r_rel**2 + target_c_rel**2)
            # Add epsilon to avoid division by zero if magnitude is tiny (shouldn't happen if not visible)
            epsilon = 1e-6
            norm_dy = target_r_rel / (magnitude + epsilon)
            norm_dx = target_c_rel / (magnitude + epsilon)

        # Fill channels 4 and 5 with normalized direction
        input_tensor[4].fill(norm_dy)
        input_tensor[5].fill(norm_dx)

    return input_tensor

def run_simulation(args):
    """Runs the MAPF simulation using the potential field model."""

    # Set device
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")

    # Load Map Configs if needed by create_eval_env
    # Example:
    # if args.map_config_dir:
    #     from potential_dataset import load_and_register_maps
    #     if not load_and_register_maps(args.map_config_dir):
    #          logging.error(f"Failed to load maps from {args.map_config_dir}. Exiting.")
    #          exit(1)

    # Initialize Environment
    logging.info(f"Initializing environment: map={args.map_name}, agents={args.num_agents}, obs_radius={args.obs_radius}")
    env_cfg = Environment(
        with_animation = True,
        map_name=args.map_name,
        num_agents=args.num_agents,
        obs_radius=args.obs_radius,
        observation_type="POMAPF", # Must match model's expected input channels
        seed=args.seed,
        max_episode_steps=args.max_steps,
        on_target="finish", # Agents finish when reaching target
        collision_system="block_both", # Consistent with preprocessing code example
    )
    # Using create_eval_env might add wrappers, ensure it's compatible
    # If create_eval_env is just `_make_pogema`, it should be fine.
    env = create_eval_env(env_cfg)

    # Initialize Model
    # --- Controller Initialization (MODIFIED) ---
    controller = None
    action_decoder = None # Only needed for LPF

    if args.controller == 'lpf':
        logging.info("Using Learned Potential Field (LPF) controller.")
        # Initialize Model    
        model = PotentialFieldUNet(
            n_channels_in=args.input_channels, # Should be 4
            n_channels_out=1,
            base_c=args.base_channels
        )
        # Load Checkpoint
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_file():
            logging.error(f"Checkpoint file not found: {checkpoint_path}")
            return None
        logging.info(f"Loading model checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            return None
    
        model.to(device)
        model.eval() # Set to evaluation mode
    
        # Initialize Action Decoder and Conflict Resolver
        action_decoder = ActionDecoder(strategy=args.decoder_strategy,temperature=args.temperature)

    elif args.controller == 'slbfs':
        logging.info("Using Step-wise Local BFS (SL-BFS) controller.")
        if not args.obs_radius:
             logging.error("Observation radius (--obs-radius) is required for SL-BFS controller.")
             return None
        controller = SLBFSController(obs_radius=args.obs_radius)
    else:
        # Should not happen with argparse choices
        logging.error(f"Invalid controller type specified: {args.controller}")
        return None
    
    conflict_resolver = ConflictResolver(strategy=args.resolver_strategy)

    # Simulation Loop
    start_time = time.time()
    try:
        obs, info = env.reset(seed=args.seed) # Get initial observations
    except Exception as e:
        logging.error(f"Failed during env.reset(): {e}", exc_info=True)
        return None

    done = [False] * args.num_agents
    step_count = 0
    # Removed collision tracking based on info dict, as it's not provided by default
    # total_collisions_this_ep = 0
    agents_finished_at_step = [-1] * args.num_agents # Track when each agent finishes

    # Get initial positions/goals needed for relative target calculation
    try:
         # Use unwrapped if possible for direct grid access
         global_positions_xy = env.unwrapped.grid.get_agents_xy()
         global_goals_xy = env.unwrapped.grid.get_targets_xy()
    except AttributeError:
         logging.warning("Falling back to potentially wrapped env methods for pos/goal.")
         # Ensure these methods exist in your wrapped env if not using unwrapped
         try:
              global_positions_xy = env.get_agents_xy()
              global_goals_xy = env.get_targets_xy()
         except AttributeError as e:
              logging.error(f"Environment object missing get_agents_xy or get_targets_xy method: {e}")
              env.close()
              return None


    logging.info("Starting simulation loop...")
    while not all(done) and step_count < args.max_steps:
        step_start_time = time.time()
        # Prepare batch of inputs for the model

        valid_agent_indices = [idx for idx, d in enumerate(done) if not d] # Indices of agents still active
        # Use the globally tracked positions for conflict resolution input later
        current_global_positions_for_conflict = [global_positions_xy[idx] for idx in range(args.num_agents)]

        if not valid_agent_indices:
             logging.warning("No valid agents remaining but simulation not finished? Breaking loop.")
             break
        full_intended_actions = [0] * args.num_agents # Default STAY


        #### my method  LPF ####
        if args.controller == 'lpf':
        
            input_tensors_batch = []
            obs_window_shape = (args.obs_radius * 2 + 1, args.obs_radius * 2 + 1)
            current_observations_list = [] # Store observations used for decoding later (only for active agents)
            valid_agent_input_indices = [] # Track indices corresponding to tensors in the batch
    
            for agent_idx in valid_agent_indices:
                 # Ensure observation list/array is indexed correctly
                 if agent_idx >= len(obs):
                      logging.error(f"Agent index {agent_idx} out of bounds for observations list (length {len(obs)}) at step {step_count}. Skipping step.")
                      # This indicates a critical state mismatch, maybe break early?
                      break # Exit inner loop
    
                 agent_obs_data = obs[agent_idx]
                 agent_obs_channels = None
    
                 # Handle dict obs if necessary (like in preprocessing)
                 if isinstance(agent_obs_data, dict):
                      obstacles_ch = agent_obs_data.get("obstacles")
                      agents_ch = agent_obs_data.get("agents")
                      if obstacles_ch is None or agents_ch is None:
                           logging.warning(f"Agent {agent_idx}: Observation dict missing 'obstacles' or 'agents' at step {step_count}. Skipping agent this step.")
                           continue
                      agent_obs_channels = np.stack([obstacles_ch, agents_ch], axis=0)
                 elif isinstance(agent_obs_data, np.ndarray):
                      agent_obs_channels = agent_obs_data
                 else:
                       logging.warning(f"Agent {agent_idx}: Unexpected observation type {type(agent_obs_data)} at step {step_count}. Skipping agent this step.")
                       continue # Skip this agent
    
    
                 # Check shape consistency
                 if agent_obs_channels.shape[1:] != obs_window_shape:
                      logging.warning(f"Agent {agent_idx}: Observation shape mismatch. Expected {obs_window_shape}, Got {agent_obs_channels.shape[1:]} at step {step_count}. Skipping agent this step.")
                      continue # Skip this agent
    
                 # Calculate relative target
                 agent_pos = global_positions_xy[agent_idx] # Use current global position
                 agent_goal = global_goals_xy[agent_idx]
                 relative_target_xy = (agent_goal[0] - agent_pos[0], agent_goal[1] - agent_pos[1])
    
                 # Create input tensor
                 try:
                      input_tensor = create_dynamic_input_tensor(agent_obs_channels, relative_target_xy, obs_window_shape)
                      input_tensors_batch.append(input_tensor)
                      # Store observation channels corresponding to the input tensor for ActionDecoder
                      current_observations_list.append(agent_obs_channels) # Store the raw channels used
                      valid_agent_input_indices.append(agent_idx) # Store original index
                 except Exception as e_create:
                      logging.error(f"Error creating input tensor for agent {agent_idx} at step {step_count}: {e_create}", exc_info=True)
                      # Decide how to handle - skip agent or stop? Skipping for now.
                      continue
    
            # Check if the inner loop finished prematurely
            if len(valid_agent_input_indices) != len(valid_agent_indices):
                logging.warning("Some active agents were skipped during input tensor creation. Proceeding with available agents.")
                # Potentially break here if this is critical
    
            if input_tensors_batch == False:
                 # Check if simulation should actually be done
                 if all(done):
                     logging.info(f"Step {step_count}: All agents are done. Finishing simulation.")
                     break
                 else:
                     logging.warning(f"Step {step_count}: No valid input tensors generated for any active agents. Advancing step with STAY actions.")
                     # Force all agents to stay if none could generate input
                     final_actions = [0] * args.num_agents
                     # Or maybe break here if it indicates an unrecoverable error? Let's try STAY first.
                     # break
            else:
                # Model Inference (Batch)
                input_torch_batch = torch.from_numpy(np.stack(input_tensors_batch, axis=0)).to(device)
                with torch.no_grad():
                    predicted_potentials_batch = model(input_torch_batch) # (N_valid, 1, H, W)
    
                # Process predictions for decoder
                predicted_potentials_np = predicted_potentials_batch.squeeze(1).cpu().numpy() # (N_valid, H, W)
    
                # Decode Actions (for agents that had valid inputs)
                # Pass observations corresponding to the predictions
                intended_agent_actions = action_decoder.decode_actions(predicted_potentials_np, current_observations_list)
    
                # Create full action list (including STAY for done agents and agents skipped during input gen)
                full_intended_actions = [0] * args.num_agents # Default STAY
                for i, original_idx in enumerate(valid_agent_input_indices):
                    full_intended_actions[original_idx] = intended_agent_actions[i]


        elif args.controller == 'slbfs':
            # Compute actions using SL-BFS controller for all agents (it handles active/inactive internally if needed)
            # SL-BFS needs the raw observations list directly
            active_obs = [obs[i] for i in valid_agent_indices]
            active_pos = [global_positions_xy[i] for i in valid_agent_indices]
            active_goals = [global_goals_xy[i] for i in valid_agent_indices]

            # We need to compute for all agents, as SLBFSController doesn't know about 'done' status
            # Pass full observation list, let the controller handle it (or adjust controller if needed)
            # Let's assume controller takes full lists and internally uses active ones if needed.
            # Easier: just pass the full observation list and positions/goals
            intended_actions = controller.compute_actions(obs, global_positions_xy, global_goals_xy)
            # SLBFS controller should return actions for *all* agents, respecting done status implicitly (agent at goal likely stays)
            # We only override actions for agents that are NOT done
            for i in range(args.num_agents):
                 if not done[i]: # Only update actions for agents not done
                      full_intended_actions[i] = intended_actions[i]
                 else: # Ensure done agents stay
                      full_intended_actions[i] = 0



            
        # Resolve Conflicts (using current global positions)
        final_actions = conflict_resolver.resolve_conflicts(
            full_intended_actions,
            current_global_positions_for_conflict # Use the positions from *before* the step
        )

        # Step Environment
        try:
            obs, rewards, terminated, truncated, info = env.step(final_actions)
            step_count += 1
        except Exception as e_step:
             logging.error(f"Error during env.step() at step {step_count}: {e_step}", exc_info=True)
             break # Stop simulation if environment step fails

        # Update state for next iteration
        done = terminated # Use terminated from env
        try: # Update global positions based on env state after step
             # Use unwrapped if possible for direct grid access
             global_positions_xy = env.unwrapped.grid.get_agents_xy()
        except AttributeError:
             # Fallback, ensure method exists in wrapped env
             try:
                 global_positions_xy = env.get_agents_xy()
             except AttributeError as e:
                 logging.error(f"Failed to update agent positions after step {step_count}: {e}. State may be inconsistent.")
                 # Optionally break here if position tracking is critical and fails


        # BUG FIX AREA: The following lines attempting to access info['collisions'] are removed.
        # collisions = info.get('collisions', []) # <-- REMOVED - info is a list
        # if collisions:
        #      total_collisions_this_ep += len(collisions)

        # Update finished times
        for agent_idx in range(args.num_agents):
             if terminated[agent_idx] and agents_finished_at_step[agent_idx] == -1:
                  agents_finished_at_step[agent_idx] = step_count

        step_duration = time.time() - step_start_time
        # Log step info periodically
        if step_count % 50 == 0 or all(done): # Log more frequently or on completion
            active_agents_count = sum(1 for d in done if not d)
            logging.info(f"Step {step_count}/{args.max_steps} | Active Agents: {active_agents_count} | Step Time: {step_duration:.3f}s")


    # Simulation End
    end_time = time.time()
    total_duration = end_time - start_time
    logging.info(f"Simulation finished after {step_count} steps. Duration: {total_duration:.2f} seconds.")
    
    svg_path = f"SVG/SLBFS_{args.map_name}-potential_field-seed-{args.seed}.svg"
    env.save_animation(svg_path)
    ToolboxRegistry.info(f"动画已保存至: {svg_path}")
    
    # Calculate final metrics
    success = all(done) # Check if all agents reached their target state according to env
    isr = sum(1 for d in done if d) / args.num_agents # Individual Success Rate (ISR)
    valid_finish_times = [t for t in agents_finished_at_step if t != -1]
    # Handle cases where no agent finishes:
    if not valid_finish_times: # No agent finished
        sum_of_costs = args.max_steps * args.num_agents
        makespan = args.max_steps
    else:
        # Calculate SOC/Makespan only considering agents that should have finished
        finished_count = sum(1 for d in done if d)
        unfinished_count = args.num_agents - finished_count
        # SOC includes max_steps for agents that didn't finish
        sum_of_costs = sum(valid_finish_times) + unfinished_count * args.max_steps
        # Makespan is the time the last agent finished, or max_steps if not all finished
        makespan = max(valid_finish_times) if success else args.max_steps


    logging.info(f"--- Results for {args.map_name}, {args.num_agents} Agents ---")
    logging.info(f"Success (All Agents Reached Target): {success}")
    logging.info(f"ISR (Individual Success Rate): {isr:.4f} ({sum(1 for d in done if d)}/{args.num_agents})")
    logging.info(f"Makespan: {makespan} steps")
    logging.info(f"Sum of Costs (SOC): {sum_of_costs} steps")
    # Removed collision logging as it's not available from info
    # logging.info(f"Total Collisions Detected (approx): {total_collisions_this_ep}")

    env.close()
    return {"success": success, "isr": isr, "soc": sum_of_costs, "makespan": makespan, "steps": step_count}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MAPF Simulation with Potential Field Model")

    parser.add_argument('--controller', type=str, default='lpf', choices=['lpf', 'slbfs'], help='Controller type: Learned Potential Field (lpf) or Step-wise Local BFS (slbfs)')

    
    # Environment Args
    parser.add_argument('--map-name', type=str, required=True, help='Name of the map registered in POGEMA')
    # parser.add_argument('--map-config-dir', type=str, help='Directory containing map YAMLs (if needed by create_eval_env)') # Optional
    parser.add_argument('--num-agents', type=int, required=True, help='Number of agents')
    parser.add_argument('--obs-radius', type=int, default=5, help='Observation radius for the environment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for the environment')
    parser.add_argument('--max-steps', type=int, default=512, help='Maximum simulation steps')
    # Model Args
    
    parser.add_argument('--checkpoint', type=str, required=False, help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--input-channels', type=int, default=6, help='Number of input channels for the model')
    parser.add_argument('--base-channels', type=int, default=64, help='Base number of channels for U-Net (must match trained model)')
    # Simulation Logic Args
    parser.add_argument('--decoder-strategy', type=str, default='gradient', choices=['gradient', 'probabilistic'], help='Action decoding strategy')
    parser.add_argument('--resolver-strategy', type=str, default='priority', choices=['priority', 'stay', 'random'], help='Conflict resolution strategy')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for probabilistic action decoding (higher = more random)')

    args = parser.parse_args()

    # --- Run Simulation ---
    results = run_simulation(args)

    if results:
         logging.info("Simulation function returned final metrics.")
    else:
         logging.error("Simulation function failed to return results.")

    logging.info("--- Simulation Script Finished ---")