# demo.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import argparse

# Import model and visualization helper (if needed)
from unet_model import PotentialFieldUNet
# Assuming visualize_processed_data exists or we implement a similar one here
# from potential_dataset import visualize_processed_data # Or define locally

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def visualize_demo(input_channels_np, target_potential_np, predicted_potential_np, title="Demo Visualization"):
    """Visualizes input channels, target, and prediction for a single agent."""
    num_channels = input_channels_np.shape[0]
    obs_H, obs_W = input_channels_np.shape[1:]

    fig, axes = plt.subplots(1, num_channels + 2, figsize=(4 * (num_channels + 2), 4))
    fig.suptitle(title, fontsize=16)

    # Channel titles (adjust if your channel setup differs)
    channel_titles = ["Input: Obstacles", "Input: Other Agents", "Input: Target Loc", "Input: Self Loc"]

    # Plot Input Channels
    for i in range(num_channels):
        ax = axes[i]
        img = ax.imshow(input_channels_np[i], cmap='gray', origin='upper')
        ax.set_title(channel_titles[i] if i < len(channel_titles) else f"Input: Channel {i}")
        ax.set_xticks(np.arange(-.5, obs_W, 1), minor=True)
        ax.set_yticks(np.arange(-.5, obs_H, 1), minor=True)
        ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.tick_params(which='minor', size=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.plot(obs_W // 2, obs_H // 2, 'r+', markersize=10) # Center marker
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    # Plot Target Potential Field
    ax = axes[num_channels]
    # Use vmin/vmax based on your normalization range (POTENTIAL_NORM_VALUE)
    norm_val = 10.0 # Assuming POTENTIAL_NORM_VALUE = 10.0 from preprocessing
    img = ax.imshow(target_potential_np, cmap='viridis', origin='upper', vmin=0, vmax=norm_val)
    ax.set_title("Ground Truth Potential")
    ax.set_xticks(np.arange(-.5, obs_W, 1), minor=True)
    ax.set_yticks(np.arange(-.5, obs_H, 1), minor=True)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(which='minor', size=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.plot(obs_W // 2, obs_H // 2, 'r+', markersize=10) # Center marker
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    # Plot Predicted Potential Field
    ax = axes[num_channels + 1]
    img = ax.imshow(predicted_potential_np, cmap='viridis', origin='upper', vmin=0, vmax=norm_val)
    ax.set_title("Predicted Potential")
    ax.set_xticks(np.arange(-.5, obs_W, 1), minor=True)
    ax.set_yticks(np.arange(-.5, obs_H, 1), minor=True)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(which='minor', size=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.plot(obs_W // 2, obs_H // 2, 'r+', markersize=10) # Center marker
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("out.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo Potential Field U-Net Model")
    parser.add_argument('--npz-file', type=str, required=True, help='Path to the specific .npz file for demo')
    parser.add_argument('--agent-index', type=int, default=0, help='Index of the agent within the NPZ file to visualize')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint (.pth file)')
    # Include model parameters necessary to instantiate it correctly
    parser.add_argument('--input-channels', type=int, default=4, help='Number of input channels for the model')
    parser.add_argument('--base-channels', type=int, default=64, help='Base number of channels for U-Net (must match trained model)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')


    args = parser.parse_args()

    # Set device
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")

    # Load data from NPZ file
    npz_path = Path(args.npz_file)
    if not npz_path.is_file():
        logging.error(f"NPZ file not found: {npz_path}")
        exit(1)

    try:
        logging.info(f"Loading data from: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        input_tensors_np = data['input_tensors']
        target_potentials_np = data['target_potentials']
        metadata = data.get('agent_metadata', None) # Optional metadata loading
        map_name = data.get('map_name', 'Unknown Map')
        num_agents_in_file = input_tensors_np.shape[0]
    except Exception as e:
        logging.error(f"Failed to load data from NPZ file {npz_path}: {e}")
        exit(1)


    # Select the specific agent
    agent_idx = args.agent_index
    if not (0 <= agent_idx < num_agents_in_file):
        logging.error(f"Agent index {agent_idx} is out of bounds for file with {num_agents_in_file} agents.")
        exit(1)

    input_np = input_tensors_np[agent_idx]
    target_np = target_potentials_np[agent_idx] # Shape (H, W)
    # Get metadata for title if available
    agent_meta_info = ""
    if metadata is not None and agent_idx < len(metadata):
        agent_meta = metadata[agent_idx]
        agent_meta_info = f" (Agent {agent_meta.get('agent_index', agent_idx)}, Goal: {agent_meta.get('global_goal_xy', 'N/A')})"


    # Initialize model
    model = PotentialFieldUNet(
        n_channels_in=args.input_channels,
        n_channels_out=1,
        base_c=args.base_channels
    )

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        logging.error(f"Checkpoint file not found: {checkpoint_path}")
        exit(1)

    logging.info(f"Loading model state from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle potential differences in keys
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
             model.load_state_dict(checkpoint)
        logging.info("Model state loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}")
        exit(1)

    # Prepare model for inference
    model.to(device)
    model.eval()

    # Prepare input tensor for model
    input_tensor = torch.from_numpy(input_np.astype(np.float32)).unsqueeze(0).to(device) # Add batch dim, move to device

    # Perform inference
    logging.info("Running model inference...")
    with torch.no_grad():
        predicted_output = model(input_tensor) # Shape: (1, 1, H, W)

    # Process output
    predicted_potential_np = predicted_output.squeeze(0).squeeze(0).cpu().numpy() # Remove batch and channel dims, move to CPU, convert to numpy

    # Visualize
    logging.info("Generating visualization...")
    vis_title = f"Demo: '{map_name}'{agent_meta_info}"
    visualize_demo(input_np, target_np, predicted_potential_np, title=vis_title)

    logging.info("Demo script finished.")