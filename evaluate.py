# evaluate.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
import argparse

# Import model and dataset
from unet_model import PotentialFieldUNet
from potential_dataset import PotentialFieldDataset # Use the same dataset class

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(
    model,
    device,
    test_loader,
    criterion
):
    """Evaluates the model on the test dataset."""
    model.to(device)
    model.eval()  # Set model to evaluation mode

    total_loss = 0.0
    total_samples = 0

    logging.info("Starting evaluation...")
    with torch.no_grad(): # Disable gradient calculations
        for batch in test_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0) # Loss per sample
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    logging.info(f"Evaluation Complete:")
    logging.info(f"  Total Samples: {total_samples}")
    logging.info(f"  Average Loss ({criterion.__class__.__name__}): {avg_loss:.6f}")

    return avg_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Potential Field U-Net Model")
    parser.add_argument('--dataset-dir', type=str, required=True, help='Directory containing the preprocessed .npz test data')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--obs-radius', type=int, default=5, help='Observation radius used during preprocessing (for verification)')
    parser.add_argument('--input-channels', type=int, default=4, help='Number of input channels for the model')
    parser.add_argument('--base-channels', type=int, default=64, help='Base number of channels for U-Net (must match trained model)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--loss', type=str, default='MSE', choices=['MSE', 'L1'], help='Loss function used during training (for reporting)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--file-limit', type=int, default=None, help='Limit number of NPZ files loaded (for debugging)')


    args = parser.parse_args()

    # Set device
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")

    # Load test dataset
    logging.info("Loading test dataset...")
    try:
        test_dataset = PotentialFieldDataset(args.dataset_dir, args.obs_radius, file_limit=args.file_limit)
    except (FileNotFoundError, RuntimeError) as e:
        logging.error(f"Failed to load test dataset: {e}")
        exit(1)

    if len(test_dataset) == 0:
         logging.error("Test dataset is empty. Cannot evaluate.")
         exit(1)


    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    logging.info(f"Test DataLoader created with {len(test_dataset)} samples.")

    # Initialize model
    model = PotentialFieldUNet(
        n_channels_in=args.input_channels,
        n_channels_out=1,
        base_c=args.base_channels
    )

    # Load checkpoint
    if not Path(args.checkpoint).is_file():
        logging.error(f"Checkpoint file not found: {args.checkpoint}")
        exit(1)

    logging.info(f"Loading model state from: {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # Handle potential differences in keys ('model_state_dict' vs direct state dict)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
             model.load_state_dict(checkpoint) # Assume checkpoint is just the state dict
        logging.info("Model state loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}")
        exit(1)

    # Initialize loss function (same as used in training for fair comparison)
    if args.loss == 'MSE':
        criterion = nn.MSELoss()
    elif args.loss == 'L1':
        criterion = nn.L1Loss()
    else: # Should not happen with argparse choices
         logging.error(f"Unsupported loss function specified: {args.loss}")
         exit(1)


    # Run evaluation
    evaluate_model(
        model=model,
        device=device,
        test_loader=test_loader,
        criterion=criterion
    )

    logging.info("Evaluation script finished.")