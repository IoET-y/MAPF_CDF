# train.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter # For logging
import numpy as np
from pathlib import Path
import logging
import argparse
import time
import os

# Import model and dataset
from unet_model import PotentialFieldUNet
from potential_dataset import PotentialFieldDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(
    model,
    device,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    epochs,
    output_dir,
    checkpoint_freq=1,
    start_epoch=0
):
    """Main training loop."""
    writer = SummaryWriter(log_dir=Path(output_dir) / 'logs') # TensorBoard logs
    model.to(device)
    best_val_loss = float('inf')
    global_step = start_epoch * len(train_loader) # Adjust global step if resuming

    logging.info(f"Starting training from epoch {start_epoch+1}/{epochs}")

    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        processed_samples = 0

        for i, batch in enumerate(train_loader):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            # Gradient clipping (optional, but can help stability)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0) # Loss per sample
            processed_samples += inputs.size(0)
            global_step += 1

            # Log training loss periodically
            if (i + 1) % 100 == 0: # Log every 100 batches
                avg_loss = running_loss / processed_samples
                logging.info(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {avg_loss:.6f}')
                writer.add_scalar('Loss/train_step', avg_loss, global_step)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        writer.add_scalar('Loss/train_epoch', epoch_train_loss, epoch + 1)
        logging.info(f'End of Epoch [{epoch+1}/{epochs}] - Training Loss: {epoch_train_loss:.6f}')

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        writer.add_scalar('Loss/validation_epoch', epoch_val_loss, epoch + 1)
        logging.info(f'End of Epoch [{epoch+1}/{epochs}] - Validation Loss: {epoch_val_loss:.6f}')

        epoch_duration = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch+1} duration: {epoch_duration:.2f} seconds")

        # --- Checkpoint Saving ---
        # Save checkpoint every `checkpoint_freq` epochs
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = Path(output_dir) / 'checkpoints' / f'checkpoint_epoch_{epoch+1}.pth'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_train_loss, # Store training loss
                'val_loss': epoch_val_loss,
            }, checkpoint_path)
            logging.info(f'Checkpoint saved to {checkpoint_path}')

        # Save the best model based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_path = Path(output_dir) / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), # Save optimizer state too
                'val_loss': best_val_loss,
            }, best_model_path)
            logging.info(f'New best model saved to {best_model_path} (Val Loss: {best_val_loss:.6f})')

    writer.close()
    logging.info('Finished Training')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Potential Field U-Net Model")
    parser.add_argument('--dataset-dir', type=str, required=True, help='Directory containing the preprocessed .npz training data')
    parser.add_argument('--output-dir', type=str, default='training_output', help='Directory to save checkpoints and logs')
    parser.add_argument('--val-split', type=float, default=0.1, help='Fraction of data to use for validation (e.g., 0.1 for 10%)')
    parser.add_argument('--obs-radius', type=int, default=5, help='Observation radius used during preprocessing (for verification)')
    parser.add_argument('--input-channels', type=int, default=6, help='Number of input channels for the model')
    parser.add_argument('--base-channels', type=int, default=64, help='Base number of channels for U-Net')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'], help='Optimizer type')
    parser.add_argument('--loss', type=str, default='MSE', choices=['MSE', 'L1'], help='Loss function (MSELoss or L1Loss)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--checkpoint-freq', type=int, default=5, help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--resume-checkpoint', type=str, default=None, help='Path to checkpoint file to resume training from')
    parser.add_argument('--file-limit', type=int, default=None, help='Limit number of NPZ files loaded (for debugging)')

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir) / 'checkpoints' # Ensure checkpoints subdir exists

    # Set device
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")

    # Load dataset
    logging.info("Loading dataset...")
    try:
        full_dataset = PotentialFieldDataset(args.dataset_dir, args.obs_radius, file_limit=args.file_limit)
    except (FileNotFoundError, RuntimeError) as e:
        logging.error(f"Failed to load dataset: {e}")
        exit(1)

    # Split dataset into training and validation
    val_size = int(args.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    if val_size == 0 and len(full_dataset) > 0: # Ensure there's at least one validation sample if possible
        logging.warning("Validation split size is zero. Consider increasing val_split or dataset size.")
        # Adjust slightly if needed, e.g., val_size = 1; train_size = len(full_dataset) - 1
        if len(full_dataset) > 1 :
             val_size = max(1, int(len(full_dataset)*0.01)) # Use 1% or at least 1 sample
             train_size = len(full_dataset) - val_size

    if train_size <= 0 or val_size <= 0:
        logging.error(f"Cannot split dataset with train_size={train_size} and val_size={val_size}. Check dataset and val_split.")
        exit(1)

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    logging.info(f"Dataset split: Train={len(train_dataset)}, Validation={len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True if args.device == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True if args.device == 'cuda' else False)
    logging.info("DataLoaders created.")

    # Initialize model
    model = PotentialFieldUNet(
        n_channels_in=args.input_channels,
        n_channels_out=1, # Output is single channel potential field
        base_c=args.base_channels
    )

    # Initialize optimizer
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-8) # Small weight decay for AdamW
    logging.info(f"Optimizer: {args.optimizer}, Learning Rate: {args.lr}")

    # Initialize loss function
    if args.loss == 'MSE':
        criterion = nn.MSELoss()
    elif args.loss == 'L1':
        criterion = nn.L1Loss() # MAE
    logging.info(f"Loss Function: {args.loss}")

    # Handle resuming from checkpoint
    start_epoch = 0
    if args.resume_checkpoint:
        if Path(args.resume_checkpoint).is_file():
            logging.info(f"Resuming training from checkpoint: {args.resume_checkpoint}")
            checkpoint = torch.load(args.resume_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', -1) + 1 # Get epoch number, default to start from 0 if not found
            # Load best_val_loss if available? Or recalculate? Let's recalculate best loss.
            logging.info(f"Resuming from epoch {start_epoch}")
            # Clear memory
            del checkpoint
            torch.cuda.empty_cache() if device == 'cuda' else None
        else:
            logging.warning(f"Resume checkpoint not found: {args.resume_checkpoint}. Starting from scratch.")


    # Start training
    train_model(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=args.epochs,
        output_dir=args.output_dir,
        checkpoint_freq=args.checkpoint_freq,
        start_epoch=start_epoch
    )