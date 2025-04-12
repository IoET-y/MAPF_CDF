import torch
import torch.nn as nn
import torch.nn.functional as F
import logging # Use logging consistent with preprocessing script

# --- U-Net Building Blocks ---

class DoubleConv(nn.Module):
    """(Convolution => [Batch Norm] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), # Bias False because using BatchNorm
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with MaxPool then DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then DoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # If bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Use ConvTranspose2d for learned upsampling
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels) # Input channels = skip_connection + upsampled

    def forward(self, x1, x2):
        """
        Args:
            x1: Input tensor from the upsampling path (higher resolution).
            x2: Input tensor from the skip connection (lower resolution).
        """
        x1 = self.up(x1)
        # Input is CHW

        # Handle potential size mismatch due to odd dimensions during pooling/upsampling
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Pad x1 to match x2's spatial dimensions
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # If you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e547a391f19575c68fde5e83
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59d5ed319156d52a576980df3b

        x = torch.cat([x2, x1], dim=1) # Concatenate along the channel dimension
        return self.conv(x)

class OutConv(nn.Module):
    """Final 1x1 Convolution to map features to the desired output channels."""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# --- Full U-Net Model ---

class PotentialFieldUNet(nn.Module):
    """
    U-Net model for predicting potential field patches.

    Takes agent-centric observations as input and outputs a potential field patch.
    """
    def __init__(self, n_channels_in, n_channels_out=1, bilinear=True, base_c=64):
        """
        Args:
            n_channels_in (int): Number of input channels (e.g., 4: obstacles, agents, target, self).
            n_channels_out (int): Number of output channels (1 for potential field).
            bilinear (bool): Whether to use bilinear upsampling or ConvTranspose2d.
            base_c (int): Number of channels in the first convolutional layer. Determines network capacity.
        """
        super(PotentialFieldUNet, self).__init__()
        if n_channels_in <= 0:
             raise ValueError("Number of input channels must be positive.")

        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear

        logging.info(f"Initializing PotentialFieldUNet:")
        logging.info(f"  Input Channels: {self.n_channels_in}")
        logging.info(f"  Output Channels: {self.n_channels_out}")
        logging.info(f"  Bilinear Upsampling: {self.bilinear}")
        logging.info(f"  Base Channels: {base_c}")


        self.inc = DoubleConv(n_channels_in, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        factor = 2 if bilinear else 1 # Factor accounts for ConvTranspose halving channels before conv
        self.down3 = Down(base_c * 4, base_c * 8 // factor) # Adjusted for potential ConvTranspose

        # Decoder path
        # Note: Input channels for Up layers = channels from below + channels from skip connection
        self.up1 = Up(base_c * 8, base_c * 4 // factor, bilinear) # Input: (base_c*8) + (base_c*4)
        self.up2 = Up(base_c * 4, base_c * 2 // factor, bilinear) # Input: (base_c*4) + (base_c*2)
        self.up3 = Up(base_c * 2, base_c, bilinear)              # Input: (base_c*2) + (base_c)
        self.outc = OutConv(base_c, n_channels_out)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (BatchSize, n_channels_in, H, W).
                              H and W should match OBS_WINDOW_SIZE from preprocessing.

        Returns:
            torch.Tensor: Output potential field patch of shape (BatchSize, n_channels_out, H, W).
                          Values are raw outputs (linear activation).
        """
        # Encoder
        x1 = self.inc(x)    # -> base_c channels
        x2 = self.down1(x1) # -> base_c*2 channels
        x3 = self.down2(x2) # -> base_c*4 channels
        x4 = self.down3(x3) # -> base_c*8 // factor channels (bottleneck)

        # Decoder with Skip Connections
        x = self.up1(x4, x3) # Upsample x4, concat with x3 -> base_c*4 // factor channels
        x = self.up2(x, x2)  # Upsample, concat with x2 -> base_c*2 // factor channels
        x = self.up3(x, x1)  # Upsample, concat with x1 -> base_c channels
        logits = self.outc(x) # Final 1x1 conv -> n_channels_out channels

        # No final activation here - raw logits are outputted.
        # Loss function (e.g., MSELoss) will compare these directly to the normalized target potential fields.
        # If a specific range or distribution was desired (e.g., strict positive), an activation could be added.
        return logits

# --- Example Usage (for verification) ---
if __name__ == '__main__':
    # Configuration matching the preprocessor output
    OBS_RADIUS_EXAMPLE = 5
    OBS_H = OBS_W = OBS_RADIUS_EXAMPLE * 2 + 1 # e.g., 11
    INPUT_CHANNELS = 4 # Obstacles, Other Agents, Target Loc, Self Loc
    OUTPUT_CHANNELS = 1 # Potential field
    BATCH_SIZE = 8

    # Create dummy input data matching the expected dimensions
    dummy_input = torch.randn(BATCH_SIZE, INPUT_CHANNELS, OBS_H, OBS_W)

    # Instantiate the model
    # Use smaller base_c for quick testing if needed
    model = PotentialFieldUNet(n_channels_in=INPUT_CHANNELS, n_channels_out=OUTPUT_CHANNELS, base_c=32)

    logging.info(f"Model instantiated: {model.__class__.__name__}")
    print(f"Model structure:\n{model}") # Print model layers

    # Perform a forward pass
    with torch.no_grad():
        output = model(dummy_input)

    logging.info(f"Input shape: {dummy_input.shape}")
    logging.info(f"Output shape: {output.shape}")

    # Verify output shape
    assert output.shape == (BATCH_SIZE, OUTPUT_CHANNELS, OBS_H, OBS_W), \
        f"Output shape mismatch! Expected {(BATCH_SIZE, OUTPUT_CHANNELS, OBS_H, OBS_W)}, Got {output.shape}"

    logging.info("Model forward pass successful and output shape is correct.")

    # --- Calculate Number of Parameters ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total model parameters: {total_params:,}")
    logging.info(f"Trainable model parameters: {trainable_params:,}")