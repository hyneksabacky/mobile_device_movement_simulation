import torch
from model import Generator, ND, NA

# Set device to GPU if available, else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model parameters
nz = 100  # Length of noise vector
na = NA   # Number of activity classes
nd = ND   # Number of data channels

# Initialize Generator and load trained weights
netG = Generator(nz).to(device)
net_name = "cdc-gan_walk_small_nz"
netG.load_state_dict(torch.load(f'../models/{net_name}.pkl', weights_only=True))
netG.eval()

# Prepare dummy input for ONNX export
dummy_noise = torch.randn(1, nz, 1).to(device)      # Shape: (batch, nz, 1)
dummy_labels = torch.zeros(1, na, 1).to(device)     # Shape: (batch, na, 1)
dummy_labels[0, 0, 0] = 1  # Example: set first activity to 1 (one-hot)

# Export the model to ONNX format
torch.onnx.export(
    netG,
    (dummy_noise, dummy_labels),
    f"models/{net_name}.onnx",
    input_names=['noise', 'labels'],
    output_names=['output'],
    dynamic_axes={
        'noise': {0: 'batch_size', 2: 'sequence_length'},
        'labels': {0: 'batch_size', 2: 'sequence_length'},
        'output': {0: 'batch_size', 2: 'sequence_length'}
    }
)

print("Trained model exported to ../models/{}.onnx".format(net_name))