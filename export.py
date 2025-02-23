import torch
from model import Generator, nd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nz = 100  # length of noise
na = 3

netG = Generator(nz).to(device)

netG.load_state_dict(torch.load('nets/dcgan_netG_more_activities.pkl', weights_only=True))
netG.to(device)
netG.eval()

# Prepare a dummy input for ONNX export
dummy_noise = torch.randn(1, nz, 1).to(device)
dummy_labels = torch.zeros(1, na, 1).to(device)
dummy_labels[0, 0, 0] = 1  # Example: setting the first activity to 1 (one-hot encoding)

# Concatenate noise and labels
dummy_input = torch.cat([dummy_noise, dummy_labels], 1)

# Export the loaded model to ONNX format
torch.onnx.export(
    netG,
    (dummy_noise, dummy_labels),
    "gan_activities.onnx",
    input_names=['noise', 'labels'],
    output_names=['output'],
    dynamic_axes={'noise': {0: 'batch_size', 2: 'sequence_length'},
                  'labels': {0: 'batch_size', 2: 'sequence_length'},
                  'output': {0: 'batch_size', 2: 'sequence_length'}}
)

print("Trained model exported to gan_activities.onnx")