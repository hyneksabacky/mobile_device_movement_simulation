import torch
import sys
from model import Generator, ND, NA

def main(net_name, activity):
    """
    Main function to export the trained Generator model to ONNX format.
    """
    # Set device to GPU if available, else CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model parameters
    if activity == 'car':
        nz = 10  # Length of noise vector
    elif activity == 'walk':
        nz = 100  # Length of noise vector
    na = NA   # Number of activity classes
    nd = ND   # Number of data channels

    # Initialize Generator and load trained weights
    netG = Generator(nz).to(device)
    netG.load_state_dict(torch.load(net_name, weights_only=True))
    netG.eval()

    # Prepare dummy input for ONNX export
    dummy_noise = torch.randn(1, nz, 1).to(device)      # Shape: (batch, nz, 1)
    dummy_labels = torch.zeros(1, na, 1).to(device)     # Shape: (batch, na, 1)
    dummy_labels[0, 0, 0] = 1  # Example: set first activity to 1 (one-hot)

    net_name = net_name.split('/')[-1].split('.')[0]  # Extract model name from path

    # Export the model to ONNX format
    torch.onnx.export(
        netG,
        (dummy_noise, dummy_labels),
        f"../models/{net_name}.onnx",
        input_names=['noise', 'labels'],
        output_names=['output'],
        dynamic_axes={
            'noise': {0: 'batch_size', 2: 'sequence_length'},
            'labels': {0: 'batch_size', 2: 'sequence_length'},
            'output': {0: 'batch_size', 2: 'sequence_length'}
        }
    )

    print("Trained model exported to ../models/{}.onnx".format(net_name))

def print_help():
    """
    Print help message for using the script.
    """
    print("Usage: python export.py <activity> <net_name>")
    print("net_name: Path to the .pkl model file")
    print("activity: Name of the activity: walk/car")

if __name__ == "__main__":
    net_name = ''
    activity = 'walk'
    if len(sys.argv) == 3:
        activity = sys.argv[1]
        net_name = sys.argv[2]
    else:
        print_help()
        sys.exit(1)
    main(net_name, activity)
    