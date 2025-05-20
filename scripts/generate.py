import torch
import matplotlib.pyplot as plt
from model import Generator
import json

net_name = '../models/cdc-gan_walk_big_nz.pkl'

# Latent vector size and number of data channels
nz = 100
nd = 15

device = torch.device("cpu")

# Activity label mapping
activities = {'walk': 0}

# Initialize and load the trained Generator model
netG = Generator(nz)
netG.load_state_dict(torch.load(net_name, weights_only=True))
netG.eval()

# Generate synthetic data for each activity
for seed in range(1):
    # Generate a random seed for reproducibility
    seed = torch.randint(0, 10000, (1,)).item()
    print(f"Seed: {seed}")
    for key in activities.keys():
        torch.manual_seed(seed)
        fixed_noise = torch.randn(1, nz, 1, device=device)
        one_hot_label = torch.zeros(1, len(activities.keys()), device=device)
        one_hot_label[0, activities[key]] = 1

        # Generate fake data using the Generator
        fake = netG(fixed_noise, one_hot_label).detach().cpu()

        # Split generated data into sensor channels
        accel = fake[0][:3]
        gyro = fake[0][3:6]
        mag = fake[0][6:9]
        absOri = fake[0][9:12]
        relOri = fake[0][12:15]

        print(accel.shape)

        # Prepare data in JSON format
        fake_data = {
            "activity": f"{key}",
            "uid": "",
            "elapsedTime": 5000,
            "sensorData": {
                "accelerometer": [{
                    "t": i * 50,
                    "x": accel[0][i].item(),
                    "y": accel[1][i].item(),
                    "z": accel[2][i].item()
                } for i in range(100)],
                "gyroscope": [{
                    "t": i * 50,
                    "x": gyro[0][i].item(),
                    "y": gyro[1][i].item(),
                    "z": gyro[2][i].item()
                } for i in range(100)],
                "magnetometer": [{
                    "t": i * 50,
                    "x": mag[0][i].item(),
                    "y": mag[1][i].item(),
                    "z": mag[2][i].item()
                } for i in range(100)],
                "absOrientation": [{
                    "t": i * 50,
                    "x": absOri[0][i].item(),
                    "y": absOri[1][i].item(),
                    "z": absOri[2][i].item()
                } for i in range(100)],
                "relOrientation": [{
                    "t": i * 50,
                    "x": relOri[0][i].item(),
                    "y": relOri[1][i].item(),
                    "z": relOri[2][i].item()
                } for i in range(100)]
            }
        }

        # Save generated data to JSON file
        with open(f'../data/export/visual_{key}_{seed}.json', 'w') as f:
            json.dump(fake_data, f, indent=4)
