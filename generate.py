import torch
import matplotlib.pyplot as plt
from model import Generator

# seed = 420
nz = 100
nd = 15

device = torch.device("cpu")

activities = {'walk' : 0, 'sit' : 1, 'car' : 2, 'ontable' : 3}

netG = Generator(nz)
netG.load_state_dict(torch.load('./models/cdc-gan_4-act.pkl', weights_only=True))
netG.eval()

for seed in range(100):
    for key in activities.keys():
        torch.manual_seed(seed)
        fixed_noise = torch.randn(1, nz, 1, device=device)
        one_hot_label = torch.zeros(1, 4, device=device)
        one_hot_label[0, activities[key]] = 1

        fake = netG(fixed_noise, one_hot_label).detach().cpu()

        accel = fake[0][:3]
        gyro = fake[0][3:6]
        mag = fake[0][6:9]
        absOri = fake[0][9:12]
        relOri = fake[0][12:15]

        print(accel.shape)

        # # save as json
        import json

        fake_data = {
            "activity": f"{key}",
            "uid" : "",
            "elapsedTime" : 5000,
            "sensorData" : {
                "accelerometer" : [{
                    "t" : i * 50,
                    "x" : accel[0][i].item(),
                    "y" : accel[1][i].item(),
                    "z" : accel[2][i].item()
                }
                for i in range(100)],
                "gyroscope" : [{
                    "t" : i * 50,
                    "x" : gyro[0][i].item(),
                    "y" : gyro[1][i].item(),
                    "z" : gyro[2][i].item()
                }
                for i in range(100)],
                "magnetometer" : [{
                    "t" : i * 50,
                    "x" : mag[0][i].item(),
                    "y" : mag[1][i].item(),
                    "z" : mag[2][i].item()
                }
                for i in range(100)],
                "absOrientation" : [{
                    "t" : i * 50,
                    "x" : absOri[0][i].item(),
                    "y" : absOri[1][i].item(),
                    "z" : absOri[2][i].item()
                }
                for i in range(100)],
                "relOrientation" : [{
                    "t" : i * 50,
                    "x" : relOri[0][i].item(),
                    "y" : relOri[1][i].item(),
                    "z" : relOri[2][i].item()
                }
                for i in range(100)]
            }
        }

        with open(f'data/export/{key}_{seed}.json', 'w') as f:
            json.dump(fake_data, f, indent=4)
