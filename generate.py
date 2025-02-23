import torch
import matplotlib.pyplot as plt
from model import Generator

seed = 420
nz = 100

device = torch.device("cpu")

netG = Generator(nz)
netG.load_state_dict(torch.load('./nets/dcgan_netG_more_activities.pkl', weights_only=True))
netG.eval()

torch.manual_seed(seed)
fixed_noise = torch.randn(1, nz, 1, device=device)
one_hot_label = torch.zeros(1, 3, device=device)
one_hot_label[0, 1] = 1

fake = netG(fixed_noise, one_hot_label).detach().cpu()

plot_labels = ["Accelerometer X", "Accelerometer Y", "Accelerometer Z", "Gyroscope X", "Gyroscope Y", "Gyroscope Z"]

f, a = plt.subplots(6)
for i in range(6):
    a[i].plot(fake[0][i].view(-1))
    a[i].set_title(f'{plot_labels[i]}')

plt.show()