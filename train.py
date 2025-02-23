import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

from model import Discriminator, Generator, weights_init
from prepare import Dataset

lr = 2e-4
beta1 = 0.5
epoch_num = 128
batch_size = 128
nz = 100
ngpu = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nd = 6

activities = {'"walk"' : 0, '"sit"' : 1, '"car"' : 2}

def main():
    trainset = Dataset('./data/preprocessed/data_xyz.h5', activities)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )

    netD = Discriminator().to(device)
    netD.apply(weights_init)

    netG = Generator(nz).to(device)
    netG.apply(weights_init)

    criterion = nn.BCELoss()

    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    G_losses = []
    D_losses = []

    loop = tqdm(range(epoch_num), total=epoch_num, leave=False)
    for epoch in loop:
        for step, (data, labels) in enumerate(trainloader):
            labels = labels.to(device)
            labels_one_hot = torch.zeros(labels.size(0), len(activities.keys()), device=device)
            labels_one_hot.scatter_(1, labels.view(-1, 1), 1)

            real_cpu = data.to(device)
            b_size = real_cpu.size(0)

            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            netD.zero_grad()
            output = netD(real_cpu, labels_one_hot).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(b_size, nz, 1, device=device)
            fake = netG(noise, labels_one_hot)
            label.fill_(fake_label)
            output = netD(fake.detach(), labels_one_hot).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)

            output = netD(fake, labels_one_hot).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            
            G_losses.append(errG.item())
            D_losses.append(errD.item())

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    torch.save(netG.state_dict(), './models/cdc-gan.pkl')
    torch.save(netD, './models/dcgan_netD.pkl')

if __name__ == '__main__':
    main()