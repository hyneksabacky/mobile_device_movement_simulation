import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import Discriminator, Generator, weights_init
from prepare import Dataset

# Hyperparameters
lr = 2e-4
beta1 = 0.5
epoch_num = 4096
batch_size = 128
nz = 100
ngpu = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nd = 15

# Activity label mapping
activities = {'walk' : 0}

def main():
    print("Loading dataset...")
    # Load the dataset
    trainset = Dataset('../data/preprocessed/marekstraka_xyz.h5', activities)

    # Create DataLoader for batching
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False
    )

    # Initialize Discriminator and Generator
    netD = Discriminator().to(device)
    netD.apply(weights_init)

    netG = Generator(nz).to(device)
    netG.apply(weights_init)

    # Loss function
    criterion = nn.BCELoss()

    # Labels for real and fake data
    real_label = 0.9
    fake_label = 0.

    # Optimizers for Discriminator and Generator
    optimizerD = optim.Adam(netD.parameters(), lr=lr/2, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr*32, betas=(beta1, 0.999))

    # Lists to store losses for plotting
    G_losses = []
    D_losses = []

    # Training loop
    loop = tqdm(range(epoch_num), total=epoch_num, leave=False)
    for epoch in loop:
        for step, (data, labels) in enumerate(trainloader):
            # Prepare labels and one-hot encoding
            labels = labels.to(device)
            labels_one_hot = torch.zeros(labels.size(0), len(activities.keys()), device=device)
            labels_one_hot.scatter_(1, labels.view(-1, 1), 1)

            # Get real data batch
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)

            # Train Discriminator with real data
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            netD.zero_grad()
            output = netD(real_cpu, labels_one_hot).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            # Generate fake data and train Discriminator with it
            noise = torch.randn(b_size, nz, 1, device=device)
            fake = netG(noise, labels_one_hot)
            label.fill_(fake_label)
            output = netD(fake.detach(), labels_one_hot).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Train Generator
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake, labels_one_hot).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            # Store losses
            G_losses.append(errG.item())
            D_losses.append(errD.item())

    # Calculate iterations per epoch for plotting
    num_iterations = len(D_losses)
    iterations_per_epoch = num_iterations // epoch_num
    epochs = [i / iterations_per_epoch for i in range(num_iterations)]

    # Save the trained Generator model
    torch.save(netG.state_dict(), '../models/cdc-gan_walk_big_nz_huge_e.pkl')

    # Plot losses
    _, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(D_losses, label="Discriminator")
    ax1.plot(G_losses, label="Generator")
    ax1.set_xlabel("Iterations", fontsize=14)
    ax1.set_ylabel("Loss", fontsize=14)
    ax1.legend(loc='upper right', fontsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)

    # Add secondary x-axis for epochs
    def iterations_to_epochs(x):
        return x / iterations_per_epoch

    def epochs_to_iterations(x):
        return x * iterations_per_epoch

    ax2 = ax1.secondary_xaxis('top', functions=(iterations_to_epochs, epochs_to_iterations))
    
    # Set epoch ticks
    ticks_with_end = [x for x in range(0, epoch_num, 256)]
    ticks_with_end.append(epoch_num)
    print(ticks_with_end)

    ax2.set_xticks(ticks_with_end)
    ax2.set_xlabel("Epochs", fontsize=14)
    ax2.tick_params(axis='x', labelsize=14)

    # Draw vertical lines for every 256 epochs
    for i in range(0, num_iterations + iterations_per_epoch, iterations_per_epoch * 256):
        plt.axvline(x=i, color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('../models/losses.png', dpi=300)
    
if __name__ == '__main__':
    main()