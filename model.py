import torch.nn as nn
import torch
import sys

ND = 15
NA = 4

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self, nd=ND, na=NA):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(nd + na, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(512, 1, kernel_size=6, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    # def forward(self, x, y):
    #     print("DISCRIMINATOR FORWARD PASS")
    #     print(f"Input x shape: {x.shape}")  # Debug input shape
    #     print(f"Input y shape: {y.shape}")  # Debug label shape

    #     y = y.view(y.size(0), y.size(1), 1).expand(-1, -1, x.size(2))  # Expand labels to match input size
    #     print(f"Expanded y shape: {y.shape}")  # Debug expanded label shape

    #     x = torch.cat([x, y], 1)  # Concatenate input and labels along the channel dimension
    #     print(f"Concatenated x shape: {x.shape}")  # Debug concatenated input shape

    #     for i, layer in enumerate(self.main):
    #         x = layer(x)
    #         print(f"After layer {i} ({layer.__class__.__name__}): {x.shape}")  # Debug shape after each layer

    #     return x
    def forward(self, x, y):
        y = y.view(y.size(0), y.size(1), 1).expand(-1, -1, x.size(2))
        x = torch.cat([x, y], 1)
        x = self.main(x)
        return x


class Generator(nn.Module):
    def __init__(self, nz, nd=ND, na=NA):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(nz + na, 512, kernel_size=16, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.ConvTranspose1d(256, 128, kernel_size=6, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, nd, kernel_size=16, stride=2, padding=1, bias=False),
        )

    # def forward(self, x, y):
    #     print("GENERATOR FORWARD PASS")
    #     print(f"Input x shape: {x.shape}")
    #     print(f"Input y shape: {y.shape}")

    #     y = y.view(y.size(0), y.size(1), 1).expand(-1, -1, x.size(2))
    #     print(f"Expanded y shape: {y.shape}")
    #     x = torch.cat([x, y], 1)
    #     print(f"Concatenated x shape: {x.shape}")
    #     for i, layer in enumerate(self.main):
    #         x = layer(x)
    #         print(f"After layer {i} ({layer.__class__.__name__}): {x.shape}")  # Debug shape after each layer

    #     # sys.exit(0)
    #     return x
    def forward(self, x, y):
        y = y.view(y.size(0), y.size(1), 1).expand(-1, -1, x.size(2))
        x = torch.cat([x, y], 1)
        x = self.main(x)
        return x
