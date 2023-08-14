import torch.nn as nn

class Generator(nn.Module) :
    def __init__(self) :
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 120), 
            nn.ReLU(),
            nn.Linear(120, 784),
            nn.Tanh()
        )

    def forward(self, x) :
        return self.model(x)

class Discriminator(nn.Module) :
    def __init__(self) :
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128), 
            nn.LeakyReLU(0.2), # 0.1 ~ 0.3
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x) :
        return self.model(x)