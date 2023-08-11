import torch.nn as nn

class AutoEncoder(nn.Module) :
    def __init__(self) :
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # input image size 28 x 28 = 784
            nn.Linear(784,128), # 데이터의 중요한 특성을 추출
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32,64), 
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,784),
            nn.Tanh()
        )

    def forward(self, x) :
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded