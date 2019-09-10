import torch.nn as nn

class PilotNet(nn.Module):

    '''
    PilotNet architecture from NVIDIA
    https://arxiv.org/pdf/1604.07316.pdf
    '''

    def __init__(self, input_shape=(66, 200)):

        super(PilotNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=0),
            nn.ReLU())

        self.layer3 = nn.Sequential(
            nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=0),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU())

        self.layer6 = nn.Sequential(
            nn.Linear(64 * 1 * 18, 100),
            nn.ReLU())

        self.layer7 = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU())

        self.layer8 = nn.Sequential(
            nn.Linear(50, 10),
            nn.ReLU())

        self.layer9 = nn.Sequential(
            nn.Linear(10, 1),
            nn.Tanh())

    def forward(self, x):
        x = (x - 0.5) * 2.0
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = x.view(x.size(0), -1)       # flatten

        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)

        return x
    
