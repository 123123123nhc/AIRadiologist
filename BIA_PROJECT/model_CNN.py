import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 128 * 128, 120)
        self.fc2 = nn.Linear(120, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        # initial size (3, 512, 512)
        x = self.pool(F.relu(self.conv1(x))) # (32,512,512) ->  (32,256,256)
        x = self.pool(F.relu(self.conv2(x))) # (64, 256,256) ->  (64,128,128)
        x = x.view(-1, 64 * 128 * 128)
        x = F.relu(self.fc1(x)) # (64 * 128 * 128 -> 120)
        x = F.relu(self.fc2(x)) # 120 -> 32
        x = self.fc3(x) # 32 -> 2
        return x