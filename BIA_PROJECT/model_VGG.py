import torch.nn as nn


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),
            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            # (224,224,64) -> (224,224,64)
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # (224,224,64) -> (112,112,64)
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.layer2 = nn.Sequential(
            #(112,112,64) -> (112,112,128)
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # (112,112,128) -> (112,112,128)
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # (112,112,128) -> (56,56,128)
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.layer3 = nn.Sequential(
            # (56,56,128) -> (56,56,256)
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # (56,56,256) -> (56,56,256)
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # (56,56,256) -> (56,56,256)
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # (56,56,256) -> (28,28,256)
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.layer4=nn.Sequential(
            # (28,28,256) -> (28,28,512)
            nn.Conv2d(256,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # (28,28,512) -> (28,28,512)
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # (28,28,512) -> (28,28,512)
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # (28,28,512) -> (14,14,512)
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.layer5=nn.Sequential(
            # 14,14,512 -> 14,14,512
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 14,14,512 -> 14,14,512
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 14,14,512 --> 14,14,512
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 14,14,512 --> 7,7,512
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.conv_layer = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

        self.fc = nn.Sequential(
            # (1,1,25088) -> (1,1,4096)
            nn.Linear(25088,4096),
            nn.ReLU(inplace=1),
            nn.Dropout(),
            # (1,1,4096) -> (1,1,4096)
            nn.Linear(4096,4096),
            nn.ReLU(inplace=1),
            nn.Dropout(),
            # (1,1,4096) -> (1,1,1000)
            nn.Linear(4096,1000),
            nn.ReLU(inplace=1),
            nn.Dropout(),
            nn.Linear(1000, 2),
        )

    def forward(self,x):
        x = self.conv_layer(x)
        # 7,7,512 -> 1,1,25088
        x=x.view(-1,25088)
        x=self.fc(x)
        return x
