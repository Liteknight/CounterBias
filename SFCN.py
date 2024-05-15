from torch import nn
import torch


class SFCNModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,padding='same')
        self.norm1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.relu1 = nn.ReLU()
        self.block1 = nn.Sequential(self.conv1, self.norm1, self.maxpool1, self.relu1)

        # Block 2
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,padding='same')
        self.norm2 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.relu2 = nn.ReLU()
        self.block2 = nn.Sequential(self.conv2, self.norm2, self.maxpool2, self.relu2)
        
        # Block 3
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,padding='same')
        self.norm3 = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.relu3 = nn.ReLU()
        self.block3 = nn.Sequential(self.conv3, self.norm3, self.maxpool3, self.relu3)

        # Block 4
        self.conv4 = nn.Conv2d(128,256,kernel_size=3,padding='same')
        self.norm4 = nn.BatchNorm2d(256)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.relu4 = nn.ReLU()
        self.block4 = nn.Sequential(self.conv4, self.norm4, self.maxpool4, self.relu4)

        # Block 5
        self.conv5 = nn.Conv2d(256,256,kernel_size=3,padding='same')
        self.norm5 = nn.BatchNorm2d(256)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.relu5 = nn.ReLU()
        self.block5 = nn.Sequential(self.conv5, self.norm5, self.maxpool5, self.relu5)

        # Block 6
        self.conv6 = nn.Conv2d(256,64, kernel_size=1, padding='same')
        self.norm6 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU()
        self.block6 = nn.Sequential(self.conv6, self.norm6, self.relu6)

        # Block 7
        self.avgpool1 = nn.AvgPool2d(kernel_size=1)
        self.dropout1 = nn.Dropout(.5)
        self.flat1 = nn.Flatten()
        self.linear1 = nn.Linear(1600, 1)

        self.sigmoid = nn.Sigmoid()

        self.block7 = nn.Sequential(self.avgpool1, self.dropout1, self.flat1, self.linear1, self.sigmoid)
    
    def forward(self, x):
#        print(x.shape)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        # return torch.squeeze(x,1) # change to 2?
        return torch.squeeze(x)

            
