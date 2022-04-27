import torch.nn as nn
import torch.nn.functional as F
import torch

# define a CNN model


class DoubleCnnThreeMlp(nn.Module):
    def __init__(self):
        super(DoubleCnnThreeMlp, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3) # 26
        self.pool = nn.MaxPool2d(2, 2) # 13
        self.conv2 = nn.Conv2d(6, 16, 3) # 11

        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x))) #6x6x16
        x = x.view(-1, 5 * 5 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class DoubleCnnMaxPool(nn.Module):
    def __init__(self):
        super(DoubleCnnMaxPool, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=3)                      
        self.pool = nn.MaxPool2d(2, 2)          
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=3)
        self.fc1 = nn.Linear(11 * 11 * 16, 10)
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))#26
        x = self.pool(x)#13x13x6
        x = F.relu(self.conv2(x))#11x11x16
        x = x.view(-1, 11 * 11 * 16)
        x = self.fc1(x)
        return x    
    

class DoubleCNN(nn.Module):
    def __init__(self):
        super(DoubleCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=3)  
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=3)
        self.fc1 = nn.Linear(24 * 24 * 16, 10)
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 26x26x6
        x = F.relu(self.conv2(x)) # 24x24x16
        x = x.view(-1, 24 * 24 * 16)
        x = self.fc1(x)
        return x


class SingleCNN(nn.Module):
    def __init__(self):
        super(SingleCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=5)  # 28x28x6
        self.fc1 = nn.Linear(24 * 24 * 6, 10)

    def forward(self, x):
        #x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.conv1(x)) # 24 x 24 x 6
        x = x.view(-1, 24 * 24 * 6)
        x = self.fc1(x)
        return x


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 10) 

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        return self.fc1(x)


"""Test"""
basic = SingleCNN()


image = torch.randn(1, 3, 32, 32)
output = basic(image)
print(output.shape)
