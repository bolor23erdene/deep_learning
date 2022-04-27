

import torch.nn as nn
import torch.nn.functional as F
import torch

# define a CNN model 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5) # 28x28x6
        self.fc1 = nn.Linear(28*28*6, 10)
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        #x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.conv1(x))
        x = self.fc1(x)
        return x

class LogisticRegression(nn.Module):
	def __init__(self):
		super(LogisticRegression, self).__init__()
		self.fc1 = nn.Linear(32*32*3, 10)

	def forward(self, x):
		x = x.view(-1, 32*32*3)
		return self.fc1(x)



"""Test"""
basic = BasicNet()


image = torch.randn(1, 3, 32, 32)
output = basic(image)
print(output.shape)