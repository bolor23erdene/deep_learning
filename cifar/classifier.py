import torch

# torchvision # downloads dataset, transforms data
import torchvision #for data
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')

from model import Net

from model import LogisticRegression
import torch.nn as nn
from pylab import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nb_epochs = 5
batch_size = 128

# defining a function called transform
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')





import matplotlib.pyplot as plt
import numpy as np

# functions to show an image

"""
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
"""





#net = Net()
net = LogisticRegression()
net.to(device)

print(net)


import torch.optim as optim

criterion = nn.CrossEntropyLoss() # define loss 
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # define optimizer 


for epoch in range(nb_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        # accumulates the gradient for RNN for subsequent backpropagation, but, for CNNs you need to zer the gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        
        # calling backward() multiple times accumulates the gradient for each parameter 
        loss.backward()

        # parameter update based on the current gradient (stored in .grad attributes of parameters)
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (i+1) % 100 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / (batch_size*(i+1))))
            running_loss = 0.0

print('Finished Training')



PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)


dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
# imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


"""
Evaluation test: 
"""
PATH = './cifar_net.pth'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


dataiter = iter(testloader)
images, labels = dataiter.next()


# net = Net()
net = LogisticRegression()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
