import torch
import torchvision
import torch.nn as nn
from model import *
import torch.optim as optim
import pandas as pd 
import torchvision.transforms as transforms

nb_epochs = 5

batch_size = 2000

batch_size = 2000

trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=True, download=True,
                                                         transform=torchvision.transforms.Compose([
                                                             torchvision.transforms.ToTensor(),
                                                             torchvision.transforms.Normalize(
                                                                 (0.1307,), (0.3081,))
                                                         ])),
    batch_size=batch_size, shuffle=True)

testloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=False, download=True,
                                                         transform=torchvision.transforms.Compose([
                                                             torchvision.transforms.ToTensor(),
                                                             torchvision.transforms.Normalize(
                                                                 (0.1307,), (0.3081,))
                                                         ])),
    batch_size=batch_size, shuffle=True)


net = DoubleCnnThreeMlp()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
loss = []

for r in range(nb_epochs):

    total_loss = 0
    total_cnt = 0

    for i, batch in enumerate(trainloader, 0):

        images, labels = batch

        optimizer.zero_grad()

        prediction = net(images) # softmax

        loss = criterion(prediction, labels)

        loss.backward() # compute the backpropagation 

        optimizer.step() # based on the computed gradients update the weights of the 

        total_cnt += batch_size

        total_loss += loss.item()
        if i % 5 == 0:
            print("Loss: ", total_loss/total_cnt)
            print(images.shape)
            
    #loss.append([i, total_loss.item()])

print('Finished Training')


PATH = './mnist_net.pth'
torch.save(net.state_dict(), PATH)


dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
# imshow(torchvision.utils.make_grid(images))
#print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


"""
Evaluation
"""
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.1307,), (0.3081,))])

# testset = torchvision.datasets.CIFAR10(
#     root='./data',
#     train=False,
#     download=True,
#     transform=transform)

# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('0', '1')

dataiter = iter(testloader)
images, labels = dataiter.next()
net = DoubleCnnThreeMlp()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print("predicted: ", predicted)

#print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))


"""
Accuracy
"""
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









