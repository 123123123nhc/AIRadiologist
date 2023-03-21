import numpy
from matplotlib import pyplot as plt
from torch.autograd import Variable

import torch.utils.data as data
import pathlib
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time

from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler

from model_CNN import Net
torch.cuda.empty_cache()
# the path of the dateset
base_path = r"Data\TB_Chest_Radiography_Database"
base_path = pathlib.Path(base_path)

# transform the data in the path
img_size =(512,512)
# transform the data in the path
transform=transforms.Compose(
            [transforms.Grayscale(num_output_channels=3),transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
data_all = torchvision.datasets.ImageFolder(root=base_path, transform=transform)


def dataset_sampler(dataset, test_size):
    """
    split dataset into train set and val set
    :param dataset:
    :param val_percentage: validation percentage
    :return: split sampler
    """

    sample_num = len(dataset)
    file_idx = list(range(sample_num))
    # split the test data
    train_idx, val_idx = train_test_split(file_idx, test_size=test_size, random_state=42)
    # create the label
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    return train_sampler, val_sampler


train_sampler, val_sampler = dataset_sampler(data_all, 0.2)

batch_size = 16
# create trainloader and testloader
trainloader = data.DataLoader(data_all, batch_size=batch_size, num_workers=0, sampler=train_sampler)
testloader = data.DataLoader(data_all, batch_size=batch_size, num_workers=0, sampler=val_sampler)

# Define the pytorch network

net = Net()
print(net)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device) # change to GPU
criterion = nn.CrossEntropyLoss() # define the loss function
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # define the optimizer

start = time.time()
loss_list = []
accuracy_list = []
epoch_num = 0
softmax = nn.Softmax(dim=1)
for epoch in range(20):
    epoch_num += 1
    accuracy = 0


    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        test_y = Variable(labels)

        optimizer.zero_grad()
        outputs = net(inputs)
        outputs = softmax(outputs)
        loss = criterion(outputs, labels)
        # backward loss function
        loss.backward()
        optimizer.step()
        outputs = outputs.cpu()
        test_y = test_y.cpu()
        running_loss += loss.item()
        accuracy += numpy.sum(torch.max(outputs, 1)[1].numpy() == test_y.numpy())

    print("epoch" + str(epoch_num))
    loss_list.append(running_loss)
    accuracy /= (len(data_all) * 0.8)
    print("accuracy=", accuracy)
    accuracy_list.append(accuracy)

print('Finished Training! Total cost time: ', time.time() - start)
x1 = range(0, 20)
x2 = range(0, 20)
y1 = accuracy_list
y2 = loss_list
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('Test accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('Test loss vs. epoches')
plt.ylabel('Test loss')
plt.show()
plt.savefig(r"accuracy_loss.jpg")

# test the model
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        # record the correct number
        correct += (predicted == labels).sum().item()

print('Accuracy of the network : %d %%' % (100 * correct / total))
torch.save(net, r"Model/model1_cnn.pt")


