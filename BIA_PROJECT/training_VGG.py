import numpy
import torch.utils.data as data
import pathlib
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from model_VGG import VGGNet

torch.cuda.empty_cache()
# the path of the dateset
base_path =r"Data\TB2"
base_path = pathlib.Path(base_path)

# transform the data in the path
img_size =(224,224)
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
    train_sampler = torch.utils.data.RandomSampler(train_idx)
    val_sampler = torch.utils.data.RandomSampler(val_idx)
    return train_sampler, val_sampler

train_sampler, val_sampler = dataset_sampler(data_all, 0.2)


batch_size = 16
# create trainloader and testloader
trainloader = data.DataLoader(data_all, batch_size=batch_size, num_workers=0, sampler=train_sampler)
testloader = data.DataLoader(data_all, batch_size=batch_size, num_workers=0, sampler=val_sampler)

# Define the pytorch network
net = VGGNet()
print(net)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device) # change to GPU
criterion = nn.CrossEntropyLoss() # define the loss function
optimizer = optim.SGD(net.parameters(), lr=0.001) # define the optimizer
softmax = nn.Softmax(dim=1)

start = time.time()
epoch_num = 0
loss_list = []
accuracy_list = []
for epoch in range(20):
    epoch_num += 1
    running_loss = 0.0
    accuracy = 0
    for i, data in (enumerate(trainloader, 0)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        test_y = labels

        outputs = net(inputs)
        outputs = softmax(outputs)
        print(outputs)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        # backward loss function
        loss.backward()
        optimizer.step()
        outputs = outputs.cpu()
        test_y = test_y.cpu()

        # print the loss value
        running_loss += loss.item()
        accuracy +=  numpy.sum(torch.max(outputs, 1)[1].numpy() == test_y.numpy())

    print("epoch" + str(epoch_num))
    loss_list.append(running_loss)

    accuracy /= (len(data_all)*0.8)
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
plt.savefig(r"accuracy_loss_vgg.jpg")

print('Finished Training! Total cost time: ', time.time() - start)
torch.save(net.state_dict(), r"model/model1_vgg.pt")
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

print('Accuracy of the network : {:.3%}'.format(correct / total))



