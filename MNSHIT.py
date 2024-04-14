#Code for project based on this project https://nextjournal.com/gkoehler/pytorch-mnist

#environment setup
import time # Time tracking library used to evaluate program preformance
import torch #Machine learning library with dynamic graph
import torchvision as tv #Image recognition library
import matplotlib.pyplot as plt #plot library for graphs and picture output
import numpy as np # math library
start = time.perf_counter_ns()
print("Librarys imported", (time.perf_counter_ns() - start))

seed = 1 #seed for all random processes


# Dataset setup
start = time.perf_counter_ns()
n_epochs = 3 # number of iteration for training
batch_size_train = 64 # number of examples used for training
batch_size_test = 1000 # number of examples used for testing
learning_rate = 0.01
momentum = 0.5
log_interval = 10
print("Data setup complete", (time.perf_counter_ns() - start))

random_seed = seed #for repeatable results we set a seed for random functions, this makes our outputs deterministic
torch.backends.cudnn.enabled = False #this dissables the use of certain cuda finctions that are non-deterministic
torch.manual_seed(random_seed) #this manualy sets the seed for pytorch

#Loading MNIST Dataset from http://yann.lecun.com/exdb/mnist
#loads MNIST data into training data and converts images into normalized tensors then shuffles the data
start = time.perf_counter_ns()
train_loader = torch.utils.data.DataLoader(
  tv.datasets.MNIST('/files/', train=True, download=True,
                             transform=tv.transforms.Compose([
                               tv.transforms.ToTensor(),
                               tv.transforms.Normalize((0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)
print("Training data loaded", (time.perf_counter_ns() - start))

start = time.perf_counter_ns()
test_loader = torch.utils.data.DataLoader(
  tv.datasets.MNIST('/files/', train=False, download=True,
                             transform=tv.transforms.Compose([
                               tv.transforms.ToTensor(),
                               tv.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)
print("Test data loaded", (time.perf_counter_ns() - start))

'''examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

print(example_data.shape)'''

#example plot function
'''fig = plt.figure()
for i in range(6): #plots 6 example dataset pictures
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
fig
plt.show()'''

#This is where we seutp the nural network

'''We'll use two 2-D convolutional layers followed by two fully-connected (or linear) layers. 
  As activation function we'll choose rectified linear units (ReLUs in short) and as a means of 
  regularization we'll use two dropout layers.'''

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#in pytorch networks are setup as classes here we make a class called net that uses the torch.nn.Module
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # first convelution layer, this takes the large image tensor and recudes it.
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # second convelution layer
        self.conv2_drop = nn.Dropout2d() # first dropout layer
        self.fc1 = nn.Linear(320, 50) # first linear layer
        self.fc2 = nn.Linear(50, 10) # second linear layer

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
#Lets use the network and optimize it
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
