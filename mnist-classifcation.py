# Import libraries
import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.init as init
import torchvision.datasets as dset 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable


# Set hyperparameters
batch_size = 16 
learning_rate = 0.0002
num_epoch = 10


# Download MNIST dataset
mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)


# Set dataloader
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)


# Define CNN Model 
class CNN(nn.Module):
  def __init__(self):
    super(CNN,self).__init__()
    self.layer = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=5),
        nn.ReLU(),

        nn.Conv2d(16, 32, kernel_size=5, padding="same"),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        
        nn.Conv2d(32, 64, kernel_size=5, padding="same"),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
    )

    self.fc_layer = nn.Sequential(
        nn.Linear(6*6*64, 100), 
        nn.ReLU(),
        nn.Linear(100,10)     
    )
        
  def forward(self, x):
    out = self.layer(x)
    out = out.view(batch_size, -1) 
    out = self.fc_layer(out) 

    return out 

model = CNN().cuda()


# Define loss function and optimizer 
loss_func = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# Train CNN Model 
for i in range(num_epoch):
  for j, [image, label] in enumerate(train_loader):
    x = Variable(image).cuda()
    y_= Variable(label).cuda()   
    optimizer.zero_grad()  
    
    output = model.forward(x) 
    loss = loss_func(output, y_) 
    loss.backward() 
    
    optimizer.step() 

    if j % 1000 == 0:
      print(loss)
      
      
# Test CNN Model
correct = 0
total = 0
for image,label in test_loader:
    x = Variable(image,volatile=True).cuda()
    y_= Variable(label).cuda()
    output = model.forward(x)
    _,output_index = torch.max(output,1) # Select the digits with the highest output scores (0-9)
    total += label.size(0) # Counting the total number of test data
    correct += (output_index == y_).sum().float() # Counting the number of correctly classified data
print("Accuracy of Test Data: {}".format(100*correct/total))