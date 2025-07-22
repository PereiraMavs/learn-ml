import torch
import torch.nn as nn
import torch.nn.functional as F


# specification of the CNN model
# This is a simple CNN architecture for image classification tasks
# It consists of two convolutional layers followed by two fully connected layers
# The model is designed to work with CIFAR-10 dataset images (32x32 pixels)
# The input is expected to have 3 channels (RGB)
# The output is a tensor of shape (batch_size, 10) for 10 classes


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
        x = x.view(-1, 16 * 5 * 5)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Define a CNN model
def get_model():
    model = Net()
    return model

