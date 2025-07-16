import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

from utils import imshow
from model import cnn


def test(model_path, testloader, batch_size=4, classes=None):
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    # Show images
    imshow(torchvision.utils.make_grid(images))
    # Print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    
    net = cnn.get_model()
    net.load_state_dict(torch.load(model_path, weights_only=True))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(batch_size)))

