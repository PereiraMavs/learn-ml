# this file trains a CNN teacher model on the CIFAR-10 dataset

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import argparse

from model import cnn
from test_teacher import test
from utils import imshow


def loss_fn(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return criterion, optimizer

def train(trainloader):
    model = cnn.get_model()
    criterion, optimizer = loss_fn(model)

    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()  # zero the parameter gradients

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # backpropagation
            optimizer.step()  # optimize the parameters

            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

def main(args):
    # Create argparse arguments for the script




    # Define a transform to normalize the data
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4
    # Load the CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    # Show images
    #imshow(torchvision.utils.make_grid(images))
    # Print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    if args.train:
        # Train the model
        train(trainloader)
        # Save the trained model
        torch.save(cnn.get_model().state_dict(), args.model_path)
    if args.test:
        test(model_path=args.model_path, testloader=testloader, batch_size=batch_size, classes=classes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN teacher model on CIFAR-10')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--train', action = 'store_true', help='Flag to train the model')
    parser.add_argument('--test', action = 'store_true', help='Flag to test the model')
    parser.add_argument('--model_path', type=str, default='trained_models/cnn_teacher.pth', help='Path to the trained model')
    args = parser.parse_args()
    main(args)