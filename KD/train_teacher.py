# this file trains a CNN teacher model on the CIFAR-10 dataset

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

import argparse
import os

from model import cnn
from test_teacher import test
from utils import imshow


def loss_fn(model):
    criterion = nn.CrossEntropyLoss()
    # Try Adam optimizer with higher learning rate
    #optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    # Alternative: SGD with higher learning rate
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    return criterion, optimizer

def save_model(model, model_path):
    # Save the trained model and create the directory if it does not exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

def load_model(model, model_path):
    """Load a previously trained model"""
    if os.path.exists(model_path):
        print(f'Loading model from {model_path}')
        model.load_state_dict(torch.load(model_path))
        print('Model loaded successfully')
        return True
    else:
        print(f'Model file {model_path} not found. Starting fresh training.')
        return False

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """Save training checkpoint with optimizer state"""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved to {checkpoint_path}')

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load training checkpoint with optimizer state"""
    if os.path.exists(checkpoint_path):
        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f'Checkpoint loaded. Resuming from epoch {start_epoch}')
        return start_epoch, loss
    else:
        print(f'Checkpoint file {checkpoint_path} not found. Starting fresh training.')
        return 0, None

def validate_model(model, testloader):
    """Validate model during training"""
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    model.train()  # Switch back to training mode
    return accuracy

def train(trainloader, testloader=None, resume_training=False, pretrained_model_path=None, resume_checkpoint=None):
    # Add device support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = cnn.get_model().to(device)
    criterion, optimizer = loss_fn(model)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 35], gamma=0.1)
    
    start_epoch = 0
    best_accuracy = 0
    
    # Option 1: Load pretrained model (model weights only)
    if resume_training and pretrained_model_path:
        success = load_model(model, pretrained_model_path)
        if not success:
            print("Failed to load pretrained model. Starting fresh training.")
    
    # Option 2: Resume from checkpoint (model + optimizer + epoch)
    if resume_checkpoint:
        start_epoch, _ = load_checkpoint(model, optimizer, resume_checkpoint)
    
    total_epochs = 50
    for epoch in range(start_epoch, total_epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        num_batches = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move to device
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()
            num_batches += 1
            
            if i % 200 == 199:  # More frequent logging
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        avg_epoch_loss = epoch_loss / num_batches
        
        # Validate every 5 epochs
        if testloader and (epoch + 1) % 5 == 0:
            val_accuracy = validate_model(model, testloader)
            print(f'Epoch [{epoch + 1}/{total_epochs}] - Loss: {avg_epoch_loss:.4f}, LR: {current_lr:.6f}, Val Acc: {val_accuracy:.2f}%')
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model_path = 'trained_models/best_cnn_teacher.pth'
                save_model(model, best_model_path)
                print(f'New best model saved! Accuracy: {best_accuracy:.2f}%')
        else:
            print(f'Epoch [{epoch + 1}/{total_epochs}] - Loss: {avg_epoch_loss:.4f}, LR: {current_lr:.6f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'trained_models/checkpoint_epoch_{epoch + 1}.pth'
            save_checkpoint(model, optimizer, epoch + 1, avg_epoch_loss, checkpoint_path)

    print(f'Training completed. Best validation accuracy: {best_accuracy:.2f}%')
    
    # FIX: Save the final trained model
    if resume_training or resume_checkpoint:
        final_model_path = f'trained_models/cnn_teacher_continued_epoch_{total_epochs}.pth'
    else:
        final_model_path = 'trained_models/cnn_teacher.pth'
    
    save_model(model, final_model_path)
    print(f'Final model saved to: {final_model_path}')
    
    return model

def validate_saved_model(model_path, testloader):
    """Validate that the saved model works correctly"""
    try:
        # Load the saved model
        test_model = cnn.get_model()
        
        # Check if it's a checkpoint or regular model file
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            test_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Successfully loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            test_model.load_state_dict(checkpoint)
            print("✓ Successfully loaded model weights")
        
        # Test with a small batch
        test_model.eval()
        data_iter = iter(testloader)
        images, labels = next(data_iter)
        
        with torch.no_grad():
            outputs = test_model(images)
            _, predicted = torch.max(outputs, 1)
            
        print(f"✓ Model inference working. Batch size: {images.size(0)}")
        print(f"✓ Output shape: {outputs.shape}")
        print(f"✓ Sample predictions: {predicted[:5].tolist()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model validation failed: {str(e)}")
        return False

def main(args):
    # Enhanced data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 specific
    ])
    
    # Standard normalization for testing
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    batch_size = args.batch_size
    # Load the CIFAR-10 dataset with improved transforms
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if args.train:
        # Train the model with validation
        trained_model = train(
            trainloader, 
            testloader,
            resume_training=args.resume_training,
            pretrained_model_path=args.pretrained_model_path,
            resume_checkpoint=args.resume_checkpoint
        )
        
        # Validate the correct saved model
        print("\nValidating saved model...")
        if args.resume_training or args.resume_checkpoint:
            final_model_path = f'trained_models/cnn_teacher_continued_epoch_50.pth'  # Use total_epochs
        else:
            final_model_path = 'trained_models/cnn_teacher.pth'
        
        is_valid = validate_saved_model(final_model_path, testloader)
        if is_valid:
            print("✓ Model saved and validated successfully!")
        else:
            print("✗ Model validation failed!")
        
    if args.test:
        test(model_path=args.model_path, testloader=testloader, 
             batch_size=batch_size, classes=classes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN teacher model on CIFAR-10')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--train', action='store_true', help='Flag to train the model')
    parser.add_argument('--test', action='store_true', help='Flag to test the model')
    parser.add_argument('--model_path', type=str, default='trained_models/cnn_teacher.pth', 
                       help='Path to the trained model')
    
    # New arguments for resume training
    parser.add_argument('--resume_training', action='store_true', 
                       help='Resume training from a pretrained model')
    parser.add_argument('--pretrained_model_path', type=str, 
                       default='trained_models/cnn_teacher.pth',
                       help='Path to pretrained model to resume from')
    parser.add_argument('--resume_checkpoint', type=str, 
                       help='Path to checkpoint file to resume training')
    parser.add_argument('--save_path', type=str, 
                       help='Custom save path for the trained model')
    
    args = parser.parse_args()
    main(args)