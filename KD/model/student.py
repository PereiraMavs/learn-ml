import torch
import torch.nn as nn
import torch.nn.functional as F

class StudentCNN(nn.Module):
    """Lightweight student model for knowledge distillation"""
    def __init__(self, num_classes=10):
        super(StudentCNN, self).__init__()
        
        # Smaller architecture compared to teacher
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Calculate the size after convolutions
        # Assuming input size 32x32 (CIFAR-10 native or resized from 64x64)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)  # 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # First conv block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x32 -> 16x16
        
        # Second conv block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16x16 -> 8x8
        
        # Third conv block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8x8 -> 4x4
        
        # Flatten and fully connected layers
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def get_student_model():
    """Factory function to create student model"""
    return StudentCNN(num_classes=10)