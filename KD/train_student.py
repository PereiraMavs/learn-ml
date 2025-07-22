import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import numpy as np
from model.student import get_student_model
from model.generator import Generator
import model.cnn as teacher_cnn
from art_noise import AdaptiveRandomTesting

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class KnowledgeDistillationLoss(nn.Module):
    """Knowledge Distillation Loss combining soft targets and hard targets"""
    def __init__(self, temperature=4.0, alpha=0.7):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_outputs, teacher_outputs, targets=None):
        # Distillation loss (soft targets)
        soft_student = F.log_softmax(student_outputs / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_outputs / self.temperature, dim=1)
        distillation_loss = self.kl_div(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # If targets are provided, combine with hard target loss
        if targets is not None:
            hard_loss = self.ce_loss(student_outputs, targets)
            return self.alpha * distillation_loss + (1 - self.alpha) * hard_loss
        else:
            return distillation_loss

def save_student_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """Save student training checkpoint"""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'Student checkpoint saved to {checkpoint_path}')

def load_student_checkpoint(model, optimizer, checkpoint_path):
    """Load student training checkpoint"""
    if os.path.exists(checkpoint_path):
        print(f'Loading student checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f'Checkpoint loaded. Resuming from epoch {start_epoch}')
        return start_epoch, loss
    else:
        print(f'Checkpoint file {checkpoint_path} not found. Starting fresh training.')
        return 0, None

def load_student_model(model_path):
    """Load a previously trained student model"""
    if os.path.exists(model_path):
        print(f'Loading student model from {model_path}')
        student_model = get_student_model().to(device)
        
        # Check if it's a checkpoint or regular model file
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            student_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded student checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            student_model.load_state_dict(checkpoint)
            print('Student model loaded successfully')
        
        return student_model
    else:
        print(f'Student model file {model_path} not found.')
        return None

def load_teacher_model(model_path):
    """Load the trained teacher model"""
    teacher_model = teacher_cnn.get_model()
    checkpoint = torch.load(model_path, map_location=device)
    teacher_model.load_state_dict(checkpoint)
    teacher_model.to(device)
    teacher_model.eval()
    return teacher_model

def load_generator(model_path, nz=100, ngpu=1):
    """Load the trained generator model"""
    generator = Generator(ngpu).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint['model_state_dict'])
    generator.eval()
    return generator

def generate_synthetic_data(generator, teacher_model, num_samples=10000, batch_size=128, nz=100):
    """Generate synthetic data and get teacher predictions"""
    synthetic_images = []
    teacher_predictions = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
            # Generate random noise
            noise = torch.randn(current_batch_size, nz, 1, 1, device=device)
            
            # Generate fake images
            fake_images = generator(noise)
            
            # Resize images if needed (generator might output 64x64, teacher expects 32x32)
            if fake_images.size(-1) != 32:
                fake_images = F.interpolate(fake_images, size=(32, 32), mode='bilinear', align_corners=False)
            
            # Get teacher predictions (soft labels)
            teacher_outputs = teacher_model(fake_images)
            
            synthetic_images.append(fake_images.cpu())
            teacher_predictions.append(teacher_outputs.cpu())
            
            if (i + 1) % 10 == 0:
                print(f"Generated {(i + 1) * batch_size}/{num_samples} samples")
    
    # Concatenate all batches
    synthetic_images = torch.cat(synthetic_images, dim=0)
    teacher_predictions = torch.cat(teacher_predictions, dim=0)
    
    return synthetic_images, teacher_predictions

def generate_synthetic_data_with_art(generator, teacher_model, num_samples=10000, 
                                   batch_size=64, nz=100, use_art=True, 
                                   distance_metric='euclidean'):
    """Generate synthetic data using Adaptive Random Testing (ART) for noise diversity"""
    synthetic_images = []
    teacher_predictions = []
    
    print(f"Using Adaptive Random Testing with {distance_metric} distance metric")
    art_generator = AdaptiveRandomTesting(nz=nz, device=device, exclusion_radius=0.1)
    
    # Generate images in batches using ART noise
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
            # Generate ART noise directly in batch
            art_noise = art_generator.generate_art_noise(
                batch_size=current_batch_size,
                shape=(1, 1),  # For conv input
                metric=distance_metric,
                distribution='normal'
            )
            
            # Generate fake images using ART noise
            fake_images = generator(art_noise)

            
            
            # Resize images if needed (generator might output 64x64, teacher expects 32x32)
            if fake_images.size(-1) != 32:
                fake_images = F.interpolate(fake_images, size=(32, 32), mode='bilinear', align_corners=False)
            
            # Get teacher predictions (soft labels)
            teacher_outputs = teacher_model(fake_images)
            
            synthetic_images.append(fake_images.cpu())
            teacher_predictions.append(teacher_outputs.cpu())
            
            if (i + 1) % 10 == 0:
                print(f"Generated {(i + 1) * current_batch_size}/{num_samples} samples using ART")
    
    # Concatenate all batches
    synthetic_images = torch.cat(synthetic_images, dim=0)
    teacher_predictions = torch.cat(teacher_predictions, dim=0)
    
    print(f"Generated {len(synthetic_images)} synthetic samples using ART method")
    return synthetic_images, teacher_predictions

def train_student(student_model, synthetic_data, teacher_predictions, 
                 num_epochs=20, batch_size=128, lr=0.001, 
                 resume_training=False, checkpoint_path=None, method_suffix=""):
    """Train student model with optional resume capability"""
    
    # Create data loader for synthetic data
    dataset = torch.utils.data.TensorDataset(synthetic_data, teacher_predictions)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup optimizer and loss
    # use sgd instead of adam for better convergence
    optimizer = optim.SGD(student_model.parameters(), lr=lr, momentum=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    kd_loss = KnowledgeDistillationLoss(temperature=4.0, alpha=1.0)  # Pure distillation
    
    start_epoch = 0
    best_loss = float('inf')
    
    # Resume from checkpoint if specified
    if resume_training and checkpoint_path:
        start_epoch, last_loss = load_student_checkpoint(student_model, optimizer, checkpoint_path)
        if last_loss is not None:
            best_loss = last_loss
    
    student_model.train()
    
    # Training loop for knowledge distillation
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, teacher_outputs) in enumerate(dataloader):
            images = images.to(device)
            teacher_outputs = teacher_outputs.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            student_outputs = student_model(images)
            
            # Calculate knowledge distillation loss
            loss = kd_loss(student_outputs, teacher_outputs)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')
        
        scheduler.step()
        avg_loss = running_loss / num_batches
        
        # Save best model with method-specific name
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = f'trained_models/best_student_model_{method_suffix}.pth' if method_suffix else 'trained_models/best_student_model.pth'
            save_student_model(student_model, best_model_path)
            print(f'New best student model saved! Loss: {best_loss:.4f}')
        
        # Save checkpoint every 5 epochs with method-specific name
        if (epoch + 1) % 5 == 0:
            checkpoint_name = f'student_checkpoint_{method_suffix}_epoch_{epoch + 1}.pth' if method_suffix else f'student_checkpoint_epoch_{epoch + 1}.pth'
            checkpoint_save_path = f'trained_models/{checkpoint_name}'
            save_student_checkpoint(student_model, optimizer, epoch + 1, avg_loss, checkpoint_save_path)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}')
    
    print('Student training completed!')
    return student_model

def save_student_model(model, save_path):
    """Save the trained student model"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f'Student model saved to {save_path}')

def main(resume_from_model=None, resume_from_checkpoint=None, 
         additional_epochs=10, use_art=False):
    """
    Main training function with resume capability
    
    Args:
        resume_from_model: Path to saved student model to continue training
        resume_from_checkpoint: Path to checkpoint to resume from
        additional_epochs: Number of additional epochs to train
        use_art: Whether to use Adaptive Random Testing for noise generation
    """
    print("Starting Knowledge Distillation Training...")
    
    # Paths to saved models
    teacher_model_path = 'trained_models/best_cnn_teacher.pth'
    generator_model_path = 'best_models/best_generator.pth'
    
    # Check if required models exist
    if not os.path.exists(teacher_model_path):
        print(f"Teacher model not found at {teacher_model_path}")
        return
    
    if not os.path.exists(generator_model_path):
        print(f"Generator model not found at {generator_model_path}")
        return
    
    # Load models
    print("Loading teacher model...")
    teacher_model = load_teacher_model(teacher_model_path)
    
    print("Loading generator model...")
    generator = load_generator(generator_model_path)
    
    # Create or load student model
    if resume_from_model:
        print(f"Loading existing student model from {resume_from_model}...")
        student_model = load_student_model(resume_from_model)
        if student_model is None:
            print("Failed to load student model. Creating new one...")
            student_model = get_student_model().to(device)
    else:
        print("Creating new student model...")
        student_model = get_student_model().to(device)
    
    # Generate synthetic data
    if use_art:
        print("Generating synthetic data with Adaptive Random Testing...")
        synthetic_images, teacher_predictions = generate_synthetic_data_with_art(
            generator, teacher_model, 
            num_samples=1000, 
            batch_size=64,
            use_art=True,
            distance_metric='euclidean'
        )
    else:
        print("Generating synthetic data with standard random noise...")
        synthetic_images, teacher_predictions = generate_synthetic_data(
            generator, teacher_model, num_samples=20000, batch_size=128
        )
    
    print(f"Generated {len(synthetic_images)} synthetic samples")
    
    # Train student model
    print("Training student model...")
    method_suffix = "art" if use_art else "standard"
    trained_student = train_student(
        student_model, synthetic_images, teacher_predictions, 
        num_epochs=additional_epochs, batch_size=128, lr=0.001,
        resume_training=bool(resume_from_checkpoint),
        checkpoint_path=resume_from_checkpoint,
        method_suffix=method_suffix
    )
    
    # Save the final trained model with different names based on method used
    if use_art:
        if resume_from_model or resume_from_checkpoint:
            student_save_path = f'trained_models/student_model_art_retrained_epoch_{additional_epochs}.pth'
        else:
            student_save_path = 'trained_models/student_model_art.pth'
        print(f"🎯 ART-trained student model will be saved to: {student_save_path}")
    else:
        if resume_from_model or resume_from_checkpoint:
            student_save_path = f'trained_models/student_model_standard_retrained_epoch_{additional_epochs}.pth'
        else:
            student_save_path = 'trained_models/student_model_standard.pth'
        print(f"📊 Standard-trained student model will be saved to: {student_save_path}")
    
    save_student_model(trained_student, student_save_path)
    print(f"Final student model saved to: {student_save_path}")
    
    # Also update the best model saving in train_student function
    method_suffix = "art" if use_art else "standard"
    print(f"Best model during training saved as: trained_models/best_student_model_{method_suffix}.pth")
    
    print("Knowledge distillation training completed successfully!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train student model with knowledge distillation')
    parser.add_argument('--resume_model', type=str, 
                       help='Path to saved student model to continue training')
    parser.add_argument('--resume_checkpoint', type=str,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of (additional) epochs to train')
    parser.add_argument('--use_art', action='store_true',
                       help='Use Adaptive Random Testing for noise generation')
    
    args = parser.parse_args()
    
    main(
        resume_from_model=args.resume_model,
        resume_from_checkpoint=args.resume_checkpoint,
        additional_epochs=args.epochs,
        use_art=args.use_art
    )