import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from model.student import get_student_model
import model.cnn as teacher_cnn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_student_model(model_path):
    """Load trained student model with checkpoint support"""
    student_model = get_student_model()
    
    # Check if it's a checkpoint or regular model file
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        student_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded student checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'loss' in checkpoint:
            print(f"✓ Training loss: {checkpoint['loss']:.4f}")
    else:
        student_model.load_state_dict(checkpoint)
        print("✓ Loaded student model weights")
    
    student_model.to(device)
    student_model.eval()
    return student_model

def load_teacher_model(model_path):
    """Load trained teacher model"""
    teacher_model = teacher_cnn.get_model()
    teacher_model.load_state_dict(torch.load(model_path, map_location=device))
    teacher_model.to(device)
    teacher_model.eval()
    return teacher_model

def test_student_vs_teacher_predictions(student_model, teacher_model, testloader):
    """Test how well student mimics teacher predictions"""
    correct_predictions = 0
    total_samples = 0
    
    # For soft label comparison
    kl_divergences = []
    mse_losses = []
    
    with torch.no_grad():
        for data in testloader:
            images, _ = data  # We ignore ground truth labels
            images = images.to(device)
            
            # Get teacher predictions (ground truth for student)
            teacher_outputs = teacher_model(images)
            teacher_predictions = torch.max(teacher_outputs, 1)[1]  # Hard predictions
            teacher_soft = F.softmax(teacher_outputs, dim=1)  # Soft predictions
            
            # Get student predictions
            student_outputs = student_model(images)
            student_predictions = torch.max(student_outputs, 1)[1]  # Hard predictions
            student_soft = F.softmax(student_outputs, dim=1)  # Soft predictions
            
            # Compare hard predictions (student vs teacher)
            correct_predictions += (student_predictions == teacher_predictions).sum().item()
            total_samples += images.size(0)
            
            # Compare soft predictions
            # KL Divergence between student and teacher distributions
            kl_div = F.kl_div(F.log_softmax(student_outputs, dim=1), 
                             teacher_soft, reduction='batchmean')
            kl_divergences.append(kl_div.item())
            
            # MSE between soft predictions
            mse = F.mse_loss(student_soft, teacher_soft)
            mse_losses.append(mse.item())
    
    # Calculate metrics
    hard_agreement = 100 * correct_predictions / total_samples
    avg_kl_divergence = np.mean(kl_divergences)
    avg_mse_loss = np.mean(mse_losses)
    
    return hard_agreement, avg_kl_divergence, avg_mse_loss

def test_student_vs_ground_truth(model, testloader, model_name):
    """Test model accuracy against actual CIFAR-10 labels"""
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
    return accuracy

def analyze_prediction_differences(student_model, teacher_model, testloader, num_samples=1000):
    """Analyze where student and teacher disagree"""
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    agreement_by_class = {cls: {'agree': 0, 'total': 0} for cls in classes}
    samples_processed = 0
    
    with torch.no_grad():
        for data in testloader:
            if samples_processed >= num_samples:
                break
                
            images, true_labels = data
            images, true_labels = images.to(device), true_labels.to(device)
            
            # Get predictions
            teacher_outputs = teacher_model(images)
            student_outputs = student_model(images)
            
            teacher_pred = torch.max(teacher_outputs, 1)[1]
            student_pred = torch.max(student_outputs, 1)[1]
            
            # Analyze agreement by class
            for i in range(len(true_labels)):
                if samples_processed >= num_samples:
                    break
                    
                true_class = classes[true_labels[i].item()]
                agreement_by_class[true_class]['total'] += 1
                
                if teacher_pred[i] == student_pred[i]:
                    agreement_by_class[true_class]['agree'] += 1
                
                samples_processed += 1
    
    print("\n=== Agreement by Class ===")
    for cls in classes:
        if agreement_by_class[cls]['total'] > 0:
            agreement_rate = 100 * agreement_by_class[cls]['agree'] / agreement_by_class[cls]['total']
            print(f"{cls}: {agreement_rate:.1f}% ({agreement_by_class[cls]['agree']}/{agreement_by_class[cls]['total']})")

def main(model_path='trained_models/student_model.pth'):
    # Load CIFAR-10 test data with same normalization as training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                           shuffle=False, num_workers=2)
    
    # Load models
    print(f"Loading models from {model_path}...")
    student_model = load_student_model(model_path)
    teacher_model = load_teacher_model('trained_models/best_cnn_teacher.pth')
    
    print("="*60)
    print("KNOWLEDGE DISTILLATION EVALUATION")
    print("="*60)
    
    # Test 1: Student vs Teacher Predictions (Knowledge Transfer)
    print("\n1. Testing Student vs Teacher Predictions...")
    hard_agreement, kl_div, mse_loss = test_student_vs_teacher_predictions(
        student_model, teacher_model, testloader
    )
    
    print(f"Hard Prediction Agreement: {hard_agreement:.2f}%")
    print(f"Average KL Divergence: {kl_div:.4f}")
    print(f"Average MSE Loss: {mse_loss:.4f}")
    
    # Test 2: Both models vs Ground Truth (Real Performance)
    print("\n2. Testing vs Ground Truth Labels...")
    teacher_acc = test_student_vs_ground_truth(teacher_model, testloader, "Teacher")
    student_acc = test_student_vs_ground_truth(student_model, testloader, "Student")
    
    print(f"Teacher Accuracy: {teacher_acc:.2f}%")
    print(f"Student Accuracy: {student_acc:.2f}%")
    print(f"Performance Gap: {teacher_acc - student_acc:.2f}%")
    
    # Test 3: Detailed Analysis
    print("\n3. Analyzing Prediction Differences...")
    analyze_prediction_differences(student_model, teacher_model, testloader)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Knowledge Transfer Success: {hard_agreement:.1f}% agreement with teacher")
    print(f"Performance Retention: {student_acc/teacher_acc*100:.1f}% of teacher's accuracy")
    
    if hard_agreement > 80:
        print("✓ Excellent knowledge transfer!")
    elif hard_agreement > 60:
        print("~ Good knowledge transfer")
    else:
        print("✗ Poor knowledge transfer - consider retraining")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test student model performance')
    parser.add_argument('--model_path', type=str, default='trained_models/student_model.pth',
                       help='Path to student model to test')
    
    args = parser.parse_args()
    main(args.model_path)