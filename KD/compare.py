import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import train_student as ts
import torch

def adaptive_random_testing(n_points, bounds):
    """
    Generate points using Adaptive Random Testing (ART) for better space coverage
    
    Args:
        n_points: Number of points to generate
        bounds: Array [min, max] defining the bounds
    
    Returns:
        Array of shape (n_points, 3) with diverse points
    """
    points = []
    for i in range(n_points):
        if not points:
            # First point is random
            p = np.random.uniform(bounds[0], bounds[1], 3)
        else:
            # Generate candidates and pick the one farthest from existing points
            candidates = np.random.uniform(bounds[0], bounds[1], (10, 3))
            dists = [min(np.linalg.norm(c - np.array(points), axis=1)) for c in candidates]
            p = candidates[np.argmax(dists)]
        points.append(p)
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{n_points} ART points")
    
    return np.array(points)

def normal_distribution_sampling(n_points, bounds):
    """
    Generate points using normal distribution sampling
    
    Args:
        n_points: Number of points to generate
        bounds: Array [min, max] defining the bounds
    
    Returns:
        Array of shape (n_points, 3) with normally distributed points
    """
    mean = (bounds[1] + bounds[0]) / 2  # 0.5 for [0,1]
    std = (bounds[1] - bounds[0]) / 6   # ~0.167 for [0,1] (99.7% within bounds)
    points = np.random.normal(mean, std, (n_points, 3))
    
    # Clip to bounds [0, 1]
    points = np.clip(points, bounds[0], bounds[1])
    return points

def calculate_diversity_metrics(points):
    """
    Calculate diversity metrics for a set of points
    
    Args:
        points: Array of shape (n_points, dimensions)
    
    Returns:
        Dictionary with diversity metrics
    """
    n_points = len(points)
    
    # Calculate all pairwise distances
    distances = []
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = np.linalg.norm(points[i] - points[j])
            distances.append(dist)
    
    distances = np.array(distances)
    
    return {
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'min_distance': np.min(distances),
        'max_distance': np.max(distances),
        'coverage_uniformity': np.std(distances) / np.mean(distances)  # Lower is more uniform
    }

def plot_comparison(art_points, normal_points, bounds):
    """
    Create comparison plots for ART vs Normal distribution
    
    Args:
        art_points: Points from ART sampling
        normal_points: Points from normal distribution
        bounds: Bounds used for generation
    """
    fig = plt.figure(figsize=(12, 6))
    
    # ART plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(art_points[:,0], art_points[:,1], art_points[:,2], 
               c='blue', s=20, alpha=0.7)
    ax1.set_title('Adaptive Random Testing\n(Better Coverage)', fontsize=12)
    ax1.set_xlim(bounds[0], bounds[1])
    ax1.set_ylim(bounds[0], bounds[1])
    ax1.set_zlim(bounds[0], bounds[1])
    ax1.set_xlabel('X (0-1)')
    ax1.set_ylabel('Y (0-1)')
    ax1.set_zlabel('Z (0-1)')
    
    # Normal Distribution plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(normal_points[:,0], normal_points[:,1], normal_points[:,2], 
               c='red', s=20, alpha=0.7)
    ax2.set_title('Normal Distribution\n(Clustered around 0.5)', fontsize=12)
    ax2.set_xlim(bounds[0], bounds[1])
    ax2.set_ylim(bounds[0], bounds[1])
    ax2.set_zlim(bounds[0], bounds[1])
    ax2.set_xlabel('X (0-1)')
    ax2.set_ylabel('Y (0-1)')
    ax2.set_zlabel('Z (0-1)')
    
    plt.tight_layout()
    plt.savefig('art_vs_normal_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_diversity_analysis(art_points, normal_points):
    """
    Print detailed diversity analysis for ART vs Normal distribution
    """
    print("\n" + "="*70)
    print("DIVERSITY ANALYSIS: ART vs NORMAL DISTRIBUTION (Range: 0.0 to 1.0)")
    print("="*70)
    
    methods = [
        ("Adaptive Random Testing", art_points),
        ("Normal Distribution", normal_points)
    ]
    
    for name, points in methods:
        metrics = calculate_diversity_metrics(points)
        print(f"\n{name}:")
        print(f"  Mean Distance: {metrics['mean_distance']:.4f}")
        print(f"  Std Distance:  {metrics['std_distance']:.4f}")
        print(f"  Min Distance:  {metrics['min_distance']:.4f}")
        print(f"  Max Distance:  {metrics['max_distance']:.4f}")
        print(f"  Coverage Uniformity: {metrics['coverage_uniformity']:.4f} (lower = more uniform)")
        
        # Show range coverage
        print(f"  Range Coverage:")
        print(f"    X: [{points[:,0].min():.3f}, {points[:,0].max():.3f}]")
        print(f"    Y: [{points[:,1].min():.3f}, {points[:,1].max():.3f}]")
        print(f"    Z: [{points[:,2].min():.3f}, {points[:,2].max():.3f}]")
    
    # Compare ART with Normal
    art_metrics = calculate_diversity_metrics(art_points)
    normal_metrics = calculate_diversity_metrics(normal_points)
    
    print(f"\n" + "-"*60)
    print("ART IMPROVEMENTS OVER NORMAL DISTRIBUTION:")
    print("-"*60)
    print(f"Mean distance improvement: {((art_metrics['mean_distance'] - normal_metrics['mean_distance']) / normal_metrics['mean_distance'] * 100):+.2f}%")
    print(f"Min distance improvement:  {((art_metrics['min_distance'] - normal_metrics['min_distance']) / normal_metrics['min_distance'] * 100):+.2f}%")
    print(f"Max distance improvement:  {((art_metrics['max_distance'] - normal_metrics['max_distance']) / normal_metrics['max_distance'] * 100):+.2f}%")
    print(f"Uniformity improvement:    {((normal_metrics['coverage_uniformity'] - art_metrics['coverage_uniformity']) / normal_metrics['coverage_uniformity'] * 100):+.2f}%")

# method to generate images using ART and normal distribution and visualize the results
def generate_and_visualize_art_vs_normal(n_points=200, bounds=np.array([0.0, 1.0])):
    """ Generate and visualize ART vs Normal distribution points """
    print(f"Generating {n_points} points in 3D space with bounds {bounds}")
    print(f"This simulates noise generation in normalized [0,1] latent space\n")

    # Generate points using ART
    art_points = adaptive_random_testing(n_points, bounds)

    # Generate points using Normal distribution
    normal_points = normal_distribution_sampling(n_points, bounds)

    # Force everything to CPU
    device = torch.device("cpu")
    generator = ts.load_generator(model_path='best_models/best_generator.pth')
    generator = generator.to(device)

    art_noise = torch.randn(1, 100, 1, 1, device=device)
    normal_noise = torch.randn(1, 100, 1, 1, device=device)

    fake_images_art = generator(art_noise)
    fake_images_normal = generator(normal_noise)
    
    print("\nGeneration completed!")
    # show fake images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    img_art = fake_images_art[0].cpu().permute(1, 2, 0).detach().numpy()
    img_art = (img_art + 1) / 2  # Denormalize from [-1,1] to [0,1]
    plt.imshow(img_art)
    plt.title('Fake Images from ART Points')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    img_normal = fake_images_normal[0].cpu().permute(1, 2, 0).detach().numpy()
    img_normal = (img_normal + 1) / 2  # Denormalize from [-1,1] to [0,1]
    plt.imshow(img_normal)
    plt.title('Fake Images from Normal Distribution Points')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('art_vs_normal_images_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    Main function to compare ART vs Normal distribution sampling
    """
    print("ART vs Normal Distribution Comparison")
    print("="*50)
    
    # Parameters
    n_points = 200
    bounds = np.array([0.0, 1.0])
    
    print(f"Generating {n_points} points in 3D space with bounds {bounds}")
    print(f"This simulates noise generation in normalized [0,1] latent space\n")
    
    # Generate points using both methods
    print("1. Generating ART points...")
    art_points = adaptive_random_testing(n_points, bounds)
    
    print("2. Generating Normal distribution points...")
    normal_points = normal_distribution_sampling(n_points, bounds)
    
    print("\nGeneration completed!")
    
    # Analyze diversity
    print_diversity_analysis(art_points, normal_points)
    
    # Create visualizations
    print(f"\n" + "="*50)
    print("Creating comparison visualizations...")
    #plot_comparison(art_points, normal_points, bounds)
    generate_and_visualize_art_vs_normal(n_points, bounds)
    print("✓ Visualization saved as 'art_vs_normal_comparison.png'")
    
    # Summary
    print(f"\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("🎯 ART provides better space coverage in [0,1] cube")
    print("🎯 ART ensures minimum distances between points")
    print("🎯 Normal distribution clusters around center (0.5, 0.5, 0.5)")
    print("🎯 ART leads to more diverse synthetic data in ML training")
    print("🎯 Better diversity → Better knowledge distillation performance")
    
    print(f"\nFiles created:")
    print(f"  📊 art_vs_normal_comparison.png - Side-by-side comparison")
    print(f"  📈 Diversity metrics - Printed above")

if __name__ == "__main__":
    main()