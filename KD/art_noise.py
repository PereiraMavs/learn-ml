import torch
import numpy as np
from typing import List, Tuple

class AdaptiveRandomTesting:
    """
    Adaptive Random Testing for generating well-distributed noise vectors
    """
    def __init__(self, nz: int, device: str = 'cpu', exclusion_radius: float = 0.1):
        self.nz = nz
        self.device = device
        self.exclusion_radius = exclusion_radius
        self.generated_samples = []
        self.max_candidates = 5  # Number of candidates to consider per sample
        
    def euclidean_distance(self, sample1: torch.Tensor, sample2: torch.Tensor) -> float:
        """Calculate Euclidean distance between two noise vectors"""
        return torch.norm(sample1 - sample2, p=2).item()
    
    def manhattan_distance(self, sample1: torch.Tensor, sample2: torch.Tensor) -> float:
        """Calculate Manhattan distance between two noise vectors"""
        return torch.norm(sample1 - sample2, p=1).item()
    
    def cosine_distance(self, sample1: torch.Tensor, sample2: torch.Tensor) -> float:
        """Calculate cosine distance between two noise vectors"""
        cos_sim = torch.nn.functional.cosine_similarity(
            sample1.flatten(), sample2.flatten(), dim=0
        )
        return (1 - cos_sim).item()
    
    def calculate_distance(self, sample1: torch.Tensor, sample2: torch.Tensor, 
                          metric: str = 'euclidean') -> float:
        """Calculate distance using specified metric"""
        if metric == 'euclidean':
            return self.euclidean_distance(sample1, sample2)
        elif metric == 'manhattan':
            return self.manhattan_distance(sample1, sample2)
        elif metric == 'cosine':
            return self.cosine_distance(sample1, sample2)
        else:
            raise ValueError(f"Unknown distance metric: {metric}")
    
    def find_min_distance_to_existing(self, candidate: torch.Tensor, 
                                    metric: str = 'euclidean') -> float:
        """Find minimum distance from candidate to all existing samples"""
        if not self.generated_samples:
            return float('inf')
        
        min_distance = float('inf')
        for existing_sample in self.generated_samples:
            distance = self.calculate_distance(candidate, existing_sample, metric)
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def generate_art_noise(self, batch_size: int, shape: Tuple = (1, 1), 
                          metric: str = 'euclidean', 
                          distribution: str = 'normal') -> torch.Tensor:
        """
        Generate noise using Adaptive Random Testing
        
        Args:
            batch_size: Number of noise vectors to generate
            shape: Additional shape dimensions (e.g., (1, 1) for conv input)
            metric: Distance metric ('euclidean', 'manhattan', 'cosine')
            distribution: Base distribution ('normal', 'uniform')
        
        Returns:
            Tensor of shape (batch_size, nz, *shape)
        """
        noise_vectors = []
        
        for i in range(batch_size):
            best_candidate = None
            best_distance = -1
            
            # Generate multiple candidates and pick the one with maximum minimum distance
            for _ in range(self.max_candidates):
                if distribution == 'normal':
                    candidate = torch.randn(self.nz, *shape, device=self.device)
                elif distribution == 'uniform':
                    candidate = torch.rand(self.nz, *shape, device=self.device) * 2 - 1
                else:
                    raise ValueError(f"Unknown distribution: {distribution}")
                
                # Calculate minimum distance to existing samples
                min_distance = self.find_min_distance_to_existing(candidate, metric)
                
                # Keep the candidate with maximum minimum distance (best coverage)
                if min_distance > best_distance:
                    best_distance = min_distance
                    best_candidate = candidate.clone()
            
            # Add the best candidate to our collection
            noise_vectors.append(best_candidate.unsqueeze(0))
            self.generated_samples.append(best_candidate)
            
            # Limit memory usage by keeping only recent samples
            if len(self.generated_samples) > 1000:
                self.generated_samples.pop(0)
            
            # Print progress for large batches
            if (i + 1) % 50 == 0:
                print(f"Generated {i + 1}/{batch_size} ART noise vectors")
        
        return torch.cat(noise_vectors, dim=0)
    
    def reset(self):
        """Reset the generated samples history"""
        self.generated_samples = []
    
    def set_exclusion_radius(self, radius: float):
        """Set the exclusion radius for diversity control"""
        self.exclusion_radius = radius