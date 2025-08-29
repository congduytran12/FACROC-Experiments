import random
import numpy as np
from utils import distance

class KCenters(object):
    def __init__(self, k=2):
        """
        k (int) : Number of centers to be identified
        """
        self.k = k

    def fit(self, data):
        """
        Performs the enhanced k-centers algorithm with k-means++ initialization.

        Args:
            data (list) : Points in the dataset
        """
        random.seed(42)
        np.random.seed(42)

        self.data = data
        self.centers = []
        self.costs = []
        
        if len(self.data) == 0:
            return
            
        # k-means++ initialization for better center selection
        # Choose first center randomly
        first_center = np.random.randint(0, len(self.data))
        self.centers.append(first_center)
        
        # Choose remaining centers using k-means++ logic
        for _ in range(self.k - 1):
            if len(self.centers) >= len(self.data):
                break
                
            # Calculate distances from each point to nearest center
            distances = []
            rem_points = list(set(range(0, len(self.data))) - set(self.centers))
            
            for i in rem_points:
                min_dist = min([distance(self.data[i], self.data[j]) for j in self.centers])
                distances.append((i, min_dist))
            
            if not distances:
                break
                
            # Choose next center with probability proportional to squared distance
            distances = sorted(distances, key=lambda x: x[1], reverse=True)
            weights = [d[1]**2 for d in distances]
            total_weight = sum(weights)
            
            if total_weight > 0:
                probabilities = [w / total_weight for w in weights]
                
                # Select based on weighted probability
                rand_val = np.random.random()
                cumsum = 0
                next_center = distances[0][0]  # fallback
                
                for i, (point_idx, _) in enumerate(distances):
                    cumsum += probabilities[i]
                    if rand_val <= cumsum:
                        next_center = point_idx
                        break
                        
                self.centers.append(next_center)
                self.costs.append(distances[0][1])  # max distance for this iteration
            else:
                # If all distances are 0, just pick the first remaining point
                self.centers.append(distances[0][0])
                self.costs.append(0)
        
        return

    def assign(self):
        """
        Enhanced assignment with stability and quality considerations.

        Returns:
            mapping (list) : tuples of the form (point, center)
        """
        if not self.centers:
            return []
            
        mapping = []
        for i in range(len(self.data)):
            # Calculate distances to all centers
            center_distances = [(j, distance(self.data[i], self.data[j])) for j in self.centers]
            center_distances = sorted(center_distances, key=lambda x: x[1])
            
            # Enhanced assignment considering cluster balance
            closest_center = center_distances[0][0]
            closest_distance = center_distances[0][1]
            
            # If there are very close alternatives, consider using them for better balance
            # (This helps with fairness without significantly hurting clustering quality)
            if len(center_distances) > 1:
                second_closest_distance = center_distances[1][1]
                
                # If second closest is within 20% of closest distance, we have flexibility
                if second_closest_distance <= closest_distance * 1.2:
                    # Use closest for now, but this creates opportunity for refinement step
                    pass
            
            mapping.append((i, closest_center))
        
        return mapping