"""
Optimized K-Centers clustering implementation with improved initialization,
iterative refinement, and quality assessment.

This module provides the KCentersOptimized class, which addresses the limitations
of the basic K-Centers algorithm with several key improvements:

1. **Improved Initialization**: Implements K-means++ style initialization and
   multiple random starts with best selection, providing better initial center
   placement compared to single random initialization.

2. **Center Refinement**: Adds iterative optimization using medoid-based 
   refinement to improve center positions after initial selection.

3. **Better Selection Strategy**: Uses quality metrics (silhouette score,
   intra-cluster distance) rather than just maximum distance for selection.

4. **Quality Metrics**: Incorporates clustering quality measures during center
   selection, including silhouette score and intra-cluster distance optimization.

5. **Multiple Runs**: Runs the algorithm multiple times with different 
   initializations and selects the best result based on internal clustering metrics.

The optimized algorithm maintains the same interface as the original KCenters
class (fit() and assign() methods) for seamless integration with existing
fair clustering pipelines while providing significantly better clustering quality.

Expected improvements:
- Better cluster separation and cohesion
- Higher AUCC scores indicating improved clustering quality  
- More stable results across different runs
- Better handling of fairlet centers in fair clustering

Usage:
    >>> kcenters = KCentersOptimized(k=3, n_init=10, init_method='kmeans++')
    >>> kcenters.fit(data)
    >>> assignments = kcenters.assign()
"""

import numpy as np
import random
from sklearn.metrics import silhouette_score
from utils import distance

class KCentersOptimized(object):
    """
    Optimized K-Centers clustering algorithm with multiple improvements:
    1. K-means++ style initialization
    2. Multiple random starts with best selection
    3. Iterative center refinement
    4. Quality-based selection using clustering metrics
    5. Better handling of cluster balance
    """
    
    def __init__(self, k=2, n_init=10, max_iter=50, init_method='kmeans++', 
                 quality_metric='silhouette', random_state=None):
        """
        Initialize optimized K-Centers clustering.
        
        Args:
            k (int): Number of centers to identify
            n_init (int): Number of different initializations to try
            max_iter (int): Maximum iterations for center refinement
            init_method (str): Initialization method ('random', 'kmeans++', 'farthest_first')
            quality_metric (str): Quality metric for selection ('silhouette', 'intra_cluster', 'combined')
            random_state (int): Random state for reproducibility
        """
        self.k = k
        self.n_init = n_init
        self.max_iter = max_iter
        self.init_method = init_method
        self.quality_metric = quality_metric
        self.random_state = random_state
        
        # Results from fitting
        self.data = None
        self.centers = None
        self.costs = None
        self.best_score = None
        
    def _set_random_state(self, seed_offset=0):
        """Set random state for reproducible results."""
        if self.random_state is not None:
            np.random.seed(self.random_state + seed_offset)
            random.seed(self.random_state + seed_offset)
    
    def _kmeans_plus_plus_init(self, data, k):
        """
        K-means++ initialization for better center placement.
        
        Args:
            data (list): Dataset points
            k (int): Number of centers to select
            
        Returns:
            list: Indices of selected centers
        """
        n = len(data)
        centers = []
        
        # Choose first center randomly
        centers.append(np.random.randint(0, n))
        
        for _ in range(1, k):
            # Calculate squared distances to nearest centers
            distances = []
            for i in range(n):
                if i in centers:
                    distances.append(0)
                else:
                    min_dist = min([distance(data[i], data[c]) for c in centers])
                    distances.append(min_dist ** 2)
            
            # Choose next center with probability proportional to squared distance
            total_dist = sum(distances)
            if total_dist == 0:
                # All points are centers or identical
                remaining = [i for i in range(n) if i not in centers]
                if remaining:
                    centers.append(random.choice(remaining))
                break
            
            probabilities = [d / total_dist for d in distances]
            cumulative = np.cumsum(probabilities)
            r = np.random.random()
            
            for i, cum_prob in enumerate(cumulative):
                if r <= cum_prob:
                    centers.append(i)
                    break
        
        return centers
    
    def _farthest_first_init(self, data, k):
        """
        Farthest-first initialization (original k-centers approach).
        
        Args:
            data (list): Dataset points
            k (int): Number of centers to select
            
        Returns:
            list: Indices of selected centers
        """
        n = len(data)
        centers = [np.random.randint(0, n)]
        
        for _ in range(1, k):
            max_dist = -1
            farthest_point = -1
            
            for i in range(n):
                if i in centers:
                    continue
                
                min_dist_to_centers = min([distance(data[i], data[c]) for c in centers])
                if min_dist_to_centers > max_dist:
                    max_dist = min_dist_to_centers
                    farthest_point = i
            
            if farthest_point != -1:
                centers.append(farthest_point)
            else:
                break
        
        return centers
    
    def _random_init(self, data, k):
        """
        Random initialization.
        
        Args:
            data (list): Dataset points
            k (int): Number of centers to select
            
        Returns:
            list: Indices of selected centers
        """
        n = len(data)
        return random.sample(range(n), min(k, n))
    
    def _initialize_centers(self, data, k):
        """
        Initialize centers using the specified method.
        
        Args:
            data (list): Dataset points
            k (int): Number of centers to select
            
        Returns:
            list: Indices of selected centers
        """
        if self.init_method == 'kmeans++':
            return self._kmeans_plus_plus_init(data, k)
        elif self.init_method == 'farthest_first':
            return self._farthest_first_init(data, k)
        else:  # random
            return self._random_init(data, k)
    
    def _refine_centers(self, data, centers):
        """
        Iteratively refine center positions to improve clustering quality.
        
        Args:
            data (list): Dataset points
            centers (list): Current center indices
            
        Returns:
            tuple: (refined_centers, converged)
        """
        current_centers = centers.copy()
        
        for iteration in range(self.max_iter):
            # Assign points to nearest centers
            assignments = self._assign_points(data, current_centers)
            
            # Create clusters
            clusters = {}
            for point_idx, center_idx in assignments:
                if center_idx not in clusters:
                    clusters[center_idx] = []
                clusters[center_idx].append(point_idx)
            
            # Find new centers (medoids) for each cluster
            new_centers = []
            for center_idx in current_centers:
                if center_idx in clusters and len(clusters[center_idx]) > 1:
                    # Find point in cluster that minimizes total distance to other points in cluster
                    cluster_points = clusters[center_idx]
                    best_center = center_idx
                    best_total_dist = float('inf')
                    
                    for candidate in cluster_points:
                        total_dist = sum([distance(data[candidate], data[p]) for p in cluster_points])
                        if total_dist < best_total_dist:
                            best_total_dist = total_dist
                            best_center = candidate
                    
                    new_centers.append(best_center)
                else:
                    new_centers.append(center_idx)
            
            # Check for convergence
            if set(new_centers) == set(current_centers):
                return new_centers, True
            
            current_centers = new_centers
        
        return current_centers, False
    
    def _assign_points(self, data, centers):
        """
        Assign points to nearest centers.
        
        Args:
            data (list): Dataset points
            centers (list): Center indices
            
        Returns:
            list: List of (point_index, center_index) tuples
        """
        assignments = []
        for i in range(len(data)):
            best_center = centers[0]
            best_distance = distance(data[i], data[centers[0]])
            
            for center_idx in centers[1:]:
                dist = distance(data[i], data[center_idx])
                if dist < best_distance:
                    best_distance = dist
                    best_center = center_idx
            
            assignments.append((i, best_center))
        
        return assignments
    
    def _evaluate_clustering(self, data, centers, assignments):
        """
        Evaluate clustering quality using various metrics.
        
        Args:
            data (list): Dataset points
            centers (list): Center indices
            assignments (list): Point assignments
            
        Returns:
            float: Quality score (higher is better)
        """
        if len(data) < 2 or len(set([a[1] for a in assignments])) < 2:
            return -float('inf')
        
        try:
            # Convert assignments to cluster labels
            cluster_labels = [0] * len(data)
            for point_idx, center_idx in assignments:
                # Map center_idx to cluster label
                center_to_label = {centers[i]: i for i in range(len(centers))}
                cluster_labels[point_idx] = center_to_label.get(center_idx, 0)
            
            if self.quality_metric == 'silhouette':
                return silhouette_score(data, cluster_labels)
            
            elif self.quality_metric == 'intra_cluster':
                # Lower intra-cluster distance is better, so return negative
                intra_dist = sum([distance(data[point_idx], data[center_idx]) 
                                for point_idx, center_idx in assignments])
                return -intra_dist
            
            elif self.quality_metric == 'combined':
                # Combine silhouette score and intra-cluster distance
                sil_score = silhouette_score(data, cluster_labels)
                intra_dist = sum([distance(data[point_idx], data[center_idx]) 
                                for point_idx, center_idx in assignments])
                # Normalize and combine (adjust weights as needed)
                return sil_score - (intra_dist / len(data)) / 10
            
        except Exception as e:
            # Return poor score if evaluation fails
            return -float('inf')
        
        return -float('inf')
    
    def fit(self, data):
        """
        Fit the optimized k-centers algorithm to the data.
        
        Args:
            data (list): Points in the dataset
        """
        self.data = data
        n = len(data)
        
        if n == 0:
            self.centers = []
            self.costs = []
            self.best_score = -float('inf')
            return
        
        if self.k >= n:
            self.centers = list(range(n))
            self.costs = [0] * n
            self.best_score = 1.0
            return
        
        best_centers = None
        best_score = -float('inf')
        best_costs = None
        
        # Try multiple initializations
        for init_run in range(self.n_init):
            self._set_random_state(init_run)
            
            # Initialize centers
            centers = self._initialize_centers(data, self.k)
            
            # Refine centers iteratively
            refined_centers, converged = self._refine_centers(data, centers)
            
            # Assign points and evaluate
            assignments = self._assign_points(data, refined_centers)
            score = self._evaluate_clustering(data, refined_centers, assignments)
            
            # Calculate costs (max distance from any point to its center)
            costs = []
            for center_idx in refined_centers:
                max_dist = max([distance(data[center_idx], data[point_idx]) 
                              for point_idx, assigned_center in assignments 
                              if assigned_center == center_idx])
                costs.append(max_dist)
            
            # Keep best result
            if score > best_score:
                best_score = score
                best_centers = refined_centers
                best_costs = costs
        
        self.centers = best_centers
        self.costs = best_costs
        self.best_score = best_score
    
    def assign(self):
        """
        Assign every point in the dataset to the closest center.
        
        Returns:
            list: List of (point_index, center_index) tuples
        """
        if self.centers is None or self.data is None:
            return []
        
        return self._assign_points(self.data, self.centers)