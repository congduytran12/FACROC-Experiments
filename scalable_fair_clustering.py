"""
Scalable Fair Clustering Implementation

This module implements scalable fair clustering techniques based on the 
"Scalable Fair Clustering" paper by Backurs et al. (2019).

Key improvements over MCF approach:
- HST (Hierarchically Separated Tree) embedding for O(n log n) complexity
- Greedy fairlet construction eliminating distance threshold parameters
- K-median clustering with k-means++ initialization
- Local search optimization for better clustering quality
"""

import pandas as pd
import numpy as np
import os
import time
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from data_loader import load_dataset
from utils import distance, calculate_balance, calculate_silhouette_score
import random
from typing import List, Tuple, Dict, Optional


class HSTreeEmbedding:
    """
    Hierarchically Separated Tree (HST) embedding for scalable fairlet construction.
    
    Creates a binary tree structure that enables efficient fair clustering
    by recursively partitioning the data space using k-means.
    """
    
    def __init__(self, data: List[List[float]], blues: List[int], reds: List[int], 
                 max_leaf_size: int = 10):
        """
        Initialize HST embedding.
        
        Args:
            data: List of data points (features)
            blues: Indices of majority group points
            reds: Indices of minority group points  
            max_leaf_size: Maximum points per leaf node
        """
        self.data = np.array(data)
        self.blues = blues
        self.reds = reds
        self.max_leaf_size = max_leaf_size
        self.tree = None
        
    def build_tree(self) -> Dict:
        """
        Build the HST using recursive binary partitioning.
        
        Returns:
            Root node of the HST
        """
        print("   Building HST embedding...")
        start_time = time.time()
        
        # Combine all points with their group labels
        all_indices = self.blues + self.reds
        all_points = self.data[all_indices]
        group_labels = ['blue'] * len(self.blues) + ['red'] * len(self.reds)
        
        self.tree = self._build_node(all_indices, all_points, group_labels, depth=0)
        
        build_time = time.time() - start_time
        print(f"   HST built in {build_time:.3f} seconds")
        
        return self.tree
    
    def _build_node(self, indices: List[int], points: np.ndarray, 
                   labels: List[str], depth: int) -> Dict:
        """
        Recursively build a tree node.
        
        Args:
            indices: Point indices in this node
            points: Point coordinates 
            labels: Group labels for points
            depth: Current tree depth
            
        Returns:
            Tree node dictionary
        """
        node = {
            'indices': indices,
            'points': points,
            'labels': labels,
            'depth': depth,
            'is_leaf': False,
            'left': None,
            'right': None,
            'center': np.mean(points, axis=0) if len(points) > 0 else None
        }
        
        # Check if this should be a leaf node
        if len(indices) <= self.max_leaf_size or len(np.unique(labels)) == 1:
            node['is_leaf'] = True
            return node
            
        # Use k-means for binary split
        if len(points) >= 2:
            try:
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(points)
                
                # Split points based on clustering
                left_mask = cluster_labels == 0
                right_mask = cluster_labels == 1
                
                if np.sum(left_mask) > 0 and np.sum(right_mask) > 0:
                    left_indices = [indices[i] for i in range(len(indices)) if left_mask[i]]
                    right_indices = [indices[i] for i in range(len(indices)) if right_mask[i]]
                    
                    left_points = points[left_mask]
                    right_points = points[right_mask]
                    
                    left_labels = [labels[i] for i in range(len(labels)) if left_mask[i]]
                    right_labels = [labels[i] for i in range(len(labels)) if right_mask[i]]
                    
                    # Recursively build child nodes
                    node['left'] = self._build_node(left_indices, left_points, left_labels, depth + 1)
                    node['right'] = self._build_node(right_indices, right_points, right_labels, depth + 1)
                else:
                    node['is_leaf'] = True
            except:
                node['is_leaf'] = True
        else:
            node['is_leaf'] = True
            
        return node


class ScalableFairletDecomposition:
    """
    Scalable fairlet decomposition using HST embedding and greedy matching.
    
    Achieves O(n log n) time complexity without distance threshold parameters.
    """
    
    def __init__(self, blues: List[int], reds: List[int], t: int, data: List[List[float]]):
        """
        Initialize scalable fairlet decomposition.
        
        Args:
            blues: Indices of majority group points
            reds: Indices of minority group points
            t: Fairness ratio (1:t)
            data: List of data points
        """
        self.blues = blues
        self.reds = reds
        self.t = t
        self.data = np.array(data)
        self.hst = None
        
        # Ensure blues >= reds as required
        if len(blues) < len(reds):
            self.blues, self.reds = reds, blues
            
    def decompose(self) -> Tuple[List[List[int]], List[int], List[float]]:
        """
        Perform scalable fairlet decomposition.
        
        Returns:
            Tuple of (fairlets, fairlet_centers, fairlet_costs)
        """
        print("   Creating HST embedding...")
        self.hst = HSTreeEmbedding(self.data, self.blues, self.reds)
        tree = self.hst.build_tree()
        
        print("   Performing greedy fairlet construction...")
        start_time = time.time()
        fairlets = self._construct_fairlets(tree)
        construct_time = time.time() - start_time
        print(f"   Fairlets constructed in {construct_time:.3f} seconds")
        
        # Compute fairlet centers and costs
        fairlet_centers = []
        fairlet_costs = []
        
        for fairlet in fairlets:
            if len(fairlet) > 0:
                # Find center as point with minimum maximum distance to other points in fairlet
                min_cost = float('inf')
                best_center = fairlet[0]
                
                for candidate in fairlet:
                    max_dist = 0
                    for other in fairlet:
                        if candidate != other:
                            dist = distance(self.data[candidate], self.data[other])
                            max_dist = max(max_dist, dist)
                    
                    if max_dist < min_cost:
                        min_cost = max_dist
                        best_center = candidate
                
                fairlet_centers.append(best_center)
                fairlet_costs.append(min_cost)
        
        print(f"   Created {len(fairlets)} fairlets with avg size {np.mean([len(f) for f in fairlets]):.2f}")
        
        return fairlets, fairlet_centers, fairlet_costs
    
    def _construct_fairlets(self, node: Dict) -> List[List[int]]:
        """
        Construct fairlets using greedy matching within tree structure.
        
        Args:
            node: Current tree node
            
        Returns:
            List of fairlets (each fairlet is list of point indices)
        """
        if node['is_leaf']:
            return self._construct_leaf_fairlets(node)
        
        fairlets = []
        
        # Process child nodes
        if node['left'] is not None:
            fairlets.extend(self._construct_fairlets(node['left']))
        if node['right'] is not None:
            fairlets.extend(self._construct_fairlets(node['right']))
            
        return fairlets
    
    def _construct_leaf_fairlets(self, node: Dict) -> List[List[int]]:
        """
        Construct fairlets within a leaf node using greedy matching.
        
        Args:
            node: Leaf node
            
        Returns:
            List of fairlets from this leaf
        """
        indices = node['indices']
        labels = node['labels']
        
        # Separate blues and reds in this leaf
        blues_in_leaf = [idx for idx, label in zip(indices, labels) if label == 'blue']
        reds_in_leaf = [idx for idx, label in zip(indices, labels) if label == 'red']
        
        # Shuffle for randomness in greedy matching
        random.shuffle(blues_in_leaf)
        random.shuffle(reds_in_leaf)
        
        fairlets = []
        
        # Create (1,t)-balanced fairlets greedily
        blue_idx = 0
        red_idx = 0
        
        while blue_idx < len(blues_in_leaf) and red_idx < len(reds_in_leaf):
            fairlet = [reds_in_leaf[red_idx]]  # Start with 1 red
            red_idx += 1
            
            # Add up to t blues
            blues_added = 0
            while blues_added < self.t and blue_idx < len(blues_in_leaf):
                fairlet.append(blues_in_leaf[blue_idx])
                blue_idx += 1
                blues_added += 1
                
            fairlets.append(fairlet)
        
        # Handle remaining blues (create fairlets with only blues)
        while blue_idx < len(blues_in_leaf):
            fairlet = []
            blues_added = 0
            while blues_added < (self.t + 1) and blue_idx < len(blues_in_leaf):
                fairlet.append(blues_in_leaf[blue_idx])
                blue_idx += 1
                blues_added += 1
            if fairlet:
                fairlets.append(fairlet)
        
        # Handle remaining reds (add to existing fairlets or create new ones)
        fairlet_idx = 0
        while red_idx < len(reds_in_leaf):
            if fairlet_idx < len(fairlets):
                fairlets[fairlet_idx].append(reds_in_leaf[red_idx])
            else:
                fairlets.append([reds_in_leaf[red_idx]])
            red_idx += 1
            fairlet_idx += 1
        
        return fairlets


class ImprovedKMedian:
    """
    Improved K-median clustering with k-means++ initialization and local search.
    
    Provides better clustering quality compared to K-centers approach.
    """
    
    def __init__(self, k: int = 2, max_iters: int = 100, tol: float = 1e-4):
        """
        Initialize K-median clustering.
        
        Args:
            k: Number of clusters
            max_iters: Maximum iterations for local search
            tol: Convergence tolerance
        """
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centers = None
        self.data = None
        
    def fit(self, data: List[List[float]]) -> 'ImprovedKMedian':
        """
        Fit K-median clustering to data.
        
        Args:
            data: List of data points
            
        Returns:
            Self for method chaining
        """
        self.data = np.array(data)
        
        # Use k-means++ initialization for better starting points
        self.centers = self._kmeans_plus_plus_init()
        
        # Local search optimization
        self._local_search()
        
        return self
    
    def _kmeans_plus_plus_init(self) -> List[int]:
        """
        K-means++ initialization for better starting centers.
        
        Returns:
            List of center indices
        """
        n_points = len(self.data)
        centers = []
        
        # Choose first center randomly
        centers.append(random.randint(0, n_points - 1))
        
        # Choose remaining centers with probability proportional to squared distance
        for _ in range(self.k - 1):
            distances = []
            for i in range(n_points):
                min_dist = float('inf')
                for center_idx in centers:
                    dist = distance(self.data[i], self.data[center_idx])
                    min_dist = min(min_dist, dist)
                distances.append(min_dist ** 2)
            
            # Choose next center with probability proportional to squared distance
            total_dist = sum(distances)
            if total_dist > 0:
                prob = random.random() * total_dist
                cumsum = 0
                for i, dist in enumerate(distances):
                    cumsum += dist
                    if cumsum >= prob:
                        centers.append(i)
                        break
            else:
                # Fallback: choose randomly
                remaining = set(range(n_points)) - set(centers)
                if remaining:
                    centers.append(random.choice(list(remaining)))
        
        return centers
    
    def _local_search(self):
        """
        Local search optimization to improve K-median solution.
        """
        prev_cost = float('inf')
        
        for iteration in range(self.max_iters):
            # Assign points to nearest centers
            assignments = self.assign()
            
            # Update centers to geometric medians
            new_centers = []
            for cluster_id in range(1, self.k + 1):
                cluster_points = [assignments[i][0] for i in range(len(assignments)) 
                                if assignments[i][1] == cluster_id - 1]
                
                if cluster_points:
                    # Find geometric median (point that minimizes sum of distances)
                    best_center = cluster_points[0]
                    min_cost = float('inf')
                    
                    for candidate in cluster_points:
                        total_dist = sum(distance(self.data[candidate], self.data[p]) 
                                       for p in cluster_points)
                        if total_dist < min_cost:
                            min_cost = total_dist
                            best_center = candidate
                    
                    new_centers.append(best_center)
                else:
                    # Keep original center if no points assigned
                    new_centers.append(self.centers[cluster_id - 1] if cluster_id - 1 < len(self.centers) else 0)
            
            # Check convergence
            current_cost = self._compute_cost(new_centers)
            if abs(prev_cost - current_cost) < self.tol:
                break
                
            self.centers = new_centers
            prev_cost = current_cost
    
    def _compute_cost(self, centers: List[int]) -> float:
        """
        Compute total K-median cost for given centers.
        
        Args:
            centers: List of center indices
            
        Returns:
            Total cost (sum of distances to nearest centers)
        """
        total_cost = 0
        for point in self.data:
            min_dist = float('inf')
            for center_idx in centers:
                dist = distance(point, self.data[center_idx])
                min_dist = min(min_dist, dist)
            total_cost += min_dist
        return total_cost
    
    def assign(self) -> List[Tuple[int, int]]:
        """
        Assign points to nearest centers.
        
        Returns:
            List of (point_index, center_index) tuples
        """
        assignments = []
        for i, point in enumerate(self.data):
            min_dist = float('inf')
            best_center = 0
            
            for j, center_idx in enumerate(self.centers):
                dist = distance(point, self.data[center_idx])
                if dist < min_dist:
                    min_dist = dist
                    best_center = j
                    
            assignments.append((i, best_center))
        
        return assignments


def scalable_fair_clustering_dataset(input_file: str, output_file: str, 
                                   k: int = 2, t: int = 3, 
                                   approximation_factor: float = 2.0) -> pd.DataFrame:
    """
    Apply scalable fair clustering to a dataset.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file  
        k: Number of clusters
        t: Fairness ratio (1:t)
        approximation_factor: Approximation factor for clustering quality
        
    Returns:
        DataFrame with clustering results
    """
    print("=" * 60)
    print("SCALABLE FAIR CLUSTERING WITH HST EMBEDDING")
    print("=" * 60)
    
    dataset_name = os.path.basename(input_file)
    print(f"Processing dataset: {dataset_name}")
    print(f"Configuration: k={k}, t={t}, approximation_factor={approximation_factor}")
    
    # Record total start time
    total_start_time = time.time()
    
    # Load data
    print(f"\n1. Loading dataset...")
    data, blues, reds, df, protected_attr_col = load_dataset(input_file)
    
    # Initialize scalable fairlet decomposition
    print(f"\n2. Initializing scalable fairlet decomposition...")
    print(f"   - Fairness ratio: (1, {t})")
    print(f"   - No distance threshold required")
    
    decomp = ScalableFairletDecomposition(blues, reds, t, data)
    
    # Perform fairlet decomposition
    print("\n3. Computing scalable fairlet decomposition...")
    fairlets, fairlet_centers, fairlet_costs = decomp.decompose()
    
    print(f"   - Number of fairlets created: {len(fairlets)}")
    if fairlets:
        print(f"   - Average fairlet size: {np.mean([len(f) for f in fairlets]):.2f}")
        print(f"   - Average fairlet cost: {np.mean(fairlet_costs):.4f}")
    
    # Apply improved K-median clustering on fairlet centers
    print(f"\n4. Applying improved K-median clustering with k={k}...")
    fairlet_center_data = [data[center] for center in fairlet_centers]
    
    kmedian = ImprovedKMedian(k=k)
    kmedian.fit(fairlet_center_data)
    fairlet_cluster_mapping = kmedian.assign()
    
    print(f"   - Fairlet centers clustered into {k} clusters")
    
    # Assign all data points to clusters
    print("\n5. Assigning data points to clusters...")
    point_to_cluster = {}
    
    # Create mapping from fairlet center index to cluster ID
    fairlet_to_cluster = {}
    
    for mapping in fairlet_cluster_mapping:
        fairlet_center_idx = mapping[0]
        assigned_cluster_idx = mapping[1]
        cluster_id = assigned_cluster_idx + 1  # 1-indexed clusters
        fairlet_to_cluster[fairlet_center_idx] = cluster_id
    
    # Assign all points in each fairlet to the same cluster
    for fairlet_idx, fairlet in enumerate(fairlets):
        cluster_id = fairlet_to_cluster.get(fairlet_idx, 1)
        
        for point_idx in fairlet:
            point_to_cluster[point_idx] = cluster_id
    
    # Create results dataframe
    print("\n6. Creating results...")
    results = []
    for i in range(len(df)):
        cluster_id = point_to_cluster.get(i, 1)
        protected_attr = df.iloc[i][protected_attr_col]
        results.append({
            'id': i + 1,
            'cluster_id': cluster_id,
            'protected_attribute': protected_attr
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate performance metrics
    total_time = time.time() - total_start_time
    
    print("\n7. Performance Analysis:")
    print(f"   - Total runtime: {total_time:.3f} seconds")
    print(f"   - Time complexity: O(n log n)")
    print(f"   - Dataset size: {len(df)} points")
    print(f"   - Points per second: {len(df) / total_time:.1f}")
    
    # Analyze clustering quality
    print("\n8. Cluster Analysis:")
    cluster_dist = results_df['cluster_id'].value_counts().sort_index()
    print(f"   - Cluster distribution: {dict(cluster_dist)}")
    
    # Protected attribute distribution by cluster
    attr_cluster_dist = results_df.groupby(['cluster_id', 'protected_attribute']).size().unstack(fill_value=0)
    print(f"   - Protected attribute distribution by cluster:")
    print(attr_cluster_dist)
    
    # Calculate fairness metrics
    balance_ratio = calculate_balance(results_df, 'protected_attribute')
    print(f"   - Overall balance ratio: {balance_ratio:.3f}")
    
    # Calculate silhouette score if possible
    try:
        cluster_labels = results_df['cluster_id'].values
        silhouette = calculate_silhouette_score(data, cluster_labels)
        print(f"   - Silhouette score: {silhouette:.3f}")
    except Exception as e:
        print(f"   - Silhouette score: Could not calculate ({e})")
    
    # Save results
    print(f"\n9. Saving results to {output_file}...")
    results_df.to_csv(output_file, index=False)
    
    print("\nScalable fair clustering completed successfully!")
    print("=" * 60)
    
    return results_df


def process_all_datasets_scalable():
    """
    Process all datasets with scalable fair clustering algorithms.
    """
    # Dataset configurations (no distance thresholds needed)
    dataset_configs = {
        'student-mat-encode.csv': {'k': 9, 't': 2},
        'student-por-encode.csv': {'k': 9, 't': 2},
        'german-encode.csv': {'k': 2, 't': 3},
        'compas-encode.csv': {'k': 7, 't': 2},
        'credit-encode.csv': {'k': 2, 't': 2},
        'adult-encode.csv': {'k': 2, 't': 3}
    }
    
    input_dir = "data-encoded"
    output_dir = "clustering-scalable"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Processing all datasets with scalable fair clustering...")
    print("=" * 80)
    
    results_summary = []
    performance_metrics = []
    
    for dataset_file, config in dataset_configs.items():
        input_file = os.path.join(input_dir, dataset_file)
        output_file = os.path.join(output_dir, dataset_file.replace('-encode.csv', '-clustering.csv'))
        
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found, skipping...")
            continue
            
        print(f"\n\nProcessing {dataset_file}...")
        print(f"Configuration: k={config['k']}, t={config['t']}")
        
        try:
            start_time = time.time()
            results = scalable_fair_clustering_dataset(
                input_file=input_file,
                output_file=output_file,
                k=config['k'],
                t=config['t']
            )
            processing_time = time.time() - start_time
            
            # Calculate metrics
            balance = calculate_balance(results, 'protected_attribute')
            
            try:
                # Load original data for silhouette calculation
                from data_loader import load_dataset
                data, _, _, _, _ = load_dataset(input_file)
                silhouette = calculate_silhouette_score(data, results['cluster_id'].values)
            except:
                silhouette = 0.0
            
            results_summary.append({
                'dataset': dataset_file,
                'output_file': output_file,
                'total_points': len(results),
                'clusters': sorted(results['cluster_id'].unique()),
                'num_clusters': len(results['cluster_id'].unique()),
                'processing_time': processing_time,
                'balance_ratio': balance,
                'silhouette_score': silhouette
            })
            
            performance_metrics.append({
                'dataset': dataset_file,
                'points': len(results),
                'time': processing_time,
                'points_per_second': len(results) / processing_time,
                'balance': balance,
                'silhouette': silhouette
            })
            
            print(f"Successfully processed {dataset_file}")
            
        except Exception as e:
            print(f"Error processing {dataset_file}: {str(e)}")
            continue
    
    # Print comprehensive summary
    print("\n" + "=" * 80)
    print("SCALABLE FAIR CLUSTERING SUMMARY")
    print("=" * 80)
    
    for summary in results_summary:
        print(f"Dataset: {summary['dataset']}")
        print(f"  Output: {summary['output_file']}")
        print(f"  Points: {summary['total_points']}")
        print(f"  Clusters: {summary['num_clusters']} clusters {summary['clusters']}")
        print(f"  Processing time: {summary['processing_time']:.3f}s")
        print(f"  Balance ratio: {summary['balance_ratio']:.3f}")
        print(f"  Silhouette score: {summary['silhouette_score']:.3f}")
        print()
    
    # Performance comparison
    print("PERFORMANCE METRICS:")
    print("-" * 60)
    print(f"{'Dataset':<25} {'Points':<8} {'Time(s)':<8} {'Points/s':<10} {'Balance':<8} {'Silhouette':<10}")
    print("-" * 60)
    
    for metric in performance_metrics:
        print(f"{metric['dataset']:<25} {metric['points']:<8} {metric['time']:<8.2f} "
              f"{metric['points_per_second']:<10.1f} {metric['balance']:<8.3f} {metric['silhouette']:<10.3f}")
    
    print(f"\nAll {len(results_summary)} datasets processed successfully!")
    print(f"Results saved in {output_dir}/ directory")
    
    # Calculate and display overall performance statistics
    if performance_metrics:
        total_points = sum(m['points'] for m in performance_metrics)
        total_time = sum(m['time'] for m in performance_metrics)
        avg_balance = np.mean([m['balance'] for m in performance_metrics])
        avg_silhouette = np.mean([m['silhouette'] for m in performance_metrics])
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total points processed: {total_points:,}")
        print(f"  Total processing time: {total_time:.2f}s")
        print(f"  Average processing rate: {total_points / total_time:.1f} points/second")
        print(f"  Average balance ratio: {avg_balance:.3f}")
        print(f"  Average silhouette score: {avg_silhouette:.3f}")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    process_all_datasets_scalable()