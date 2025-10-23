import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import math

def distance(a, b, order=2):
	"""
	Calculates the specified norm between two vectors.
	
	Args:
		a (list) : First vector
		b (list) : Second vector
		order (int) : Order of the norm to be calculated as distance
	
	Returns:
		Resultant norm value
	"""
	assert len(a) == len(b), "Length of the vectors for distance don't match."
	return np.linalg.norm(x=np.array(a)-np.array(b), ord=order)

def calculate_balance(clustering, protected_attr_col='protected_attribute'):
    """
    Calculate the balance of clustering, returning the smallest balance value among all clusters.
    Balance is defined as min(protected)/max(protected) for each cluster.
    """
    unique_clusters = sorted(clustering['cluster_id'].unique())
    min_balance = float('inf')
    
    for cluster in unique_clusters:
        cluster_data = clustering[clustering['cluster_id'] == cluster]  
        if len(cluster_data) > 0:
            attr_counts = cluster_data[protected_attr_col].value_counts()
            if len(attr_counts) > 1:
                values = list(attr_counts.values)
                balance = min(values) / max(values)
                min_balance = min(min_balance, balance)
            else:
                # if only one protected attribute value is present, balance is 0
                min_balance = 0.0
                break
    
    return min_balance if min_balance != float('inf') else 0.0

def calculate_silhouette_score(data, cluster_labels):
    """
    Calculate the silhouette score for the clustering.
    """
    if len(np.unique(cluster_labels)) < 2:
        return 0.0  # silhouette score is undefined for single cluster
    
    try:
        return silhouette_score(data, cluster_labels)
    except Exception as e:
        print(f"Error calculating silhouette score: {e}")
        return 0.0

def calculate_proportionality(data, cluster_labels, audit_centers=None):
    """
    Calculate the Rho-Proportionality of a clustering.
    
    Rho-Proportionality measures fairness by calculating the maximum ratio of distances
    that would incentivize a blocking coalition to deviate to an alternative center.
    Higher rho values indicate better proportionality (more stable clustering).
    """
    data = np.asarray(data)
    cluster_labels = np.asarray(cluster_labels)
    n_samples = len(data)
    k = len(np.unique(cluster_labels))
    
    if k < 2:
        return 1.0  # proportionality is trivial for single cluster
    
    # Calculate cluster centers from the current clustering
    unique_labels = np.unique(cluster_labels)
    k_centers = []
    for label in unique_labels:
        cluster_points = data[cluster_labels == label]
        center = np.mean(cluster_points, axis=0)
        k_centers.append(center)
    k_centers = np.array(k_centers)
    
    # If no audit centers provided, use all data points
    if audit_centers is None:
        audit_centers = data
    else:
        audit_centers = np.asarray(audit_centers)
    
    # Compute the nearest center in k_centers for each point
    distances_to_centers = cdist(data, k_centers, metric='euclidean')
    nearest_center_idx = np.argmin(distances_to_centers, axis=1)
    nearest_distances = distances_to_centers[np.arange(n_samples), nearest_center_idx]
    
    max_rho = 1.0
    
    # For each potential alternative center
    for potential_center in audit_centers:
        # Calculate distances from all points to this potential center
        distances_to_potential = np.linalg.norm(data - potential_center, axis=1)
        
        # Calculate the ratio for each client
        rho_list = []
        for i in range(n_samples):
            if distances_to_potential[i] <= 0:  # already at the center
                continue
            if nearest_distances[i] <= 0:  # already at assigned center
                continue
            ratio = float(nearest_distances[i]) / distances_to_potential[i]
            rho_list.append(ratio)
        
        if len(rho_list) < n_samples / k:
            # Insufficient number of deviating clients
            continue
        
        # Calculate the rho value - the (n/k)-th largest ratio
        rho_list.sort(reverse=True)
        threshold_idx = int(math.ceil(n_samples / k)) - 1
        if threshold_idx < len(rho_list):
            rho = rho_list[threshold_idx]
            max_rho = max(rho, max_rho)
    
    return 1/ max_rho

def calculate_cluster_centers(data, clustering_df, protected_attr_col='protected_attribute'):
    """
    Calculate the center (mean) of each cluster.
    """
    centers = {}
    unique_clusters = sorted(clustering_df['cluster_id'].unique())
    
    for cluster_id in unique_clusters:
        cluster_points = clustering_df[clustering_df['cluster_id'] == cluster_id]
        cluster_indices = (cluster_points['id'] - 1).values 
        cluster_data = data[cluster_indices]
        centers[cluster_id] = np.mean(cluster_data, axis=0)
    
    return centers

def calculate_point_to_cluster_distances(data, clustering_df, cluster_centers):
    """
    Calculate the distance from each point to all cluster centers.
    """
    point_distances = {}
    cluster_ids = sorted(cluster_centers.keys())
    centers_array = np.array([cluster_centers[cid] for cid in cluster_ids])
 
    # calculate all distances at once
    distances = cdist(data, centers_array, metric='euclidean')
    
    for i, row in clustering_df.iterrows():
        point_id = row['id']
        point_idx = point_id - 1  
        point_distances[point_id] = {
            cluster_ids[j]: distances[point_idx, j] 
            for j in range(len(cluster_ids))
        }
    
    return point_distances

def check_balance_constraint(cluster_data, protected_attr_col='protected_attribute', balance_threshold=0.0):
    """
    Check if a cluster satisfies the balance constraint.
    """
    if len(cluster_data) == 0:
        return True, 1.0
    
    attr_counts = cluster_data[protected_attr_col].value_counts()
    
    if len(attr_counts) < 2:
        return balance_threshold <= 0.0, 0.0
    
    values = list(attr_counts.values)
    balance = min(values) / max(values)
    
    return balance >= balance_threshold, balance


def simulate_reassignment(clustering_df, point_id, from_cluster, to_cluster, 
                         protected_attr_col='protected_attribute', balance_threshold=0.0):
    """
    Simulate reassigning a point and check if balance constraints are maintained.
    """
    # simulate removal from source cluster
    from_cluster_data = clustering_df[clustering_df['cluster_id'] == from_cluster]
    from_cluster_after = from_cluster_data[from_cluster_data['id'] != point_id]
    from_valid, from_balance = check_balance_constraint(from_cluster_after, protected_attr_col, balance_threshold)
    
    # simulate addition to target cluster
    to_cluster_data = clustering_df[clustering_df['cluster_id'] == to_cluster]
    # create temporary row for added point
    point_row = clustering_df[clustering_df['id'] == point_id].copy()
    point_row['cluster_id'] = to_cluster
    to_cluster_after = pd.concat([to_cluster_data, point_row], ignore_index=True)
    to_valid, to_balance = check_balance_constraint(to_cluster_after, protected_attr_col, balance_threshold)
    
    is_valid = from_valid and to_valid
    
    return is_valid, from_balance, to_balance

def reassign_clusters_for_quality(data, clustering_df, balance_threshold=0.0, 
                                  protected_attr_col='protected_attribute',
                                  max_iterations=100, verbose=True):
    """
    Reassign points to clusters to improve AUCC while maintaining balance constraints.
    
    This function uses a greedy approach to iteratively reassign points to nearby clusters
    if doing so improves clustering quality (reduces within-cluster distance) while
    ensuring that the balance constraint is not violated in any cluster.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("CLUSTER REASSIGNMENT FOR QUALITY IMPROVEMENT")
        print("=" * 60)
        print(f"Balance threshold: {balance_threshold:.4f}")
        print(f"Max iterations: {max_iterations}")

    clustering = clustering_df.copy()

    initial_balance = calculate_balance(clustering, protected_attr_col)
    if verbose:
        print(f"\nInitial balance: {initial_balance:.4f}")
    
    improvements = 0
    
    for iteration in range(max_iterations):
        # recalculate cluster centers
        cluster_centers = calculate_cluster_centers(data, clustering, protected_attr_col)

        # calculate distances from each point to all cluster centers
        point_distances = calculate_point_to_cluster_distances(data, clustering, cluster_centers)

        # track if any improvement was made
        improved_this_iteration = False
        
        # try to reassign each point
        for _, row in clustering.iterrows():
            point_id = row['id']
            current_cluster = row['cluster_id']
            
            # get distances to all clusters
            distances = point_distances[point_id]
            current_distance = distances[current_cluster]
            
            # find clusters that are closer than current one
            closer_clusters = [
                (cid, dist) for cid, dist in distances.items() 
                if dist < current_distance and cid != current_cluster
            ]
            
            if not closer_clusters:
                continue
            
            # sort by distance
            closer_clusters.sort(key=lambda x: x[1])

            # try to reassign to closest valid cluster
            for target_cluster, target_distance in closer_clusters:
                # check if maintain balance constraints
                is_valid, from_balance, to_balance = simulate_reassignment(
                    clustering, point_id, current_cluster, target_cluster,
                    protected_attr_col, balance_threshold
                )
                
                if is_valid:
                    # perform reassignment
                    clustering.loc[clustering['id'] == point_id, 'cluster_id'] = target_cluster
                    improvements += 1
                    improved_this_iteration = True
                    
                    if verbose and improvements % 10 == 0:
                        print(f"  Iteration {iteration + 1}: {improvements} reassignments made")
                    
                    break  
                
        if not improved_this_iteration:
            if verbose:
                print(f"\nConverged after {iteration + 1} iterations")
            break

    final_balance = calculate_balance(clustering, protected_attr_col)
    
    if verbose:
        print(f"\nReassignment complete:")
        print(f"  Total reassignments: {improvements}")
        print(f"  Final balance: {final_balance:.4f}")
        print(f"  Balance maintained: {final_balance >= balance_threshold}")
        
        # show cluster distribution
        print("\nFinal cluster distribution:")
        cluster_dist = clustering['cluster_id'].value_counts().sort_index()
        print(f"  {dict(cluster_dist)}")
        
        gender_cluster_dist = clustering.groupby(['cluster_id', protected_attr_col]).size().unstack(fill_value=0)
        print("\nProtected attribute distribution by cluster:")
        print(gender_cluster_dist)
        
        # display balance for each cluster
        unique_clusters = sorted(clustering['cluster_id'].unique())
        print("\nCluster balance ratios:")
        for cluster in unique_clusters:
            cluster_data = clustering[clustering['cluster_id'] == cluster]
            _, balance = check_balance_constraint(cluster_data, protected_attr_col, 0.0)
            print(f"  Cluster {cluster}: {balance:.4f}")
        
        print("=" * 60)
    
    return clustering

