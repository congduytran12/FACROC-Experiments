import numpy as np
from sklearn.metrics import silhouette_score

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

def refine_assignments_within_balance(dataset, cluster_assignments, colors, centroids, p, q, max_iterations=5):
    """
    Refine cluster assignments while maintaining fairness constraints.
    Points can be reassigned if it improves quality without violating balance.
    """
    n_points = len(cluster_assignments)
    k = len(centroids)
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(n_points):
            current_cluster = cluster_assignments[i]
            current_dist = np.linalg.norm(dataset[i, :] - dataset[centroids[current_cluster - 1], :])
            
            # find closest cluster
            best_cluster = current_cluster
            best_dist = current_dist
            
            for j in range(k):
                new_cluster = j + 1
                if new_cluster == current_cluster:
                    continue
                    
                new_dist = np.linalg.norm(dataset[i, :] - dataset[centroids[j], :])
                
                if new_dist < best_dist:
                    # check if reassignment maintains balance
                    if would_maintain_balance(cluster_assignments, colors, i, current_cluster, new_cluster, p, q):
                        best_cluster = new_cluster
                        best_dist = new_dist
                        
            if best_cluster != current_cluster:
                cluster_assignments[i] = best_cluster
                improved = True
    
    return cluster_assignments

def would_maintain_balance(assignments, colors, point_idx, from_cluster, to_cluster, p, q):
    """Check if moving a point maintains fairness balance in both clusters."""
    point_color = colors[point_idx]
    
    # count colors in affected clusters
    from_counts = {0: 0, 1: 0}
    to_counts = {0: 0, 1: 0}
    
    for i, cluster_id in enumerate(assignments):
        if cluster_id == from_cluster:
            from_counts[colors[i]] += 1
        elif cluster_id == to_cluster:
            to_counts[colors[i]] += 1
    
    # simulate the move
    from_counts[point_color] -= 1
    to_counts[point_color] += 1
    
    # check balance for both clusters
    for counts in [from_counts, to_counts]:
        if counts[0] == 0 or counts[1] == 0:
            continue  # allow empty color in a cluster
        ratio = min(counts[0], counts[1]) / max(counts[0], counts[1])
        if ratio < p / q: 
            return False
    
    return True
    
    try:
        return silhouette_score(data, cluster_labels)
    except Exception as e:
        print(f"Error calculating silhouette score: {e}")
        return 0.0
