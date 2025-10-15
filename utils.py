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
    
    try:
        return silhouette_score(data, cluster_labels)
    except Exception as e:
        print(f"Error calculating silhouette score: {e}")
        return 0.0
    
def select_fairlet_centers(fairlets, data):
    centers = []
    for fairlet in fairlets:
        if len(fairlet) == 1:
            centers.append(fairlet[0])
        else:
            # select most representative point (closest to fairlet centroid)
            centroid = np.mean([data[i] for i in fairlet], axis=0)
            best_point = min(fairlet, key=lambda p: distance(data[p], centroid))
            centers.append(best_point)
    return centers

def unified_cluster_optimization(point_to_cluster, data, blues, reds):
    unique_clusters = list(set(point_to_cluster.values()))
    max_iterations = 100
    
    for iteration in range(max_iterations):
        moves_made = 0
        
        # calculate cluster info
        cluster_info = {}
        for cluster_id in unique_clusters:
            points = [i for i, c in point_to_cluster.items() if c == cluster_id]
            if len(points) < 2:
                continue
                
            centroid = np.mean([data[i] for i in points], axis=0)
            avg_dist = np.mean([distance(data[i], centroid) for i in points])
            
            cluster_info[cluster_id] = {
                'points': points,
                'centroid': centroid, 
                'avg_dist': avg_dist,
                'protected': sum(1 for p in points if p in reds),
                'non_protected': sum(1 for p in points if p in blues)
            }
        
        # aggressive outlier reassignment
        for cluster_id, info in cluster_info.items():            
            for point in info['points']:
                point_dist = distance(data[point], info['centroid'])
                if point_dist > info['avg_dist'] * 1.01:  # outlier threshold
                    
                    # find better cluster
                    best_target = None
                    best_improvement = 0
                    
                    for target_id, target_info in cluster_info.items():
                        if target_id == cluster_id:
                            continue
                            
                        target_dist = distance(data[point], target_info['centroid'])
                        improvement = point_dist - target_dist
                        
                        # check fairness constraint
                        point_is_protected = point in reds
                        
                        if point_is_protected:
                            new_target_balance = min(target_info['protected'] + 1, target_info['non_protected']) / max(target_info['protected'] + 1, target_info['non_protected'])
                        else:
                            new_target_balance = min(target_info['protected'], target_info['non_protected'] + 1) / max(target_info['protected'], target_info['non_protected'] + 1)
                        
                        # accept if significant clustering improvement with reasonable fairness
                        if improvement > 0.05 and new_target_balance > 0.6:  
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_target = target_id
                    
                    if best_target is not None:
                        point_to_cluster[point] = best_target
                        moves_made += 1
                        
                        # update cluster info
                        cluster_info[cluster_id]['points'].remove(point)
                        cluster_info[best_target]['points'].append(point)
                        
                        if point in reds:
                            cluster_info[cluster_id]['protected'] -= 1
                            cluster_info[best_target]['protected'] += 1
                        else:
                            cluster_info[cluster_id]['non_protected'] -= 1
                            cluster_info[best_target]['non_protected'] += 1
        
        print(f"     Iteration {iteration + 1}: {moves_made} moves")
        
        if moves_made == 0:
            break
    
    # merge very small clusters (< 3 points) 
    cluster_sizes = {cid: len([i for i, c in point_to_cluster.items() if c == cid]) 
                    for cid in unique_clusters}
    
    small_clusters = [cid for cid, size in cluster_sizes.items() if size < 3]
    large_clusters = [cid for cid, size in cluster_sizes.items() if size >= 10]
    
    merges = 0
    for small_cluster in small_clusters:
        small_points = [i for i, c in point_to_cluster.items() if c == small_cluster]
        
        if not small_points:
            continue

        # find closest large cluster
        best_target = None
        best_dist = float('inf')
        
        small_centroid = np.mean([data[i] for i in small_points], axis=0)
        
        for large_cluster in large_clusters:
            large_points = [i for i, c in point_to_cluster.items() if c == large_cluster]
            large_centroid = np.mean([data[i] for i in large_points], axis=0)
            
            dist = distance(small_centroid, large_centroid)
            if dist < best_dist:
                best_dist = dist
                best_target = large_cluster
        
        if best_target is not None:
            for point in small_points:
                point_to_cluster[point] = best_target
            merges += 1
    
    print(f"   - Merged {merges} small clusters")
    
    return point_to_cluster

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
