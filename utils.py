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
