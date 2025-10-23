import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings

def aucc(partition, dataset=None, distance=None, distance_method='euclidean', return_rates=False):
    if dataset is None and distance is None:
        raise ValueError("You need to specify a distance matrix or a dataset.")
    
    if partition is None:
        raise ValueError("You need to specify a hard partition - clustering solution.")
    
    if distance is not None and dataset is not None:
        raise ValueError("You can only specify a dataset or a distance, not both.")

    # convert partition to integer codes
    partition = np.asarray(partition)
    if not np.issubdtype(partition.dtype, np.integer):
        le = LabelEncoder()
        partition = le.fit_transform(partition)    
    
    if dataset is not None and distance is None:
        # ensure dataset is converted to float
        try:
            dataset = np.asarray(dataset, dtype=float)  
        except ValueError:
            raise ValueError("Dataset contains non-numeric values that cannot be converted to float")
            
        if len(dataset) != len(partition):
            raise ValueError("The number of objects has to be the same as in partition.")
        
        # compute pairwise distances
        distance = pdist(dataset, metric=distance_method)
    elif dataset is None and distance is not None:
        distance = np.asarray(distance)
        expected_size = len(partition) * (len(partition) - 1) // 2
        if len(distance) != expected_size:
            raise ValueError(f"Distance matrix size ({len(distance)}) doesn't match expected size ({expected_size})")
    
    # compute pairwise labels more efficiently using broadcasting
    n = len(partition)
    partition_matrix = partition[:, None]
    pairwise_labels = (partition_matrix == partition_matrix.T)
    # extract upper triangle (excluding diagonal) and flatten
    pairwise_labels = pairwise_labels[np.triu_indices(n, k=1)].astype(np.float64)
    
    # normalize distances
    dist_min = np.min(distance)
    dist_max = np.max(distance)
    
    if dist_max != dist_min:
        distance_norm = (distance - dist_min) / (dist_max - dist_min)
    else:
        distance_norm = np.zeros_like(distance)
        warnings.warn("All distances are equal, normalized distances will be zero")
    
    # convert to similarity (higher similarity = lower distance)
    pairwise_similarities = 1 - distance_norm
    
    # calculate AUCC
    aucc_value = roc_auc_score(pairwise_labels, pairwise_similarities)
    
    if not return_rates:
        return aucc_value
    else:
        # sort by similarity in descending order 
        sorted_indices = np.argsort(-pairwise_similarities)
        sorted_labels = pairwise_labels[sorted_indices]
        
        # calculate total positives and negatives
        total_positives = np.sum(sorted_labels)
        total_negatives = len(sorted_labels) - total_positives
        
        if total_positives == 0 or total_negatives == 0:
            warnings.warn("All pairs are either same-cluster or different-cluster")
            return {
                'aucc': aucc_value,
                'tpr': np.array([0.0, 1.0]),
                'fpr': np.array([0.0, 1.0])
            }
        
        # calculate cumulative TP and FP
        tp_counts = np.concatenate([[0], np.cumsum(sorted_labels)])
        fp_counts = np.concatenate([[0], np.cumsum(1 - sorted_labels)])
        
        # calculate rates 
        tpr_values = tp_counts / total_positives
        fpr_values = fp_counts / total_negatives
        
        # remove duplicate points
        unique_mask = np.concatenate([[True], (np.diff(fpr_values) != 0) | (np.diff(tpr_values) != 0)])
        tpr_values = tpr_values[unique_mask]
        fpr_values = fpr_values[unique_mask]
        
        result = {
            'aucc': aucc_value,
            'tpr': tpr_values,
            'fpr': fpr_values
        }
        
        return result
