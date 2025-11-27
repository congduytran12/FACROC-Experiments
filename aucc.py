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
        partition = le. fit_transform(partition)
    
    n = len(partition)
    
    if dataset is not None:
        # ensure dataset is converted to float
        try:
            dataset = np.asarray(dataset, dtype=float)  
        except ValueError:
            raise ValueError("Dataset contains non-numeric values that cannot be converted to float")
            
        if len(dataset) != n:
            raise ValueError("The number of objects has to be the same as in partition.")
        
        # compute pairwise distances
        distance = pdist(dataset, metric=distance_method)
    else:
        distance = np.asarray(distance, dtype=np.float64)
        expected_size = n * (n - 1) // 2
        if len(distance) != expected_size:
            raise ValueError(f"Distance matrix size ({len(distance)}) doesn't match expected size ({expected_size})")
    
    # compute pairwise labels
    i_indices, j_indices = np.triu_indices(n, k=1)
    pairwise_labels = (partition[i_indices] == partition[j_indices])
    
    # normalize distances
    dist_min = distance.min()
    dist_max = distance.max()
    
    if dist_max != dist_min:
        pairwise_similarities = 1.0 - (distance - dist_min) / (dist_max - dist_min)
    else:
        pairwise_similarities = np.zeros_like(distance, dtype=np.float64)
        warnings.warn("All distances are equal, normalized distances will be zero")
    
    # calculate AUCC
    aucc_value = roc_auc_score(pairwise_labels, pairwise_similarities)
    
    if not return_rates:
        return aucc_value

    # sort by similarity in descending order 
    sorted_indices = np.argsort(pairwise_similarities)[::-1]
    sorted_labels = pairwise_labels[sorted_indices]
    
    # calculate total positives and negatives
    total_positives = sorted_labels.sum()
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
    fp_counts = np.concatenate([[0], np.cumsum(~sorted_labels)])
    
    # calculate rates 
    tpr_values = tp_counts / total_positives
    fpr_values = fp_counts / total_negatives
    
    # remove duplicate points
    fpr_diff = np.diff(fpr_values, prepend=np.nan)
    tpr_diff = np.diff(tpr_values, prepend=np.nan)
    unique_mask = (fpr_diff != 0) | (tpr_diff != 0)
    
    result = {
        'aucc': aucc_value,
        'tpr': tpr_values[unique_mask],
        'fpr': fpr_values[unique_mask]
    }

    return result
