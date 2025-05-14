import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics import roc_auc_score, roc_curve

class AUCC:
    def __init__(self, aucc, tpr, fpr):
        self.aucc = aucc
        self.tpr = tpr
        self.fpr = fpr


def aucc(partition, dataset=None, distance=None, distance_method='euclidean', return_rates=False):
    # input validation
    if dataset is None and distance is None:
        raise ValueError('You need to specify a distance matrix or a dataset.')
    
    if partition is None:
        raise ValueError('You need to specify a hard partition - clustering solution.')
    
    if distance is not None and dataset is not None:
        raise ValueError('You can only specify a dataset or a distance, not both.')
    
    # convert partition to integers
    partition = np.asarray(partition)
    if not np.issubdtype(partition.dtype, np.integer):
        # convert to factor-like integers
        unique_vals = np.unique(partition)
        partition_map = {val: i+1 for i, val in enumerate(unique_vals)}
        partition = np.array([partition_map[val] for val in partition])
    
    # compute distance
    if dataset is not None and distance is None:
        if len(dataset) != len(partition):
            raise ValueError('The number of objects has to be the same from partition.')
        # compute pairwise distance
        distance = pdist(dataset, metric=distance_method)
    elif dataset is None and distance is not None:
        distance = np.asarray(distance)
        expected_len = len(partition) * (len(partition) - 1) // 2
        if len(distance) != expected_len:
            raise ValueError(f'Distance and partitions sizes don\'t match. Expected length: {expected_len}')
    
    # compute pairwise partition distances
    pred = pdist(partition.reshape(-1, 1), metric='hamming')
    
    # normalize distances
    distance = (distance - np.min(distance)) / (np.max(distance) - np.min(distance))
    
    # compute ROC and AUC
    if not return_rates:
        # compute AUC directly
        r = roc_auc_score(1 - pred, 1 - distance)
    else:
        # get full ROC curve
        fpr, tpr, _ = roc_curve(1 - pred, 1 - distance)
        aucc_value = roc_auc_score(1 - pred, 1 - distance)
        r = AUCC(aucc_value, tpr, fpr)
    
    return r
