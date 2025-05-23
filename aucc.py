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
    
    # compute pairwise labels
    pairwise_labels = 1 - pdist(partition.reshape(-1, 1), metric='hamming')
    
    # normalize distances
    if np.max(distance) != np.min(distance):
        distance_norm = (distance - np.min(distance)) / (np.max(distance) - np.min(distance))
    else:
        distance_norm = np.zeros_like(distance)
        warnings.warn("All distances are equal, normalized distances will be zero")
    
    # convert to similarity
    pairwise_distances = 1 - distance_norm
    
    # calculate AUCC
    aucc_value = roc_auc_score(pairwise_labels, pairwise_distances)
    
    if not return_rates:
        return aucc_value
    else:
        # sort by distance
        sorted_indices = np.argsort(pairwise_distances)[::-1] 
        sorted_distances = pairwise_distances[sorted_indices]
        sorted_labels = pairwise_labels[sorted_indices]
        
        # calculate TP and FP
        positive_count = np.sum(sorted_labels == 1)
        negative_count = len(sorted_labels) - positive_count

        # initialize for aggregating points with same distance
        tpr_values = []
        fpr_values = []
        
        # add point for (0,0)
        tpr_values.append(0)
        fpr_values.append(0)
        
        tp_count = 0
        fp_count = 0
        
        # process all points
        for i, (dist, label) in enumerate(zip(sorted_distances, sorted_labels)):
            if label == 1:
                tp_count += 1
            else:
                fp_count += 1

            if i == len(sorted_distances) - 1 or sorted_distances[i] != sorted_distances[i+1]:
                tpr = tp_count / positive_count if positive_count > 0 else 0
                fpr = fp_count / negative_count if negative_count > 0 else 0
                
                tpr_values.append(tpr)
                fpr_values.append(fpr)
        
        # add point for (1,1) 
        if fpr_values[-1] < 1 or tpr_values[-1] < 1:
            tpr_values.append(1)
            fpr_values.append(1)
        
        result = {
            'aucc': aucc_value,
            'tpr': np.array(tpr_values),
            'fpr': np.array(fpr_values)
        }
        
        return result