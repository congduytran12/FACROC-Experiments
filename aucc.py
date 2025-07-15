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
    
    # convert to similarity (higher similarity = lower distance)
    pairwise_similarities = 1 - distance_norm
    
    # calculate AUCC
    aucc_value = roc_auc_score(pairwise_labels, pairwise_similarities)
    
    if not return_rates:
        return aucc_value
    else:
        # sort by similarity (descending order)
        sorted_indices = np.argsort(pairwise_similarities)[::-1]
        sorted_similarities = pairwise_similarities[sorted_indices]
        sorted_labels = pairwise_labels[sorted_indices]
        
        # calculate TP and FP
        positive_count = np.sum(sorted_labels == 1)
        negative_count = len(sorted_labels) - positive_count

        # initialize for aggregating points with same similarity
        tpr_values = []
        fpr_values = []
        
        # add point for (0,0) - no points classified as positive yet
        tpr_values.append(0)
        fpr_values.append(0)
        
        tp_count = 0
        fp_count = 0
        
        # group points with same similarity
        unique_similarities = np.unique(sorted_similarities)[::-1]  
        
        # process similarities from high to low 
        for sim_thresh in unique_similarities:
            # find all points with exact similarity value
            mask = sorted_similarities == sim_thresh
            sim_labels = sorted_labels[mask]
            
            # add these points to positive predictions
            tp_count += np.sum(sim_labels == 1)
            fp_count += np.sum(sim_labels == 0)
            
            # calculate rates
            tpr = tp_count / positive_count if positive_count > 0 else 0
            fpr = fp_count / negative_count if negative_count > 0 else 0
            
            tpr_values.append(tpr)
            fpr_values.append(fpr)
        
        # ensure we end at (1,1) 
        if fpr_values[-1] < 1 or tpr_values[-1] < 1:
            tpr_values.append(1)
            fpr_values.append(1)
            
        tpr_values = np.array(tpr_values)
        fpr_values = np.array(fpr_values)

        for i in range(1, len(tpr_values)):
            if tpr_values[i] < tpr_values[i-1]:
                tpr_values[i] = tpr_values[i-1]
            if fpr_values[i] < fpr_values[i-1]:
                fpr_values[i] = fpr_values[i-1]
        
        result = {
            'aucc': aucc_value,
            'tpr': tpr_values,
            'fpr': fpr_values
        }
        
        return result