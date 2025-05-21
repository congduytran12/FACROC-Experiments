from sklearn.utils import check_X_y
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.cluster._unsupervised import check_number_of_labels
from sklearn.metrics import roc_auc_score

from scipy.spatial.distance import pdist

import numpy as np

def aucc(X, labels, metric='euclidean'):
    """Computes AUCC (Area Under the (ROC) Curve for Clustering).

    Maximum score is 1.0. The higher the score, the better the partition.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A list of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.

    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.
    
    metric : distance employed between data points. Default is euclidean.

    Returns
    -------
    score: float
        The resulting AUCC score.

    References
    ----------
    .. [1] Pablo A. Jaskowiak, Ivan G. Costa, and Ricardo J. G. B. Campello. 2022. 
    The area under the ROC curve as a measure of clustering quality. 
    Data Min. Knowl. Discov. 36, 3 (May 2022), 1219–1245. 
    https://doi.org/10.1007/s10618-022-00829-0
    """
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, _ = X.shape
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, n_samples)

    n_pairs = n_samples*(n_samples-1)//2

    if np.matrix(labels).shape[0] == 1:
        pairwise_labels    = 1 - pdist(np.matrix(labels).transpose(),'hamming')
    else:
        pairwise_labels    = 1 - pdist(np.matrix(labels),'hamming')
    tmpdist = pdist(X,metric)
    tmpdist = (tmpdist - np.min(tmpdist))/(np.max(tmpdist) - np.min(tmpdist))
    pairwise_distances = 1 - tmpdist

    return roc_auc_score(pairwise_labels,pairwise_distances)