import numpy as np
from sklearn_extra.cluster import KMedoids
import time
import os
import pandas as pd
from utils import calculate_balance, reassign_clusters_for_quality
from tree_fairlet_decomposition import build_quadtree, tree_fairlet_decomposition, fair_kmedian_cost

DATASET_CONFIGS = {
    'student-mat-encode.csv': {'k': 9, 'p': 2, 'q': 5},    
    'student-por-encode.csv': {'k': 9, 'p': 2, 'q': 5},
    'german-encode.csv': {'k': 2, 'p': 2, 'q': 5},
    'compas-encode.csv': {'k': 7, 'p': 2, 'q': 5},
    'credit-encode.csv': {'k': 2, 'p': 2, 'q': 5},
    'adult-encode.csv': {'k': 2, 'p': 2, 'q': 5}
}

PROTECTED_ATTRIBUTES = {
    'student-mat-encode.csv': {'column': 'gender', 'mapping': {'F': 0, 'M': 1}},
    'student-por-encode.csv': {'column': 'gender', 'mapping': {'F': 0, 'M': 1}},
    'german-encode.csv': {'column': 'sex', 'mapping': {'M': 0, 'F': 1}},
    'compas-encode.csv': {'column': 'race', 'mapping': {'White': 0, 'Non-White': 1}},
    'credit-encode.csv': {'column': 'SEX', 'mapping': {'M': 0, 'F': 1}},
    'adult-encode.csv': {'column': 'gender', 'mapping': {'Male': 0, 'Female': 1}}
}

DATA_FOLDER = 'data-encoded'
CLUSTERING_FOLDER = 'clustering'
 
 
if __name__ == "__main__":
    for dataset_file in DATASET_CONFIGS.keys():
        print(f"Processing {dataset_file}")
        config = DATASET_CONFIGS[dataset_file]
        k = config['k']
        p = config['p']
        q = config['q']
        protected_info = PROTECTED_ATTRIBUTES[dataset_file]
        protected_column = protected_info['column']
        mapping = protected_info['mapping']
        
        # load dataset
        file_path = os.path.join(DATA_FOLDER, dataset_file)
        df = pd.read_csv(file_path)
        
        # get protected attribute
        protected_values = df[protected_column].map(mapping).values
        colors = protected_values.astype(int)
        
        # extract features 
        feature_cols = [col for col in df.columns if col != protected_column]
        points = df[feature_cols].values
        
        n_points = len(points)
        dimension = points.shape[1]
        dataset = points
        
        # reset global variables
        FAIRLETS = []
        FAIRLET_CENTERS = []
        
        print("Number of data points:", n_points)
        print("Dimension:", dimension)
        print("Balance:", p, q)
        
        print("Constructing tree...")
        fairlet_s = time.time()
        root = build_quadtree(dataset)
        
        print("Doing fair clustering...")
        cost = tree_fairlet_decomposition(p, q, root, dataset, colors)
        fairlet_e = time.time()
        
        print("Fairlet decomposition cost:", cost)
        
        print("Doing k-median clustering on fairlet centers...")
        fairlet_center_pt = np.array([dataset[index] for index in FAIRLET_CENTERS])
        
        # run k-medoids clustering
        cluster_s = time.time()
        kmedoids = KMedoids(n_clusters=k, metric='euclidean', random_state=42)
        kmedoids.fit(fairlet_center_pt)

        # get indices of medoids in fairlet center points
        medoid_indices = kmedoids.medoid_indices_
        cluster_e = time.time()
        
        # indices of center points in dataset
        centroids = [FAIRLET_CENTERS[index] for index in medoid_indices]
        
        print("Computing fair k-median cost...")
        kmedian_cost = fair_kmedian_cost(centroids, dataset)
        print("Fairlet decomposition cost:", cost)
        print("k-Median cost:", kmedian_cost)
        
        # cluster assignment
        print("Computing cluster assignments...")
        fairlet_to_cluster = {}
        for i in range(len(FAIRLETS)):
            distances = [np.linalg.norm(dataset[centroids[j], :] - dataset[FAIRLET_CENTERS[i], :]) for j in range(k)]
            cluster_id = np.argmin(distances) + 1 # 1-based
            fairlet_to_cluster[i] = cluster_id

        cluster_assignments = [0] * n_points
        for fairlet_idx, fairlet_points in enumerate(FAIRLETS):
            cluster_id = fairlet_to_cluster[fairlet_idx]
            for point_idx in fairlet_points:
                cluster_assignments[point_idx] = cluster_id
        
        # save results
        output_df = pd.DataFrame({
            'id': range(1, n_points + 1),
            'cluster_id': cluster_assignments,
            'protected_attribute': df[protected_column].values
        })
        
        # calculate initial balance
        initial_balance = calculate_balance(output_df, 'protected_attribute')
        print(f"Initial balance: {initial_balance:.4f}")
        
        # apply cluster reassignment
        print("\nApplying cluster reassignment for quality improvement...")
        balance_threshold = p / q  
        
        try:
            reassigned_df = reassign_clusters_for_quality(
                data=dataset,
                clustering_df=output_df,
                balance_threshold=balance_threshold,
                protected_attr_col='protected_attribute',
                max_iterations=100,
                verbose=True
            )
            
            # update results with reassigned clusters 
            output_df = reassigned_df
            
        except Exception as e:
            print(f"Warning: Cluster reassignment failed: {e}")
            print("Continuing with original clustering...")
        
        output_file = dataset_file.replace('-encode.csv', '-clustering.csv')
        output_path = os.path.join(CLUSTERING_FOLDER, output_file)
        output_df.to_csv(output_path, index=False)
        
        print(f"Saved clustering results to {output_path}")
        
        print("-" * 50)