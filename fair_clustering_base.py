import pandas as pd
import numpy as np
import os
from mcf_fairlet_decomposition import MCFFairletDecomposition
from data_loader import load_dataset
from kcenters import KCenters
from utils import calculate_balance, reassign_clusters_for_quality

def fair_clustering_dataset(input_file, output_file, k=2, t=3, distance_threshold=50):
    print("=" * 60)
    print("FAIR CLUSTERING WITH MCF FAIRLET DECOMPOSITION")
    print("=" * 60)
    
    dataset_name = os.path.basename(input_file)
    print(f"Processing dataset: {dataset_name}")
    
    # load data
    print(f"\n1. Loading dataset...")
    data, blues, reds, df, protected_attr_col = load_dataset(input_file)
    
    # init MCF fairlet decomposition
    print(f"\n2. Initializing MCF fairlet decomposition...")
    print(f"   - Fairness ratio: (1, {t})")
    print(f"   - Distance threshold: {distance_threshold}")
    
    mcf = MCFFairletDecomposition(blues, reds, t, distance_threshold, data)
    
    # compute distances and build graph
    print("\n3. Computing distances and building flow network...")
    mcf.compute_distances()
    mcf.build_graph()
    
    # decompose into fairlets
    print("\n4. Computing fairlet decomposition...")
    fairlets, fairlet_centers, fairlet_costs = mcf.decompose()
    
    print(f"   - Number of fairlets created: {len(fairlets)}")
    print(f"   - Average fairlet size: {np.mean([len(f) for f in fairlets]):.2f}")
    print(f"   - Average fairlet cost: {np.mean(fairlet_costs):.4f}")
    
    # apply k-centers clustering on fairlet centers
    print(f"\n5. Applying K-centers clustering with k={k}...")
    fairlet_center_data = [data[center] for center in fairlet_centers]
    
    kcenters = KCenters(k=k)
    kcenters.fit(fairlet_center_data)
    fairlet_cluster_mapping = kcenters.assign()
    
    print(f"   - Fairlet centers clustered into {k} clusters")
    
    # assign all data points to clusters
    print("\n6. Assigning data points to clusters...")
    point_to_cluster = {}

    # create mapping from fairlet center index to cluster ID
    fairlet_to_cluster = {}
    
    # get unique cluster center indices 
    cluster_centers = set()
    for mapping in fairlet_cluster_mapping:
        cluster_centers.add(mapping[1])
    
    # sort cluster centers 
    cluster_centers = sorted(list(cluster_centers))
    print(f"   - Found {len(cluster_centers)} clusters: {cluster_centers}")
    
    # create mapping from cluster center index to cluster ID
    cluster_center_to_id = {}
    for i, center_idx in enumerate(cluster_centers):
        cluster_center_to_id[center_idx] = i + 1
    
    # map each fairlet to its cluster ID
    for mapping in fairlet_cluster_mapping:
        fairlet_center_idx = mapping[0]  
        assigned_cluster_center_idx = mapping[1]  
        cluster_id = cluster_center_to_id[assigned_cluster_center_idx]
        fairlet_to_cluster[fairlet_center_idx] = cluster_id
    
    # assign all points in each fairlet to the same cluster
    for fairlet_idx, fairlet in enumerate(fairlets):
        # get cluster ID for this fairlet
        cluster_id = fairlet_to_cluster.get(fairlet_idx, 1)
        
        # assign all points in this fairlet to the same cluster
        for point_idx in fairlet:
            point_to_cluster[point_idx] = cluster_id

    # create results dataframe
    print("\n8. Creating results...")
    results = []
    for i in range(len(df)):
        cluster_id = point_to_cluster.get(i, 1) 
        protected_attr = df.iloc[i][protected_attr_col]
        results.append({
            'id': i + 1,  
            'cluster_id': cluster_id,
            'protected_attribute': protected_attr
        })
    
    results_df = pd.DataFrame(results)
    
    # verify cluster distribution
    print("\n9. Cluster Analysis:")
    cluster_dist = results_df['cluster_id'].value_counts().sort_index()
    print(f"   - Cluster distribution: {dict(cluster_dist)}")
    
    gender_cluster_dist = results_df.groupby(['cluster_id', 'protected_attribute']).size().unstack(fill_value=0)
    print(f"   - Protected attribute distribution by cluster:")
    print(gender_cluster_dist)
    
    # calculate fairness metrics
    unique_clusters = sorted(results_df['cluster_id'].unique())
    for cluster in unique_clusters:
        cluster_data = results_df[results_df['cluster_id'] == cluster]  
        if len(cluster_data) > 0:
            attr_counts = cluster_data['protected_attribute'].value_counts()
            if len(attr_counts) > 1:
                values = list(attr_counts.values)
                balance = min(values) / max(values)
                print(f"   - Cluster {cluster} balance ratio: {balance:.3f}")
            else:
                print(f"   - Cluster {cluster}: Only one protected attribute value present")
    
    # calculate initial balance
    initial_balance = calculate_balance(results_df, 'protected_attribute')
    print(f"\n   - Overall initial balance: {initial_balance:.4f}")
    
    # apply cluster reassignment
    print(f"\n10. Applying cluster reassignment for quality improvement...")
    balance_threshold = 1.0 / t  
    data_array = np.array(data)
    
    try:
        reassigned_df = reassign_clusters_for_quality(
            data=data_array,
            clustering_df=results_df,
            balance_threshold=balance_threshold,
            protected_attr_col='protected_attribute',
            max_iterations=100,
            verbose=True
        )
        
        # update results with reassigned clusters 
        results_df = reassigned_df
        
    except Exception as e:
        print(f"   Warning: Cluster reassignment failed: {e}")
        print("   Continuing with original clustering...")
    
    # save results
    print(f"\n11. Saving results to {output_file}...")
    results_df.to_csv(output_file, index=False)
    
    print("\n12. Fair clustering completed successfully!")
    print("=" * 60)
    
    return results_df


def process_all_datasets():
    """
    Process all datasets in data-encoded directory with appropriate cluster counts.
    """
    # Define cluster counts for each dataset
    dataset_configs = {
        'student-mat-encode.csv': {'k': 9, 't': 2, 'distance_threshold': 7}, # [13, 7, 10, 14]
        'xAPI-Edu-data-encode.csv': {'k': 11, 't': 2, 'distance_threshold': 81}, # [81, 43, 62, 76]
        'student-por-encode.csv': {'k': 9, 't': 2, 'distance_threshold': 6}, # [10, 6, 8]
        'german-encode.csv': {'k': 2, 't': 3, 'distance_threshold': 559}, # [3006, 559, 1368, 3311]
        'pisa-encode.csv': {'k': 9, 't': 2, 'distance_threshold': 1126}, # [1126, 341, 664, 1087]
        'compas-encode.csv': {'k': 7, 't': 2, 'distance_threshold': 106}, # [559, 106, 252, 556]
        'oulad-encode.csv': {'k': 9, 't': 2, 'distance_threshold': 21}, # [48, 2, 24, 44]
        'credit-encode.csv': {'k': 2, 't': 2, 'distance_threshold': 92332}, # [260441, 92332, 161158, 267435]
        'adult-encode.csv': {'k': 2, 't': 3, 'distance_threshold': 12} # [41, 12, 21, 5507]
    }
    
    input_dir = "data-encoded"
    output_dir = "clustering"
    
    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Processing all datasets with fair clustering...")
    print("=" * 80)
    
    results_summary = []
    
    for dataset_file, config in dataset_configs.items():
        input_file = os.path.join(input_dir, dataset_file)
        output_file = os.path.join(output_dir, dataset_file.replace('-encode.csv', '-clustering.csv'))
        
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found, skipping...")
            continue
            
        print(f"\n\nProcessing {dataset_file}...")
        print(f"Configuration: k={config['k']}, t={config['t']}, distance_threshold={config['distance_threshold']}")
        
        try:
            results = fair_clustering_dataset(
                input_file=input_file,
                output_file=output_file,
                k=config['k'],
                t=config['t'],
                distance_threshold=config['distance_threshold']
            )
            
            results_summary.append({
                'dataset': dataset_file,
                'output_file': output_file,
                'total_points': len(results),
                'clusters': sorted(results['cluster_id'].unique()),
                'num_clusters': len(results['cluster_id'].unique())
            })
            
            print(f"Successfully processed {dataset_file}")
            
        except Exception as e:
            print(f"Error processing {dataset_file}: {str(e)}")
            continue
    
    # Print summary
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    
    for summary in results_summary:
        print(f"Dataset: {summary['dataset']}")
        print(f"  Output: {summary['output_file']}")
        print(f"  Points: {summary['total_points']}")
        print(f"  Clusters: {summary['num_clusters']} clusters {summary['clusters']}")
        print()


if __name__ == "__main__":
    process_all_datasets()
