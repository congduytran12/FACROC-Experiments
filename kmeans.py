import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from data_loader import load_dataset

def kmeans_clustering(input_file, output_file, k=2):
    """
    Apply K-means clustering to a dataset.
    
    Args:
        input_file: Path to input CSV file in data-encoded directory
        output_file: Path to output CSV file in clustering directory  
        k: Number of clusters
    
    Returns:
        DataFrame with clustering results
    """
    print("=" * 60)
    print("K-MEANS CLUSTERING")
    print("=" * 60)
    
    dataset_name = os.path.basename(input_file)
    print(f"Processing dataset: {dataset_name}")
    
    # load data
    print(f"\n1. Loading dataset...")
    data, blues, reds, df, protected_attr_col = load_dataset(input_file)
    
    # convert to numpy array
    print(f"\n2. Preparing features...")
    X = np.array(data)
    print(f"   - Feature shape: {X.shape}")
    print(f"   - Number of features: {X.shape[1]}")
    
    # apply k-means clustering
    print(f"\n3. Applying K-means clustering with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    # convert cluster labels to 1-indexed
    cluster_labels = cluster_labels + 1
    
    print(f"   - Clustering completed")
    print(f"   - Inertia (WCSS): {kmeans.inertia_:.2f}")
    
    # create result dataframe
    print("\n4. Creating results...")
    results = []
    for i in range(len(df)):
        cluster_id = cluster_labels[i]
        protected_attr = df.iloc[i][protected_attr_col]
        results.append({
            'id': i + 1,  
            'cluster_id': cluster_id,
            'protected_attribute': protected_attr
        })
    
    results_df = pd.DataFrame(results)
    
    # analyze cluster distribution
    print("\n5. Cluster Analysis:")
    cluster_dist = results_df['cluster_id'].value_counts().sort_index()
    print(f"   - Cluster distribution: {dict(cluster_dist)}")

    # analyze protected attribute distribution by cluster
    attr_cluster_dist = results_df.groupby(['cluster_id', 'protected_attribute']).size().unstack(fill_value=0)
    print(f"   - Protected attribute distribution by cluster:")
    print(attr_cluster_dist)
    
    # save results
    print(f"\n6. Saving results to {output_file}...")
    results_df.to_csv(output_file, index=False)
    
    print("\nK-means clustering completed successfully!")
    print("=" * 60)
    
    return results_df


def process_all_datasets():
    """
    Process all datasets in data-encoded directory with K-means clustering.
    """
    dataset_configs = {
        'ricci-encode.csv': {'k': 10},
        'student-mat-encode.csv': {'k': 9}, 
        'xAPI-Edu-data-encode.csv': {'k': 11},
        'student-por-encode.csv': {'k': 9}, 
        'german-encode.csv': {'k': 2}, 
        'pisa-encode.csv': {'k': 9}, 
        'compas-encode.csv': {'k': 7}, 
        'oulad-encode.csv': {'k': 9}, 
        'credit-encode.csv': {'k': 2}, 
        'adult-encode.csv': {'k': 2}
    }
    
    input_dir = "data-encoded"
    output_dir = "clustering"
    
    # create output directory if not exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Processing all datasets with K-means clustering...")
    print("=" * 80)
    
    results_summary = []
    
    for dataset_file, config in dataset_configs.items():
        input_file = os.path.join(input_dir, dataset_file)
        base_name = dataset_file.replace('-encode.csv', '')
        output_file = os.path.join(output_dir, f"{base_name}-clustering.csv")
        
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found, skipping...")
            continue
            
        print(f"\n\nProcessing {dataset_file}...")
        print(f"Configuration: k={config['k']}")
        
        try:
            results = kmeans_clustering(
                input_file=input_file,
                output_file=output_file,
                k=config['k']
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
    
    print("All datasets processed successfully!")
    print("Results saved in clustering/ directory with '-clustering.csv' suffix")


if __name__ == "__main__":
    process_all_datasets()
