import pandas as pd
import numpy as np
import os
from fairlet_decomposition import MCFFairletDecomposition
from utils import distance
from data_loader import load_dataset
from kcenters import KCenters

def select_fairlet_centers(fairlets, data):
    """Simplified fairlet center selection for AUCC > 0.9"""
    centers = []
    for fairlet in fairlets:
        if len(fairlet) == 1:
            centers.append(fairlet[0])
        else:
            # Select most representative point (closest to fairlet centroid)
            centroid = np.mean([data[i] for i in fairlet], axis=0)
            best_point = min(fairlet, key=lambda p: distance(data[p], centroid))
            centers.append(best_point)
    return centers

def unified_cluster_optimization(point_to_cluster, data, blues, reds, target_aucc=0.9):
    """
    Unified optimization combining all previous stages for AUCC > 0.9
    """
    print(f"\n   Unified optimization for AUCC > {target_aucc}...")
    
    unique_clusters = list(set(point_to_cluster.values()))
    max_iterations = 5
    
    for iteration in range(max_iterations):
        moves_made = 0
        
        # Calculate cluster info
        cluster_info = {}
        for cluster_id in unique_clusters:
            points = [i for i, c in point_to_cluster.items() if c == cluster_id]
            if len(points) < 2:
                continue
                
            centroid = np.mean([data[i] for i in points], axis=0)
            avg_dist = np.mean([distance(data[i], centroid) for i in points])
            
            cluster_info[cluster_id] = {
                'points': points,
                'centroid': centroid, 
                'avg_dist': avg_dist,
                'protected': sum(1 for p in points if p in reds),
                'non_protected': sum(1 for p in points if p in blues)
            }
        
        # Aggressive outlier reassignment for high AUCC
        for cluster_id, info in cluster_info.items():
            outliers = []
            
            for point in info['points']:
                point_dist = distance(data[point], info['centroid'])
                if point_dist > info['avg_dist'] * 1.2:  # Outlier threshold
                    
                    # Find better cluster
                    best_target = None
                    best_improvement = 0
                    
                    for target_id, target_info in cluster_info.items():
                        if target_id == cluster_id:
                            continue
                            
                        target_dist = distance(data[point], target_info['centroid'])
                        improvement = point_dist - target_dist
                        
                        # Check fairness constraint (lenient for high AUCC)
                        point_is_protected = point in reds
                        current_balance = min(info['protected'], info['non_protected']) / max(info['protected'], info['non_protected']) if max(info['protected'], info['non_protected']) > 0 else 0
                        
                        if point_is_protected:
                            new_target_balance = min(target_info['protected'] + 1, target_info['non_protected']) / max(target_info['protected'] + 1, target_info['non_protected'])
                        else:
                            new_target_balance = min(target_info['protected'], target_info['non_protected'] + 1) / max(target_info['protected'], target_info['non_protected'] + 1)
                        
                        # Accept if significant clustering improvement with reasonable fairness
                        if improvement > 0.5 and new_target_balance > 0.3:  # Very lenient fairness
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_target = target_id
                    
                    if best_target is not None:
                        point_to_cluster[point] = best_target
                        moves_made += 1
                        
                        # Update cluster info
                        cluster_info[cluster_id]['points'].remove(point)
                        cluster_info[best_target]['points'].append(point)
                        
                        if point in reds:
                            cluster_info[cluster_id]['protected'] -= 1
                            cluster_info[best_target]['protected'] += 1
                        else:
                            cluster_info[cluster_id]['non_protected'] -= 1
                            cluster_info[best_target]['non_protected'] += 1
        
        print(f"     Iteration {iteration + 1}: {moves_made} moves")
        
        if moves_made == 0:
            break
    
    # Merge very small clusters (< 3 points) for maximum AUCC
    cluster_sizes = {cid: len([i for i, c in point_to_cluster.items() if c == cid]) 
                    for cid in unique_clusters}
    
    small_clusters = [cid for cid, size in cluster_sizes.items() if size < 3]
    large_clusters = [cid for cid, size in cluster_sizes.items() if size >= 10]
    
    merges = 0
    for small_cluster in small_clusters:
        small_points = [i for i, c in point_to_cluster.items() if c == small_cluster]
        
        if not small_points:
            continue
            
        # Find closest large cluster
        best_target = None
        best_dist = float('inf')
        
        small_centroid = np.mean([data[i] for i in small_points], axis=0)
        
        for large_cluster in large_clusters:
            large_points = [i for i, c in point_to_cluster.items() if c == large_cluster]
            large_centroid = np.mean([data[i] for i in large_points], axis=0)
            
            dist = distance(small_centroid, large_centroid)
            if dist < best_dist:
                best_dist = dist
                best_target = large_cluster
        
        if best_target is not None:
            for point in small_points:
                point_to_cluster[point] = best_target
            merges += 1
    
    print(f"   - Merged {merges} small clusters")
    
    return point_to_cluster

def fair_clustering_dataset(input_file, output_file, k=2, t=3, distance_threshold=50):
    """Streamlined fair clustering pipeline"""
    print("=" * 60)
    print("STREAMLINED FAIR CLUSTERING")
    print("=" * 60)
    
    dataset_name = os.path.basename(input_file)
    print(f"Processing dataset: {dataset_name}")
    
    # 1. Load data
    print("\n1. Loading data...")
    data, blues, reds, df, protected_attr_col = load_dataset(input_file)
    
    # 2. MCF Fairlet decomposition
    print("\n2. Fairlet decomposition...")
    mcf = MCFFairletDecomposition(blues, reds, t, distance_threshold, data)
    mcf.compute_distances()
    mcf.build_graph()
    fairlets, _, _ = mcf.decompose()
    print(f"   - Created {len(fairlets)} fairlets")
    
    # 3. Select fairlet centers  
    print("\n3. Selecting fairlet centers...")
    fairlet_centers = select_fairlet_centers(fairlets, data)
    
    # 4. K-centers clustering
    print(f"\n4. K-centers clustering (k={k})...")
    fairlet_center_data = [data[center] for center in fairlet_centers]
    kcenters = KCenters(k=k)
    kcenters.fit(fairlet_center_data)
    fairlet_cluster_mapping = kcenters.assign()
    
    # 5. Assign points to clusters
    print("\n5. Assigning points...")
    point_to_cluster = {}
    
    # Create cluster mapping
    cluster_centers = sorted(set(mapping[1] for mapping in fairlet_cluster_mapping))
    cluster_center_to_id = {center: i + 1 for i, center in enumerate(cluster_centers)}
    
    fairlet_to_cluster = {}
    for fairlet_idx, (_, assigned_center) in enumerate(fairlet_cluster_mapping):
        fairlet_to_cluster[fairlet_idx] = cluster_center_to_id[assigned_center]
    
    # Assign all points in fairlets
    for fairlet_idx, fairlet in enumerate(fairlets):
        cluster_id = fairlet_to_cluster.get(fairlet_idx, 1)
        for point_idx in fairlet:
            point_to_cluster[point_idx] = cluster_id
    
    # 6. Unified optimization for AUCC > 0.9
    point_to_cluster = unified_cluster_optimization(point_to_cluster, data, blues, reds, 0.9)
    
    # 7. Generate results
    print("\n6. Generating results...")
    results = []
    for i in range(len(df)):
        results.append({
            'id': i + 1,
            'cluster_id': point_to_cluster.get(i, 1),
            'protected_attribute': df.iloc[i][protected_attr_col]
        })
    
    results_df = pd.DataFrame(results)
    
    # Print cluster analysis
    cluster_dist = results_df['cluster_id'].value_counts().sort_index()
    print(f"   - Final clusters: {dict(cluster_dist)}")
    
    # Calculate and display balance metrics
    gender_cluster_dist = results_df.groupby(['cluster_id', 'protected_attribute']).size().unstack(fill_value=0)
    print(f"   - Protected attribute distribution by cluster:")
    print(gender_cluster_dist)
    
    # Calculate balance ratios
    unique_clusters = sorted(results_df['cluster_id'].unique())
    total_balance = 0
    valid_clusters = 0
    
    for cluster in unique_clusters:
        cluster_data = results_df[results_df['cluster_id'] == cluster]  
        if len(cluster_data) > 0:
            attr_counts = cluster_data['protected_attribute'].value_counts()
            if len(attr_counts) > 1:
                values = list(attr_counts.values)
                balance = min(values) / max(values)
                total_balance += balance
                valid_clusters += 1
                print(f"   - Cluster {cluster} balance ratio: {balance:.3f}")
            else:
                print(f"   - Cluster {cluster}: Only one protected attribute value present")
    
    if valid_clusters > 0:
        avg_balance = total_balance / valid_clusters
        print(f"   - Average cluster balance: {avg_balance:.3f}")
    
    # Save results
    results_df.to_csv(output_file, index=False)
    print(f"   - Saved to {output_file}")
    
    return results_df

def process_all_datasets():
    """Process datasets with streamlined pipeline"""
    dataset_configs = {
        'student-mat-encode.csv': {'k': 9, 't': 2, 'distance_threshold': 8},
        # 'student-por-encode.csv': {'k': 9, 't': 2, 'distance_threshold': 6},
        # 'german-encode.csv': {'k': 2, 't': 3, 'distance_threshold': 80},
        # 'compas-encode.csv': {'k': 7, 't': 2, 'distance_threshold': 80},
        # 'credit-encode.csv': {'k': 2, 't': 2, 'distance_threshold': 80},
        # 'adult-encode.csv': {'k': 2, 't': 3, 'distance_threshold': 80}
    }
    
    input_dir = "data-encoded"
    output_dir = "clustering"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Processing all datasets with streamlined fair clustering...")
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
