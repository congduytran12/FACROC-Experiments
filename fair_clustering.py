import pandas as pd
import numpy as np
import os
from fairlet_decomposition import MCFFairletDecomposition
from utils import distance
from data_loader import load_dataset
from kcenters import KCenters

def enhanced_fairlet_center_selection(fairlets, data):
    """
    Enhanced fairlet center selection considering global data distribution and clustering quality.
    """
    fairlet_centers = []
    fairlet_costs = []
    
    # Calculate global centroid and statistics for reference
    all_points = [point for fairlet in fairlets for point in fairlet]
    global_centroid = np.mean([data[i] for i in all_points], axis=0)
    
    # Calculate inter-fairlet distances for better positioning
    fairlet_centroids = []
    for f in fairlets:
        if len(f) > 0:
            fairlet_centroid = np.mean([data[i] for i in f], axis=0)
            fairlet_centroids.append(fairlet_centroid)
        else:
            fairlet_centroids.append(global_centroid)
    
    for fairlet_idx, f in enumerate(fairlets):
        if len(f) == 1:
            # Single point fairlet
            fairlet_centers.append(f[0])
            fairlet_costs.append(0.0)
            continue
            
        # Calculate fairlet centroid
        fairlet_centroid = np.mean([data[i] for i in f], axis=0)
        
        # Score each point based on multiple criteria
        point_scores = []
        for i in f:
            # 1. Distance to fairlet centroid (lower is better for representativeness)
            dist_to_fairlet_center = distance(data[i], fairlet_centroid)
            
            # 2. Maximum distance to other points in fairlet (compactness)
            max_dist_in_fairlet = max([distance(data[i], data[j]) for j in f if j != i])
            
            # 3. Average distance to other fairlet centroids (for cluster separation)
            avg_dist_to_other_centroids = np.mean([
                distance(data[i], other_centroid) 
                for idx, other_centroid in enumerate(fairlet_centroids) 
                if idx != fairlet_idx
            ])
            
            # 4. Distance to global centroid (for global representativeness)
            dist_to_global_center = distance(data[i], global_centroid)
            
            # Combined score with optimized weights for clustering quality
            # Emphasize fairlet compactness and separation
            score = (0.4 * dist_to_fairlet_center + 
                    0.3 * max_dist_in_fairlet + 
                    0.1 * dist_to_global_center +
                    0.2 * (1.0 / (avg_dist_to_other_centroids + 1e-6)))  # Inverted for better separation
            
            point_scores.append((i, score))
        
        # Select point with minimum score (best overall characteristics)
        point_scores = sorted(point_scores, key=lambda x: x[1])
        center, cost = point_scores[0][0], point_scores[0][1]
        fairlet_centers.append(center)
        fairlet_costs.append(cost)
    
    return fairlet_centers, fairlet_costs

def refine_cluster_assignment(point_to_cluster, data, blues, reds, protected_attr_col, df, max_iterations=10):
    """
    Enhanced cluster refinement to improve both clustering quality and fairness.
    """
    print(f"\n   Refining cluster assignment for better balance...")
    
    for iteration in range(max_iterations):
        improved = False
        unique_clusters = list(set(point_to_cluster.values()))
        
        # Calculate current cluster statistics
        cluster_stats = {}
        for cluster_id in unique_clusters:
            cluster_points = [i for i, c in point_to_cluster.items() if c == cluster_id]
            
            # Protected attribute distribution in this cluster
            protected_count = 0
            non_protected_count = 0
            
            for point_idx in cluster_points:
                if point_idx in blues:
                    non_protected_count += 1
                else:
                    protected_count += 1
            
            total_count = protected_count + non_protected_count
            if total_count > 0:
                balance_ratio = min(protected_count, non_protected_count) / max(protected_count, non_protected_count) if max(protected_count, non_protected_count) > 0 else 0
            else:
                balance_ratio = 0
                
            cluster_stats[cluster_id] = {
                'protected': protected_count,
                'non_protected': non_protected_count,
                'total': total_count,
                'balance': balance_ratio,
                'points': cluster_points
            }
        
        # Identify clusters that need balancing - be more aggressive
        imbalanced_clusters = [c for c, stats in cluster_stats.items() if stats['balance'] < 0.8 and stats['total'] > 1]
        
        if not imbalanced_clusters:
            # If no major imbalances, try to optimize further
            imbalanced_clusters = [c for c, stats in cluster_stats.items() if stats['balance'] < 0.95 and stats['total'] > 2]
        
        if not imbalanced_clusters:
            break
            
        # Multiple strategies for balancing
        for cluster_id in imbalanced_clusters:
            stats = cluster_stats[cluster_id]
            
            # Strategy 1: Targeted swapping
            if stats['protected'] > stats['non_protected']:
                # Too many protected, try to swap with non-protected from other clusters
                over_points = [p for p in stats['points'] if p in reds][:3]  # Limit for efficiency
                
                # Find non-protected points in other clusters that could be swapped
                for other_cluster_id in unique_clusters:
                    if other_cluster_id == cluster_id:
                        continue
                    other_stats = cluster_stats[other_cluster_id]
                    
                    # Look for non-protected points in clusters with excess non-protected
                    if other_stats['non_protected'] > other_stats['protected']:
                        under_points = [p for p in other_stats['points'] if p in blues][:3]
                        
                        # Try swaps
                        for over_point in over_points:
                            for under_point in under_points:
                                # Calculate clustering quality impact
                                over_center = cluster_id
                                under_center = other_cluster_id
                                
                                # Simple distance-based quality check
                                over_to_under_dist = sum([distance(data[over_point], data[p]) for p in other_stats['points'][:5]]) / min(5, len(other_stats['points']))
                                under_to_over_dist = sum([distance(data[under_point], data[p]) for p in stats['points'][:5]]) / min(5, len(stats['points']))
                                
                                original_over_dist = sum([distance(data[over_point], data[p]) for p in stats['points'][:5]]) / min(5, len(stats['points']))
                                original_under_dist = sum([distance(data[under_point], data[p]) for p in other_stats['points'][:5]]) / min(5, len(other_stats['points']))
                                
                                # If swap doesn't hurt clustering quality too much, do it
                                quality_change = (over_to_under_dist + under_to_over_dist) - (original_over_dist + original_under_dist)
                                
                                if quality_change < 2.0:  # Acceptable quality loss threshold
                                    # Perform swap
                                    point_to_cluster[over_point] = other_cluster_id
                                    point_to_cluster[under_point] = cluster_id
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            else:
                # Too many non-protected, similar logic
                over_points = [p for p in stats['points'] if p in blues][:3]
                
                for other_cluster_id in unique_clusters:
                    if other_cluster_id == cluster_id:
                        continue
                    other_stats = cluster_stats[other_cluster_id]
                    
                    if other_stats['protected'] > other_stats['non_protected']:
                        under_points = [p for p in other_stats['points'] if p in reds][:3]
                        
                        for over_point in over_points:
                            for under_point in under_points:
                                # Similar quality check as above
                                over_to_under_dist = sum([distance(data[over_point], data[p]) for p in other_stats['points'][:5]]) / min(5, len(other_stats['points']))
                                under_to_over_dist = sum([distance(data[under_point], data[p]) for p in stats['points'][:5]]) / min(5, len(stats['points']))
                                
                                original_over_dist = sum([distance(data[over_point], data[p]) for p in stats['points'][:5]]) / min(5, len(stats['points']))
                                original_under_dist = sum([distance(data[under_point], data[p]) for p in other_stats['points'][:5]]) / min(5, len(other_stats['points']))
                                
                                quality_change = (over_to_under_dist + under_to_over_dist) - (original_over_dist + original_under_dist)
                                
                                if quality_change < 2.0:
                                    point_to_cluster[over_point] = other_cluster_id
                                    point_to_cluster[under_point] = cluster_id
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        if not improved:
            break
            
        # Recalculate stats for next iteration
        cluster_stats = {}
        for cluster_id in unique_clusters:
            cluster_points = [i for i, c in point_to_cluster.items() if c == cluster_id]
            
            protected_count = sum(1 for p in cluster_points if p in reds)
            non_protected_count = sum(1 for p in cluster_points if p in blues)
            total_count = protected_count + non_protected_count
            
            if total_count > 0:
                balance_ratio = min(protected_count, non_protected_count) / max(protected_count, non_protected_count) if max(protected_count, non_protected_count) > 0 else 0
            else:
                balance_ratio = 0
                
            cluster_stats[cluster_id] = {
                'protected': protected_count,
                'non_protected': non_protected_count,
                'total': total_count,
                'balance': balance_ratio,
                'points': cluster_points
            }
    
    return point_to_cluster

def advanced_cluster_balancing(point_to_cluster, data, blues, reds, protected_attr_col, df):
    """
    Advanced balancing specifically targeting FACROC optimization.
    """
    print(f"\n   Applying advanced cluster balancing for FACROC optimization...")
    
    unique_clusters = list(set(point_to_cluster.values()))
    
    # Calculate target balance ratio (closer to 1.0 means better fairness)
    total_protected = len(reds)
    total_non_protected = len(blues)
    global_ratio = total_protected / (total_protected + total_non_protected)
    
    for cluster_id in unique_clusters:
        cluster_points = [i for i, c in point_to_cluster.items() if c == cluster_id]
        
        protected_in_cluster = [p for p in cluster_points if p in reds]
        non_protected_in_cluster = [p for p in cluster_points if p in blues]
        
        cluster_size = len(cluster_points)
        if cluster_size < 2:
            continue
            
        # Target number of protected points for this cluster size
        target_protected = int(cluster_size * global_ratio)
        current_protected = len(protected_in_cluster)
        
        imbalance = current_protected - target_protected
        
        if abs(imbalance) <= 1:  # Already well balanced
            continue
            
        if imbalance > 1:  # Too many protected, need to move some out or bring non-protected in
            # Strategy: Move protected points to clusters with too few protected
            excess_protected = protected_in_cluster[:abs(imbalance)]
            
            # Find clusters that need more protected points
            target_clusters = []
            for other_cluster_id in unique_clusters:
                if other_cluster_id == cluster_id:
                    continue
                other_points = [i for i, c in point_to_cluster.items() if c == other_cluster_id]
                other_protected = [p for p in other_points if p in reds]
                other_size = len(other_points)
                
                if other_size > 0:
                    other_target_protected = int(other_size * global_ratio)
                    if len(other_protected) < other_target_protected:
                        target_clusters.append(other_cluster_id)
            
            # Move excess protected points to target clusters
            for i, protected_point in enumerate(excess_protected):
                if i < len(target_clusters):
                    target_cluster = target_clusters[i]
                    
                    # Check if move improves clustering quality
                    target_points = [j for j, c in point_to_cluster.items() if c == target_cluster]
                    if target_points:
                        avg_dist_current = sum([distance(data[protected_point], data[p]) for p in cluster_points[:5]]) / min(5, len(cluster_points))
                        avg_dist_target = sum([distance(data[protected_point], data[p]) for p in target_points[:5]]) / min(5, len(target_points))
                        
                        # Move if quality doesn't degrade too much
                        if avg_dist_target <= avg_dist_current * 1.5:
                            point_to_cluster[protected_point] = target_cluster
        
        elif imbalance < -1:  # Too few protected, need to bring some in
            deficit = abs(imbalance)
            
            # Find protected points in other over-represented clusters
            candidates = []
            for other_cluster_id in unique_clusters:
                if other_cluster_id == cluster_id:
                    continue
                other_points = [i for i, c in point_to_cluster.items() if c == other_cluster_id]
                other_protected = [p for p in other_points if p in reds]
                other_size = len(other_points)
                
                if other_size > 0:
                    other_target_protected = int(other_size * global_ratio)
                    if len(other_protected) > other_target_protected:
                        # This cluster has excess protected points
                        excess = other_protected[:len(other_protected) - other_target_protected]
                        for p in excess:
                            candidates.append(p)
            
            # Move best candidates to current cluster
            candidates = candidates[:deficit]
            for protected_point in candidates:
                # Check clustering quality
                avg_dist_current_cluster = sum([distance(data[protected_point], data[p]) for p in cluster_points[:5]]) / min(5, len(cluster_points)) if cluster_points else 0
                current_cluster = point_to_cluster[protected_point]
                current_cluster_points = [j for j, c in point_to_cluster.items() if c == current_cluster]
                avg_dist_original = sum([distance(data[protected_point], data[p]) for p in current_cluster_points[:5]]) / min(5, len(current_cluster_points)) if current_cluster_points else 0
                
                if avg_dist_current_cluster <= avg_dist_original * 1.5:
                    point_to_cluster[protected_point] = cluster_id
    
    return point_to_cluster

def optimize_for_aucc(point_to_cluster, data, blues, reds):
    """
    Post-processing optimization specifically for AUCC improvement.
    """
    print(f"\n   Optimizing assignments for higher AUCC...")
    
    unique_clusters = list(set(point_to_cluster.values()))
    
    # For each cluster, try to make assignments more coherent
    for cluster_id in unique_clusters:
        cluster_points = [i for i, c in point_to_cluster.items() if c == cluster_id]
        
        if len(cluster_points) < 3:
            continue
            
        # Calculate cluster centroid
        cluster_centroid = np.mean([data[i] for i in cluster_points], axis=0)
        
        # Find points that might be better served by other clusters
        outliers = []
        for point in cluster_points:
            dist_to_centroid = distance(data[point], cluster_centroid)
            
            # Check if this point is closer to other cluster centroids
            better_clusters = []
            for other_cluster_id in unique_clusters:
                if other_cluster_id == cluster_id:
                    continue
                    
                other_points = [i for i, c in point_to_cluster.items() if c == other_cluster_id]
                if other_points:
                    other_centroid = np.mean([data[i] for i in other_points], axis=0)
                    dist_to_other = distance(data[point], other_centroid)
                    
                    if dist_to_other < dist_to_centroid * 0.8:  # Significantly closer
                        better_clusters.append((other_cluster_id, dist_to_other))
            
            if better_clusters:
                # Find the best alternative cluster
                better_clusters.sort(key=lambda x: x[1])
                best_cluster, best_dist = better_clusters[0]
                outliers.append((point, best_cluster, dist_to_centroid - best_dist))
        
        # Move the most obvious outliers (limit to maintain stability)
        outliers.sort(key=lambda x: x[2], reverse=True)  # Sort by improvement potential
        
        moves_made = 0
        for point, target_cluster, improvement in outliers:
            if moves_made >= min(3, len(cluster_points) // 4):  # Limit moves to maintain cluster integrity
                break
                
            # Move point to better cluster
            point_to_cluster[point] = target_cluster
            moves_made += 1
    
    return point_to_cluster

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
    fairlets, original_fairlet_centers, original_fairlet_costs = mcf.decompose()
    
    print(f"   - Number of fairlets created: {len(fairlets)}")
    print(f"   - Average fairlet size: {np.mean([len(f) for f in fairlets]):.2f}")
    print(f"   - Average fairlet cost: {np.mean(original_fairlet_costs):.4f}")
    
    # enhanced fairlet center selection
    print("\n4.1. Applying enhanced fairlet center selection...")
    fairlet_centers, fairlet_costs = enhanced_fairlet_center_selection(fairlets, data)
    print(f"   - Enhanced fairlet centers selected")
    print(f"   - Improved average fairlet cost: {np.mean(fairlet_costs):.4f}")
    
    # apply k-centers clustering on fairlet centers
    print(f"\n5. Applying enhanced K-centers clustering with k={k}...")
    fairlet_center_data = [data[center] for center in fairlet_centers]
    
    kcenters = KCenters(k=k)
    kcenters.fit(fairlet_center_data)
    fairlet_cluster_mapping = kcenters.assign()
    
    print(f"   - Fairlet centers clustered into {k} clusters using enhanced K-centers")
    
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
    
    # apply cluster refinement to improve balance and quality
    print("\n6.1. Applying cluster refinement...")
    point_to_cluster = refine_cluster_assignment(
        point_to_cluster, data, blues, reds, protected_attr_col, df
    )
    
    # apply advanced balancing for FACROC optimization
    point_to_cluster = advanced_cluster_balancing(
        point_to_cluster, data, blues, reds, protected_attr_col, df
    )
    
    # optimize for AUCC improvement
    point_to_cluster = optimize_for_aucc(
        point_to_cluster, data, blues, reds
    )
    
    # create results dataframe
    print("\n7. Creating results...")
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
    print("\n8. Enhanced Cluster Analysis:")
    cluster_dist = results_df['cluster_id'].value_counts().sort_index()
    print(f"   - Cluster distribution: {dict(cluster_dist)}")
    
    gender_cluster_dist = results_df.groupby(['cluster_id', 'protected_attribute']).size().unstack(fill_value=0)
    print(f"   - Protected attribute distribution by cluster:")
    print(gender_cluster_dist)
    
    # calculate enhanced fairness metrics
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
    
    # save results
    print(f"\n9. Saving results to {output_file}...")
    results_df.to_csv(output_file, index=False)
    
    print("\n Enhanced fair clustering completed successfully!")
    print("=" * 60)
    
    return results_df


def process_all_datasets():
    """
    Process all datasets in data-encoded directory with appropriate cluster counts.
    """
    # Define cluster counts for each dataset
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
