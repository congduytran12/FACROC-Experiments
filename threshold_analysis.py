import numpy as np
import os
from scipy.spatial.distance import pdist
from data_loader import load_dataset
import matplotlib.pyplot as plt

def analyze_dataset_distances(input_file):
    """Analyze distance distribution for a dataset."""
    print(f"\nAnalyzing distances for: {os.path.basename(input_file)}")
    
    # load dataset
    data, blues, reds, df, protected_attr_col = load_dataset(input_file)
    data_array = np.array(data)
    
    # compute all pairwise distances
    distances = pdist(data_array, metric='euclidean')
    
    # compute distances between blue and red nodes
    blue_red_distances = []
    for b_idx in blues:
        for r_idx in reds:
            dist = np.linalg.norm(np.array(data[b_idx]) - np.array(data[r_idx]))
            blue_red_distances.append(dist)
    
    blue_red_distances = np.array(blue_red_distances)
    
    # calculate statistics
    stats = {
        'all_distances': {
            'mean': np.mean(distances),
            'std': np.std(distances),
            'median': np.median(distances),
            'percentiles': np.percentile(distances, [10, 25, 50, 75, 90, 95])
        },
        'blue_red_distances': {
            'mean': np.mean(blue_red_distances),
            'std': np.std(blue_red_distances),
            'median': np.median(blue_red_distances),
            'percentiles': np.percentile(blue_red_distances, [10, 25, 50, 75, 90, 95])
        }
    }
    
    return stats, distances, blue_red_distances

def suggest_distance_threshold(stats, conservative_factor=0.8):
    """Suggest distance threshold based on statistical analysis."""
    br_stats = stats['blue_red_distances']
    
    # Method 1: Based on percentiles 
    p75_threshold = br_stats['percentiles'][3] * conservative_factor  
    p50_threshold = br_stats['percentiles'][2] * conservative_factor 
    
    # Method 2: Based on mean + std
    mean_std_threshold = (br_stats['mean'] + 0.5 * br_stats['std']) * conservative_factor
    
    # Method 3: Conservative approach (25th percentile)
    conservative_threshold = br_stats['percentiles'][1] * conservative_factor
    
    suggestions = {
        'conservative': conservative_threshold,
        'median_based': p50_threshold,
        'balanced': p75_threshold,
        'mean_std_based': mean_std_threshold
    }
    
    return suggestions

def analyze_target_datasets():
    """Analyze specific target datasets and suggest thresholds."""
    input_dir = "data-encoded"
    target_datasets = ['adult-encode.csv']
    results = {}
    
    print("=" * 80)
    print("DISTANCE THRESHOLD ANALYSIS FOR TARGET DATASETS")
    print("=" * 80)
    
    for filename in target_datasets:
        input_file = os.path.join(input_dir, filename)
        
        if not os.path.exists(input_file):
            print(f"Warning: {filename} not found in {input_dir}")
            continue
            
        try:
            stats, all_dist, br_dist = analyze_dataset_distances(input_file)
            suggestions = suggest_distance_threshold(stats)
            
            results[filename] = {
                'stats': stats,
                'suggestions': suggestions,
                'all_distances': all_dist,
                'blue_red_distances': br_dist
            }

            print(f"\nDataset: {filename}")
            print(f"Blue-Red Distance Stats:")
            print(f"  Mean: {stats['blue_red_distances']['mean']:.2f}")
            print(f"  Median: {stats['blue_red_distances']['median']:.2f}")
            print(f"  Std: {stats['blue_red_distances']['std']:.2f}")
            print(f"  25th percentile: {stats['blue_red_distances']['percentiles'][1]:.2f}")
            print(f"  75th percentile: {stats['blue_red_distances']['percentiles'][3]:.2f}")
            print(f"Suggested thresholds:")
            for method, threshold in suggestions.items():
                print(f"  {method}: {threshold:.1f}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    return results

def generate_threshold_config(analysis_results, method='balanced'):
    """Generate configuration dictionary with suggested thresholds for target datasets."""
    config = {}
    
    print(f"\n" + "=" * 80)
    print(f"SUGGESTED CONFIGURATION (using {method} method)")
    print("=" * 80)
    
    for filename, result in analysis_results.items():
        threshold = result['suggestions'][method]

        if 'student-mat' in filename:
            k, t = 9, 2
        elif 'student-por' in filename:
            k, t = 9, 2
        elif 'german' in filename:
            k, t = 2, 3
        else:
            k, t = 2, 2  
            
        config[filename] = {
            'k': k,
            't': t,
            'distance_threshold': int(round(threshold))
        }
        
        print(f"'{filename}': {{'k': {k}, 't': {t}, 'distance_threshold': {int(round(threshold))}}},")
    
    return config

if __name__ == "__main__":
    results = analyze_target_datasets()
    
    if results:
        config_balanced = generate_threshold_config(results, 'balanced')
        config_conservative = generate_threshold_config(results, 'conservative')
        config_median = generate_threshold_config(results, 'median_based')
        config_mean_std = generate_threshold_config(results, 'mean_std_based')
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print("Use the 'balanced' configuration for general purposes.")
        print("Use the 'conservative' configuration for stricter fairlet formation.")
    else:
        print("No datasets were successfully analyzed. Please check that the files exist.")