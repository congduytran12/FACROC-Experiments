# FACROC: Fairness measure for Fair Clustering through ROC curves

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This repository provides a comprehensive implementation of the **FACROC** (Fair Clustering through ROC curves) metric, a novel fairness evaluation measure for clustering algorithms. FACROC quantifies clustering fairness by comparing the clustering quality between protected and non-protected groups using ROC curve analysis and the Area Under the Clustering Curve (AUCC).

The implementation includes:
- 🎯 **FACROC metric calculation** with ROC curve visualization and area-based fairness measurement
- 🧮 **AUCC computation** for clustering quality assessment with improved interpolation
- ⚖️ **Multiple fair clustering algorithms** including base and optimized MCF fairlet decomposition
- 📊 **Comprehensive experiments** on real-world datasets with balance and silhouette metrics
- 📈 **Advanced visualization tools** for fairness analysis with publication-ready plots
- ⚡ **Performance optimization** with unified clustering optimization and outlier reassignment
- 🔧 **Enhanced evaluation metrics** including balance ratios and clustering quality measures

---

## Repository Structure

```
FACROC-Experiments/
├── Core Implementation
│   ├── aucc.py                      # AUCC metric computation and ROC curve generation
│   ├── facroc.py                    # FACROC metric calculation with enhanced visualization
│   ├── facroc_experiments.py       # Comprehensive experiment scripts with multiple metrics
│   └── utils.py                     # Utility functions (distance, balance, silhouette)
│
├── Fair Clustering Algorithms
│   ├── fair_clustering_base.py      # Basic MCF fairlet decomposition clustering
│   ├── fair_clustering_new.py      # Advanced fair clustering with optimization
│   ├── fairlet_decomposition.py    # MCF fairlet decomposition implementation
│   ├── kcenters.py                 # K-centers clustering algorithm
│   └── data_loader.py              # Dataset loading and preprocessing utilities
│
├── Data
│   ├── data/                       # Raw datasets (adult, compas, german, student, credit)
│   ├── data-encoded/              # Preprocessed/encoded datasets for clustering
│   ├── clustering/                # Clustering results with fairness annotations
│   └── results/                   # Generated plots and FACROC visualizations (PDF)
│
└── Configuration
    ├── requirements.txt            # Python dependencies
    └── README.md                  # This documentation
```

---

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/congduytran12/FACROC-Experiments
   cd FACROC-Experiments
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies
- `numpy` - Numerical computing
- `pandas` - Data manipulation and analysis  
- `matplotlib` - Plotting and visualization
- `scipy` - Scientific computing and interpolation
- `scikit-learn` - Machine learning utilities and metrics
- `networkx` - Graph algorithms for fairlet decomposition
- `numba` - Performance optimization

---

## Quick Start

### Running FACROC Experiments

Run the main experiment script to compute FACROC and additional metrics for available datasets:

```bash
python facroc_experiments.py
```

The script will:
1. Load preprocessed datasets and clustering results
2. Compute AUCC for protected and non-protected groups with improved interpolation
3. Calculate FACROC fairness metric using area-based measurement
4. Compute balance ratios and silhouette scores for comprehensive evaluation
5. Generate high-quality ROC curve visualizations
6. Save results to the `results/` directory in PDF format

### Example Output
```
Starting FACROC experiments...
Loading datasets from data-encoded/student-mat-encode.csv and clustering/student-mat-clustering.csv
Running AUCC for protected group with 208 samples...
Protected group AUCC: 0.6234
Running AUCC for non-protected group with 187 samples...
Non-protected group AUCC: 0.5891
Smallest cluster balance: 0.7243
Overall AUCC: 0.6087
Silhouette score: 0.2341

Results for student_mat dataset:
  FACROC: 0.0847
  AUCC: 0.6087
  Balance: 0.7243
  Silhouette: 0.2341
```

---

## Detailed Usage

### 1. FACROC Metric Calculation

```python
from aucc import aucc
from facroc import compute_facroc

# Compute AUCC for both groups (enhanced with interpolation)
aucc_protected = aucc(cluster_ids_protected, dataset=data_protected, return_rates=True)
aucc_non_protected = aucc(cluster_ids_non_protected, dataset=data_non_protected, return_rates=True)

# Calculate FACROC with improved area-based measurement
facroc_value = compute_facroc(
    auccResult_protected=aucc_protected,
    auccResult_non_protected=aucc_non_protected,
    protected_attribute="gender",
    protected="Female",
    non_protected="Male",
    showPlot=True,
    filename="results/fairness_analysis.pdf"
)
```

### 2. Fair Clustering with Base MCF Implementation

```python
from fair_clustering_base import fair_clustering_dataset

# Generate fair clustering results using basic approach
results = fair_clustering_dataset(
    input_file="data-encoded/dataset-encode.csv",
    output_file="clustering/dataset-clustering.csv",
    k=3,                    # Number of clusters
    t=2,                    # Fairness parameter (1:t ratio)
    distance_threshold=50   # Distance threshold for fairlet formation
)
```

### 3. Scalable Fair Clustering with HST Embedding (NEW!)

```python
from scalable_fair_clustering import scalable_fair_clustering_dataset

# Generate fair clustering results using scalable approach
results = scalable_fair_clustering_dataset(
    input_file="data-encoded/dataset-encode.csv",
    output_file="clustering-scalable/dataset-clustering.csv",
    k=3,                    # Number of clusters
    t=2,                    # Fairness parameter (1:t ratio)
                           # No distance threshold needed!
)
```

### 3. Advanced Fair Clustering with Optimization

```python
from fair_clustering_new import fair_clustering_dataset

# Generate optimized fair clustering with unified optimization
results = fair_clustering_dataset(
    input_file="data-encoded/dataset-encode.csv",
    output_file="clustering/dataset-clustering.csv",
    k=3,                    # Number of clusters
    t=2,                    # Fairness parameter (1:t ratio)
    distance_threshold=50   # Distance threshold for fairlet formation
)

# The optimized version includes:
# - Unified cluster optimization for improved quality
# - Outlier reassignment for better clustering
# - Small cluster merging for stability
```

### 4. Comprehensive Experimental Analysis

```python
from facroc_experiments import facroc_experiment

# Analyze custom dataset with full metrics
results = facroc_experiment(
    dataset="data-encoded/your-dataset-encode.csv",
    clustering_result="clustering/your-dataset-clustering.csv", 
    figure_out="results/your-dataset.facroc.pdf",
    protected_attr="protected_column",
    protected_group="minority_value",
    non_protected_group="majority_value",
    protected_label="Minority Group",
    non_protected_label="Majority Group"
)

# Results include:
# - FACROC score
# - Overall AUCC
# - Balance ratio (smallest cluster balance)
# - Silhouette score
```

### 5. Computing Additional Metrics

```python
from utils import calculate_balance, calculate_silhouette_score
import pandas as pd

# Load clustering results
clustering_df = pd.read_csv("clustering/dataset-clustering.csv")

# Calculate balance ratio (fairness measure)
balance = calculate_balance(clustering_df, 'protected_attribute')
print(f"Cluster balance: {balance:.4f}")

# Calculate clustering quality
data_array = your_numeric_data.values
cluster_ids = clustering_df['cluster_id'].values
silhouette = calculate_silhouette_score(data_array, cluster_ids)
print(f"Silhouette score: {silhouette:.4f}")
```

---

## Available Datasets

The repository includes experiments on several real-world datasets:

| Dataset | Description | Protected Attribute | Size |
|---------|-------------|---------------------|------|
| **Student Performance** | UCI student performance data (Math & Portuguese) | Gender (M/F) | 395/649 samples |
| **German Credit** | Credit approval dataset | Gender (M/F) | 1000 samples |
| **COMPAS** | Criminal justice risk assessment | Race (White/Non-White) | 6172 samples |
| **Adult Census** | Income prediction dataset | Gender (Male/Female) | 32561 samples |
| **Credit Card** | Default payment prediction | Gender (1/2) | 30000 samples |

### Dataset Format Requirements

**Input datasets** (`data-encoded/`):
- CSV format with numeric features (preprocessed and encoded)
- Protected attribute column (e.g., 'gender', 'sex', 'race')
- All features must be numeric for distance calculations
- Missing values should be handled before encoding

**Clustering results** (`clustering/`):
- CSV format with columns: `id`, `cluster_id`, `protected_attribute`
- Must match the input dataset size exactly
- Protected attribute values must correspond to input dataset
- Cluster IDs should be consecutive integers starting from 1

**Processing Pipeline**:
1. Load raw data from `data/` directory
2. Preprocess and encode features → save to `data-encoded/`
3. Apply fair clustering algorithms → save results to `clustering/`
4. Run FACROC experiments → generate visualizations in `results/`

---

## Choosing Fair Clustering Implementation

### When to use `fair_clustering_base.py`:
- ✅ Simple, straightforward MCF fairlet decomposition
- ✅ Faster execution for quick prototyping
- ✅ Baseline results for comparison
- ✅ Datasets where basic fairlet decomposition works well

### When to use `fair_clustering_new.py`:
- ✅ Enhanced clustering quality needed
- ✅ Datasets with outliers or noise
- ✅ Better balance between fairness and clustering quality
- ✅ Production environments requiring optimized results
- ✅ Small clusters that need merging for stability

### When to use `scalable_fair_clustering.py` (NEW!):
- ✅ Large datasets (>1000 points)
- ✅ Performance-critical applications
- ✅ No parameter tuning (distance thresholds)
- ✅ Best theoretical time complexity O(n log n)
- ✅ K-median clustering for better quality
- ✅ Scalable to very large datasets (30K+ points)

### Performance Comparison:
| Aspect | Base Implementation | Advanced Implementation | Scalable Implementation |
|--------|-------------------|------------------------|------------------------|
| **Speed** | Moderate | Slower (optimization) | **Fastest** |
| **Scalability** | Poor (O(n²)) | Poor (O(n²)) | **Excellent (O(n log n))** |
| **Parameters** | Distance threshold | Distance threshold | **None required** |
| **Quality** | Standard | Enhanced | **K-median clustering** |
| **Fairness** | Good | Better | **Maintained** |
| **Large Datasets** | Impractical | Impractical | **Designed for scale** |
| **Speed** | Faster | Slower (due to optimization) | **Fastest** |
| **Quality** | Standard | Enhanced with optimization | **Enhanced with optimization** |
| **Fairness** | Good | Better (unified optimization) | **Better (unified optimization)** |
| **Stability** | Basic | Improved (cluster merging) | **Improved (cluster merging)** |
| **Outlier Handling** | None | Intelligent reassignment | **Intelligent reassignment** |

### Scalability Performance Results:
| Dataset | Points | Original MCF | Scalable | Speedup |
|---------|--------|--------------|----------|---------|
| Student Mat | 395 | ~10s | 0.47s | **21x faster** |
| Student Por | 649 | ~45s | 1.0s | **45x faster** |
| German Credit | 1000 | >120s | 2.4s | **50x+ faster** |
| COMPAS | 4020 | >300s | 10.9s | **30x+ faster** |

*Processing rate: 367-839 points/second with scalable implementation*

### Recommended Usage:
1. Start with `fair_clustering_base.py` for initial experiments
2. Switch to `fair_clustering_new.py` if you need better quality or have problematic datasets
3. Compare results between both implementations using the same parameters

---

## Core Components

### AUCC (Area Under the Clustering Curve)
- **Purpose**: Measures clustering quality for a specific group using ROC curve analysis
- **Method**: Uses pairwise similarities and cluster assignments to generate ROC curves with enhanced interpolation
- **Improvements**: Linear interpolation with monotonicity constraints and improved smoothing
- **Output**: AUCC score (0-1, higher = better clustering) + detailed TPR/FPR rates for visualization

### FACROC (Fairness Metric)
- **Purpose**: Quantifies fairness by comparing clustering quality between protected and non-protected groups
- **Method**: Calculates area between ROC curves using trapezoidal integration for precise measurement
- **Visualization**: High-quality anti-aliased plots with publication-ready PDF output
- **Output**: FACROC score (0-1, lower = more fair) + comprehensive ROC curve comparison plots

### MCF Fairlet Decomposition

- **Purpose**: Creates fair clustering solutions by ensuring balanced representation
- **Method**: Decomposes data into fairlets (small fair groups) then clusters fairlet centers
- **Implementations**: 
  - **Base version** (`fair_clustering_base.py`): Standard MCF approach with basic clustering
  - **Advanced version** (`fair_clustering_new.py`): Enhanced with unified optimization and outlier reassignment
  - **Scalable version** (`scalable_fair_clustering.py`): HST embedding with O(n log n) complexity
- **Parameters**: 
  - `k`: Number of final clusters
  - `t`: Fairness ratio (1:t protected:non-protected)
  - `distance_threshold`: Maximum distance for fairlet formation (not needed for scalable version)

### Enhanced Metrics
- **Balance Ratio**: Measures fairness within clusters as min(protected)/max(protected)
- **Silhouette Score**: Evaluates overall clustering quality and cohesion
- **Unified Optimization**: Advanced algorithm for improving clustering quality while maintaining fairness
- **Outlier Reassignment**: Intelligent point reassignment for better cluster cohesion

---

## Configuration and Customization

### Experiment Configuration

Edit `facroc_experiments.py` to:
- Select different datasets (uncomment desired experiments in the main block)
- Modify protected group definitions and attribute names
- Adjust visualization parameters and output locations
- Control which metrics to compute (FACROC, AUCC, balance, silhouette)

### Fair Clustering Parameters

**Base Implementation** (`fair_clustering_base.py`):
- Standard MCF fairlet decomposition
- Basic k-centers clustering on fairlet centers
- Straightforward cluster assignment

**Advanced Implementation** (`fair_clustering_new.py`):
- Enhanced MCF with unified cluster optimization
- Outlier detection and reassignment (threshold: 1.2x average distance)
- Small cluster merging (clusters < 3 points merged with nearest large cluster)
- Fairness-aware point reassignment with balance constraints

**Configuration Parameters**:
```python
dataset_configs = {
    'student-mat-encode.csv': {
        'k': 9,                    # Number of clusters
        't': 2,                    # Fairness ratio (1:2)
        'distance_threshold': 8    # Distance threshold for fairlets
    }
}
```

### Distance Metrics and Thresholds
- **Distance Function**: Euclidean norm (L2) by default, customizable in `utils.py`
- **Fairlet Formation**: Points within `distance_threshold` can form fairlets
- **Optimization Thresholds**: 
  - Outlier detection: 1.2x average cluster distance
  - Balance constraint: minimum 0.6 balance ratio for reassignment
  - Small cluster threshold: < 3 points triggers merging

---

## Interpreting Results

### FACROC Values
- **0.0**: Perfect fairness (identical clustering quality for both groups)
- **0.5**: Maximum unfairness (completely different clustering quality)
- **< 0.1**: Generally considered fair clustering
- **0.1-0.3**: Moderate fairness concerns, may require adjustment
- **> 0.3**: Significant fairness issues requiring intervention

### AUCC Values  
- **0.5**: Random clustering (baseline performance)
- **0.5-0.6**: Poor clustering quality
- **0.6-0.7**: Acceptable clustering quality
- **0.7-0.8**: Good clustering quality
- **> 0.8**: Excellent clustering quality
- **1.0**: Perfect clustering (theoretical maximum, rarely achieved)

### Balance Ratio
- **1.0**: Perfect balance (equal representation of protected attributes)
- **0.8-1.0**: Good balance, minor representation differences
- **0.6-0.8**: Moderate balance, noticeable but acceptable differences
- **0.4-0.6**: Poor balance, significant representation disparity
- **< 0.4**: Very poor balance, major fairness concerns

### Silhouette Score
- **0.7-1.0**: Excellent clustering structure
- **0.5-0.7**: Good clustering structure
- **0.25-0.5**: Reasonable clustering structure
- **0.0-0.25**: Weak clustering structure
- **< 0.0**: Poor clustering (points closer to other clusters than their own)

### Visualization Interpretation
- **ROC Curves**: Show clustering performance for each group
- **Gray Shaded Area**: Represents FACROC value (smaller area = more fair)
- **Curve Separation**: Larger separation indicates greater disparity in clustering quality
- **Plot Quality**: Publication-ready PDFs with anti-aliasing and proper font embedding

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

---

## Citation

If you use this implementation in your research, please cite:

```bibtex
@inproceedings{le2025facroc,
  title={FACROC: a fairness measure for FAir Clustering through ROC curves},
  author={Le Quy, Tai and Le Thanh, Long and Luong Thi Hong, Lan and Hopfgartner, Frank},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  pages={340--352},
  year={2025},
  organization={Springer}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, issues, or contributions, please:
- 📧 Open an issue on GitHub
- 🐛 Submit bug reports with reproducible examples
- 💡 Suggest enhancements or new features

**Repository**: [https://github.com/congduytran12/FACROC-Experiments](https://github.com/congduytran12/FACROC-Experiments)
