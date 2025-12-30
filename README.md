# FACROC: Fair Clustering through ROC Curves

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

### What is FACROC?

FACROC (FAir Clustering through ROC curves) quantifies fairness by calculating the area between ROC curves of different demographic groups. Unlike traditional fairness metrics that only consider group representation, FACROC evaluates whether clustering quality is equally distributed across protected groups. This repository implements the methodology from the paper "FACROC: a fairness measure for FAir Clustering through ROC curves" (PAKDD 2025).

**Key Innovation**: Combines fair clustering algorithms with ROC-based quality evaluation to ensure both:
- **Fairness**: Balanced representation of protected groups in clusters
- **Quality**: Comparable clustering performance across all demographic groups

### Key Features

- **FACROC Metric**: Novel fairness quantification by measuring area between ROC curves of protected and non-protected groups
- **AUCC (Area Under the Clustering Curve)**: ROC-based clustering quality evaluation - higher values indicate better cluster separation
- **Dual Fairlet Decomposition**: 
  - Scalable quadtree-based approach for large datasets (O(n log n))
  - Optimal MCF-based approach for smaller datasets
- **Quality-Aware Post-Processing**: Iterative reassignment to improve AUCC while maintaining (p,q)-balance constraints
- **Comprehensive Metrics Suite**: Balance ratio, proportionality, silhouette score, and AUCC
- **Production-Ready Implementations**: Complete pipelines for both clustering generation and fairness evaluation
- **6 Benchmark Datasets**: Real-world experiments on Adult, COMPAS, Credit, German, and Student datasets
- **Visualization Tools**: Automatic generation of ROC curve plots with fairness analysis

---

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/congduytran12/FACROC-Experiments
cd FACROC-Experiments

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

The following packages will be installed:
- **numpy** (<2.0.0) - Numerical computations
- **pandas** - Data manipulation and analysis
- **matplotlib** - Visualization and plotting
- **scipy** - Scientific computing (ROC curves, interpolation)
- **scikit-learn** - Machine learning utilities
- **scikit-learn-extra** - KMedoids clustering algorithm
- **networkx** - Graph-based algorithms for MCF fairlet decomposition

**Note**: NumPy version is restricted to <2.0.0 for compatibility with other dependencies.

---

## Quick Start

### Main Pipeline: Scalable Fair Clustering (Recommended)

The recommended approach uses quadtree-based fairlet decomposition for scalability:

**Step 1: Generate Fair Clustering Results**

Run the scalable fair clustering algorithm to generate clustering results for all datasets:

```bash
python scalable_fair_clustering.py
```

This script:
1. Loads encoded datasets from `data-encoded/`
2. Performs fairlet decomposition using quadtree hierarchical clustering
3. Runs K-medoids clustering on fairlet centers
4. Applies post-processing reassignment to optimize quality while maintaining fairness
5. Saves clustering results to `clustering/`

**Configuration:** The script uses predefined parameters for each dataset (k=number of clusters, p:q=balance ratio)

**Step 2: Evaluate Fairness with FACROC**

Run FACROC experiments to evaluate clustering fairness:

```bash
python facroc_experiments.py
```

This script:
1. Loads datasets and clustering results
2. Computes AUCC separately for protected and non-protected groups
3. Calculates FACROC fairness metric (area between ROC curves)
4. Generates ROC curve visualizations in `results/`
5. Reports comprehensive metrics (AUCC, balance, silhouette, proportionality)

**Example Output:**
```
Results for student_mat dataset:
  FACROC: 0.0847
  AUCC: 0.6087
  Balance: 0.7243
  Silhouette: 0.2341
  Proportionality: 0.8123
```

---

### Alternative: MCF-Based Fair Clustering

For comparison, you can also use the minimum-cost flow (MCF) based approach:

```bash
python fair_clustering_base.py
```

This alternative pipeline:
1. Uses MCF fairlet decomposition (optimal but less scalable)
2. Applies k-centers clustering on fairlet centers
3. Also performs quality-aware reassignment
4. Useful for smaller datasets or when optimal fairlet decomposition is needed

---

## Usage

### Quick Reference

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `scalable_fair_clustering.py` | Generate fair clustering (scalable) | Default choice for clustering |
| `fair_clustering_base.py` | Generate fair clustering (MCF-based) | Small datasets, need optimal fairlets |
| `facroc_experiments.py` | Evaluate fairness with FACROC | After generating clustering results |
| `threshold_analysis.py` | Analyze distance thresholds | Tuning MCF fairlet decomposition |

### 1. Calculate FACROC Metric

```python
from aucc import aucc
from facroc import compute_facroc

# Load your data and clustering results
data_protected = ...  # numeric features for protected group
data_non_protected = ...  # numeric features for non-protected group
cluster_ids_protected = ...  # cluster assignments for protected group
cluster_ids_non_protected = ...  # cluster assignments for non-protected group

# Compute AUCC for both groups (with ROC curve data)
aucc_protected = aucc(
    cluster_ids_protected, 
    dataset=data_protected, 
    return_rates=True
)
aucc_non_protected = aucc(
    cluster_ids_non_protected, 
    dataset=data_non_protected, 
    return_rates=True
)

# Calculate FACROC fairness metric
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

### 2. Run Custom FACROC Experiments

```python
from facroc_experiments import facroc_experiment

# Analyze custom dataset
results = facroc_experiment(
    dataset="data-encoded/your-dataset-encode.csv",
    clustering_result="clustering/your-dataset-clustering.csv", 
    figure_out="results/your-dataset.facroc.pdf",
    protected_attr="gender",
    protected_group="F",
    non_protected_group="M",
    protected_label="Female",
    non_protected_label="Male"
)

print(f"FACROC: {results['facroc']:.4f}")
print(f"AUCC: {results['aucc']:.4f}")
print(f"Balance: {results['balance']:.4f}")
print(f"Silhouette: {results['silhouette']:.4f}")
print(f"Proportionality: {results['proportionality']:.4f}")
```

---

## Datasets

The repository includes experiments on 11 real-world benchmark datasets:

| Dataset | Protected Attribute | Protected Values | Clusters (k) |
|---------|---------------------|------------------|--------------|
| Adult Census Income | Gender | Female/Male | 2 |
| Communities Crime | Black | 0/1 | 4 |
| COMPAS Recidivism | Race | White/Non-White | 7 |
| Credit Card Default | Gender | F/M | 2 |
| German Credit | Gender | F/M | 2 |
| OULAD (Open University) | Gender | F/M | 9 |
| PISA Education | Gender | F/M | 9 |
| Ricci Firefighter | Race | White/Non-White | 10 |
| Student Performance (Math) | Gender | F/M | 9 |
| Student Performance (Portuguese) | Gender | F/M | 9 |
| xAPI Educational Data | Gender | F/M | 11 |

### Data Pipeline

**1. Raw Data** (`data/`): Original datasets in CSV format with mixed data types

**2. Encoded Data** (`data-encoded/`): 
- Preprocessed and numerically encoded features
- Protected attribute column preserved
- Ready for clustering algorithms
- Format: CSV with numeric features + protected attribute

**3. Clustering Results** (`clustering/`):
- Generated by `scalable_fair_clustering.py`
- Format: CSV with columns `id`, `cluster_id`, `protected_attribute`
- Maintains (p,q)-fairness balance constraint

**4. Evaluation Results** (`results/`):
- ROC curve visualizations (PDF format)
- Generated by `facroc_experiments.py`
- Shows FACROC fairness metric

---

## Understanding Metrics

### FACROC Score (Fairness Metric)
FACROC measures the **area between the ROC curves** of protected and non-protected groups. It quantifies the disparity in clustering quality between groups.

- **0.0** - Perfect fairness (identical clustering quality for both groups)
- **< 0.1** - Fair clustering with minimal disparity
- **0.1-0.2** - Acceptable fairness
- **0.2-0.3** - Moderate fairness concerns
- **> 0.3** - Significant fairness issues requiring intervention

**Lower is better** - smaller FACROC indicates more equitable clustering.

### AUCC (Area Under the Clustering Curve)
AUCC evaluates **clustering quality** using ROC analysis. It measures how well points in the same cluster are closer than points in different clusters.

- **0.5** - Random clustering (baseline, no structure)
- **0.6-0.7** - Acceptable clustering quality
- **0.7-0.8** - Good clustering quality
- **0.8-0.9** - Excellent clustering quality
- **> 0.9** - Outstanding clustering 

**Higher is better** - larger AUCC indicates better cluster separation.

### Balance Ratio
Balance measures the **representation of protected groups within clusters**. Returns the smallest balance value among all clusters.

For each cluster: `Balance = min(protected_count) / max(protected_count)`

- **1.0** - Perfect balance (equal representation)
- **0.6-1.0** - Good balance
- **0.4-0.6** - Moderate balance
- **0.2-0.4** - Poor balance
- **< 0.2** - Severe imbalance

**Higher is better** - the algorithm enforces (p,q)-balance constraints where balance ≥ p/q.

### Silhouette Score
Silhouette measures **cluster cohesion and separation** - how well-defined clusters are.

- **0.7-1.0** - Excellent cluster structure
- **0.5-0.7** - Good cluster structure
- **0.25-0.5** - Reasonable structure (overlapping clusters)
- **0.0-0.25** - Weak structure
- **< 0.0** - Poor clustering (many misassignments)

**Higher is better** - indicates tight, well-separated clusters.

### Proportionality (ρ-Proportionality)
Proportionality measures **stability against strategic coalition deviations**. Higher values indicate that no significant group of points would benefit from forming their own cluster.

- **0.8-1.0** - Highly proportional (stable)
- **0.6-0.8** - Good proportionality
- **0.4-0.6** - Moderate proportionality
- **< 0.4** - Low proportionality (unstable clustering)

**Higher is better** - indicates fair allocation of cluster quality across all points.


---

## Project Structure

```
FACROC-Experiments/
├── Core Fairness Metrics
│   ├── aucc.py                      # AUCC metric computation with ROC analysis
│   ├── facroc.py                    # FACROC metric calculation & visualization
│   └── utils.py                     # Evaluation utilities (balance, silhouette, proportionality)
│
├── Fair Clustering Algorithms
│   ├── scalable_fair_clustering.py  # Main: Quadtree fairlet decomposition + K-medoids + reassignment
│   ├── fair_clustering_base.py      # Alternative: MCF-based fair clustering pipeline
│   ├── mcf_fairlet_decomposition.py # Min-cost flow fairlet decomposition
│   ├── hierarchical.py              # Hierarchical clustering implementation
│   ├── kcenters.py                  # K-centers clustering
│   └── kmeans.py                    # K-means clustering
│
├── Experiments & Analysis
│   ├── facroc_experiments.py        # Main: FACROC evaluation on all datasets
│   ├── threshold_analysis.py        # Distance threshold analysis for fairlet decomposition
│   └── data_loader.py               # Dataset loading and preprocessing utilities
│
├── Data Directories
│   ├── data/                        # Raw datasets (original CSV files)
│   ├── data-encoded/                # Preprocessed numeric datasets
│   │   └── readme.md               # Encoding documentation
│   ├── clustering/                  # Generated clustering results
│   └── results/                     # FACROC visualizations (PDF)
│
├── Configuration
│   ├── requirements.txt             # Python dependencies
│   └── README.md                    # This file
│
└── __pycache__/                     # Python bytecode cache
```

### Key Components

**Primary Workflow:**
- **`scalable_fair_clustering.py`**: Main scalable fair clustering implementation
  1. Builds hierarchical quadtree structure for spatial organization
  2. Performs (p,q)-fairlet decomposition bottom-up
  3. Clusters fairlet centers using K-medoids
  4. Post-processes with quality-aware reassignment to optimize AUCC

**Alternative Workflow:**
- **`fair_clustering_base.py`**: MCF-based fair clustering (optimal but slower)
  1. Uses minimum-cost flow for optimal fairlet decomposition
  2. Applies k-centers clustering on fairlet centers
  3. Also includes quality-aware reassignment step

**Additional Clustering Algorithms:**
- **`hierarchical.py`**: Hierarchical clustering with Ward linkage
- **`kmeans.py`**: Standard K-means implementation
- **`kcenters.py`**: K-centers clustering algorithm

**Evaluation Pipeline:**
- **`facroc_experiments.py`**: Complete FACROC evaluation workflow
  1. Loads encoded data and clustering results
  2. Splits by protected attribute
  3. Computes AUCC for each group with ROC curve data
  4. Calculates FACROC as area between curves
  5. Generates comprehensive metric reports and visualizations

---

## Algorithm Details

### Two Fairlet Decomposition Approaches

This repository implements two approaches for fair clustering:

#### 1. Scalable Quadtree-Based Approach (Primary)

**File**: `scalable_fair_clustering.py` + `tree_fairlet_decomposition.py`

The scalable fair clustering algorithm uses a hierarchical approach for large datasets:

1. **Quadtree Construction**: Build a space-partitioning tree to organize points hierarchically
2. **Bottom-up Fairlet Formation**: Create balanced micro-clusters (fairlets) satisfying (p,q)-balance constraint
3. **Fairlet Center Selection**: Choose representative points (medoids) for each fairlet
4. **K-medoids Clustering**: Cluster fairlet centers to obtain k macro-clusters
5. **Quality Reassignment**: Iteratively move points to closer clusters while maintaining balance

**Advantages**: Efficient for large datasets, hierarchical structure, O(n log n) complexity

#### 2. Minimum-Cost Flow (MCF) Based Approach (Alternative)

**File**: `fair_clustering_base.py` + `mcf_fairlet_decomposition.py`

The MCF-based approach provides optimal fairlet decomposition:

1. **Distance Computation**: Calculate pairwise distances between protected groups
2. **Flow Network Construction**: Build bipartite graph with capacity and cost constraints
3. **MCF Optimization**: Solve minimum-cost flow to find optimal fairlet assignment
4. **K-centers Clustering**: Apply k-centers algorithm on fairlet centers
5. **Quality Reassignment**: Post-process to improve AUCC while maintaining fairness

**Advantages**: Optimal fairlet decomposition, theoretically sound, better for smaller datasets

### Balance Constraint

Both approaches enforce the same balance constraint. Each fairlet and final cluster must satisfy:
```
min(protected_count, non_protected_count) / max(protected_count, non_protected_count) ≥ p/q
```

**Default configuration**: `p=2, q=5` → protected group representation must be ≥ 40% (2/5)

**Configurable in code**: 
- Scalable approach: `DATASET_CONFIGS` in `scalable_fair_clustering.py`
- MCF approach: `dataset_configs` in `fair_clustering_base.py`

### FACROC Computation

FACROC measures fairness by comparing ROC curves:

1. **Split Data**: Separate dataset by protected attribute
2. **Compute AUCC**: For each group, calculate AUCC (ROC analysis of clustering quality)
3. **Interpolate Curves**: Create smooth ROC curves with 40,000 grid points
4. **Calculate Area**: Integrate absolute difference between curves using trapezoidal rule
5. **Visualize**: Plot overlapping ROC curves with shaded area representing FACROC

---

## Citation

If you use this code in your research, please cite:

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

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## Contact

- **Repository**: [github.com/congduytran12/FACROC-Experiments](https://github.com/congduytran12/FACROC-Experiments)
- **Issues**: Please open an issue on GitHub for bugs or feature requests