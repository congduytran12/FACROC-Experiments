# FACROC: Fair Clustering through ROC Curves

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

FACROC is a fairness evaluation metric for clustering algorithms that measures fairness by comparing clustering quality between protected and non-protected groups using ROC curve analysis. This repository implements the methodology from the paper "FACROC: a fairness measure for FAir Clustering through ROC curves" (PAKDD 2025).

**Key Features:**
- **FACROC Metric**: Quantifies fairness by calculating the area between ROC curves of protected and non-protected groups
- **AUCC (Area Under the Clustering Curve)**: Evaluates clustering quality using ROC analysis
- **Scalable Fair Clustering**: Implements fairlet decomposition with quadtree-based hierarchical clustering
- **Cluster Quality Optimization**: Post-processing reassignment to improve AUCC while maintaining balance constraints
- **Multiple Fairness Metrics**: Balance ratio, proportionality, and silhouette scores
- **Real-world Datasets**: Experiments on 6 benchmark datasets (Adult, COMPAS, Credit, German, Student)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/congduytran12/FACROC-Experiments
cd FACROC-Experiments

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.7+
- numpy (<2.0.0)
- pandas
- matplotlib
- scipy
- scikit-learn
- scikit-learn-extra (for KMedoids clustering)
- networkx

---

## Quick Start

### Step 1: Generate Fair Clustering Results

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

### Step 2: Evaluate Fairness with FACROC

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

## Usage

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

The repository includes experiments on 6 real-world benchmark datasets:

| Dataset | Protected Attribute | Protected Values | Samples | Features | Clusters (k) |
|---------|---------------------|------------------|---------|----------|--------------|
| Student Performance (Math) | Gender | F/M | 395 | 32 | 9 |
| Student Performance (Portuguese) | Gender | F/M | 649 | 32 | 9 |
| German Credit | Gender | F/M | 1,000 | 20 | 2 |
| COMPAS Recidivism | Race | White/Non-White | 6,172 | 10 | 7 |
| Adult Census Income | Gender | Female/Male | 32,561 | 14 | 2 |
| Credit Card Default | Gender | F/M | 30,000 | 23 | 2 |

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
│   ├── scalable_fair_clustering.py  # Main: Fairlet decomposition + K-medoids + reassignment
│   ├── fairlet_decomposition.py     # Fairlet decomposition implementation
│   ├── fair_clustering_base.py      # MCF-based fairlet decomposition
│   ├── kcenters.py                  # Fair k-centers clustering
│   └── kmeans.py                    # Fair k-means clustering
│
├── Experiments & Analysis
│   ├── facroc_experiments.py        # Main: FACROC evaluation on all datasets
│   ├── threshold_analysis.py        # Distance threshold analysis
│   └── data_loader.py               # Dataset loading utilities
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

**Scalable Fair Clustering Pipeline** (`scalable_fair_clustering.py`):
1. Builds hierarchical quadtree structure
2. Performs (p,q)-fairlet decomposition bottom-up
3. Clusters fairlet centers using K-medoids
4. Post-processes with quality-aware reassignment

**FACROC Evaluation Pipeline** (`facroc_experiments.py`):
1. Loads encoded data and clustering results
2. Splits by protected attribute
3. Computes AUCC for each group
4. Calculates FACROC as area between curves
5. Generates comprehensive metric reports

---

## Algorithm Details

### Fairlet Decomposition with Quadtree

The scalable fair clustering algorithm uses a hierarchical approach:

1. **Quadtree Construction**: Build a space-partitioning tree to organize points
2. **Bottom-up Fairlet Formation**: Create balanced micro-clusters (fairlets) satisfying (p,q)-balance
3. **Fairlet Center Selection**: Choose representative points (medoids) for each fairlet
4. **K-medoids Clustering**: Cluster fairlet centers to obtain k macro-clusters
5. **Quality Reassignment**: Iteratively move points to closer clusters while maintaining balance

**Balance Constraint**: Each fairlet and final cluster must satisfy:
```
min(protected_count, non_protected_count) / max(protected_count, non_protected_count) ≥ p/q
```

Default configuration uses `p=2, q=5`, meaning protected group representation must be at least 40% (2/5).

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