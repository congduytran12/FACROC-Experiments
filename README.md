# FACROC: Fairness measure for Fair Clustering through ROC curves (This branch is for testing new optimization method)

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This repository provides a comprehensive implementation of the **FACROC** (Fair Clustering through ROC curves) metric, a novel fairness evaluation measure for clustering algorithms. FACROC quantifies clustering fairness by comparing the clustering quality between protected and non-protected groups using ROC curve analysis and the Area Under the Clustering Curve (AUCC).

The implementation includes:
- 🎯 **FACROC metric calculation** with ROC curve visualization
- 🧮 **AUCC computation** for clustering quality assessment  
- ⚖️ **Fair clustering algorithms** using MCF fairlet decomposition
- 📊 **Comprehensive experiments** on real-world datasets
- 📈 **Visualization tools** for fairness analysis

---

## Repository Structure

```
FACROC-Experiments/
├── Core Implementation
│   ├── aucc.py                    # AUCC metric computation and ROC curve generation
│   ├── facroc.py                  # FACROC metric calculation and visualization
│   ├── facroc_experiments.py     # Experiment scripts for multiple datasets
│   └── utils.py                   # Utility functions
│
├── Fair Clustering
│   ├── fair_clustering.py         # Fair clustering using MCF fairlet decomposition
│   ├── fairlet_decomposition.py   # MCF fairlet decomposition implementation
│   ├── kcenters.py               # K-centers clustering algorithm
│   └── data_loader.py            # Dataset loading and preprocessing utilities
│
├── Data
│   ├── data/                     # Raw datasets (adult, compas, german, student, credit)
│   ├── data-encoded/             # Preprocessed/encoded datasets for clustering
│   ├── clustering/               # Clustering results with fairness annotations
│   └── results/                  # Generated plots and FACROC visualizations
│
└── Configuration
    ├── requirements.txt          # Python dependencies
    └── README.md                # This file
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

Run the main experiment script to compute FACROC metrics for available datasets:

```bash
python facroc_experiments.py
```

The script will:
1. Load preprocessed datasets and clustering results
2. Compute AUCC for protected and non-protected groups
3. Calculate FACROC fairness metric
4. Generate ROC curve visualizations
5. Save results to the `results/` directory

### Example Output
```
Starting FACROC experiments...
Loading datasets from data-encoded/student-mat-encode.csv and clustering/student-mat-clustering.csv
Data shape: (395, 58), Clustering shape: (395, 3)
Protected group AUCC: 0.8072
Non-protected group AUCC: 0.9049
FACROC value for student_mat dataset: 0.09774463022196837
```

---

## Detailed Usage

### 1. FACROC Metric Calculation

```python
from aucc import aucc
from facroc import compute_facroc

# Compute AUCC for both groups
aucc_protected = aucc(cluster_ids_protected, dataset=data_protected, return_rates=True)
aucc_non_protected = aucc(cluster_ids_non_protected, dataset=data_non_protected, return_rates=True)

# Calculate FACROC
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

### 2. Fair Clustering with MCF Fairlet Decomposition

```python
from fair_clustering import fair_clustering_dataset

# Generate fair clustering results
results = fair_clustering_dataset(
    input_file="data-encoded/dataset-encode.csv",
    output_file="clustering/dataset-clustering.csv",
    k=3,           # Number of clusters
    t=2,           # Fairness parameter (1:t ratio)
    distance_threshold=50  # Distance threshold for fairlets
)
```

### 3. Custom Dataset Analysis

```python
from facroc_experiments import facroc_experiment

# Analyze custom dataset
facroc_value = facroc_experiment(
    dataset="data-encoded/your-dataset-encode.csv",
    clustering_result="clustering/your-dataset-clustering.csv", 
    figure_out="results/your-dataset.facroc.pdf",
    protected_attr="protected_column",
    protected_group="minority_value",
    non_protected_group="majority_value",
    protected_label="Minority Group",
    non_protected_label="Majority Group"
)
```

---

## Available Datasets

The repository includes experiments on several real-world datasets:

| Dataset | Description | Protected Attribute | Size |
|---------|-------------|---------------------|------|
| **Student Performance** | UCI student performance data (Math & Portuguese) | Gender (M/F) | 395/649 samples |
| **German Credit** | Credit approval dataset | Gender (M/F) | 1000 samples |
| **COMPAS** | Criminal justice risk assessment | Race (White/Non-White) | 4020 samples |
| **Adult Census** | Income prediction dataset | Gender (Male/Female) | 45222 samples |
| **Credit Card** | Default payment prediction | Gender (M/F) | 30000 samples |

### Dataset Format Requirements

**Input datasets** (`data-encoded/`):
- CSV format with numeric features
- Protected attribute column (e.g., 'gender', 'race')
- All features preprocessed/encoded for clustering

**Clustering results** (`clustering/`):
- CSV format with columns: `id`, `cluster_id`, `protected_attribute`
- Must match the input dataset size
- Protected attribute values must correspond to input dataset

---

## Core Components

### AUCC (Area Under the Clustering Curve)
- **Purpose**: Measures clustering quality for a specific group
- **Method**: Uses pairwise similarities and cluster assignments to generate ROC curves
- **Output**: AUCC score (0-1, higher = better clustering) + TPR/FPR rates

### FACROC (Fairness Metric)
- **Purpose**: Quantifies fairness by comparing clustering quality between groups
- **Method**: Calculates area between ROC curves of protected vs non-protected groups
- **Output**: FACROC score (0-1, lower = more fair) + visualization

### MCF Fairlet Decomposition
- **Purpose**: Creates fair clustering solutions
- **Method**: Decomposes data into fairlets (small fair groups) then clusters fairlet centers
- **Parameters**: 
  - `k`: Number of final clusters
  - `t`: Fairness ratio (1:t protected:non-protected)
  - `distance_threshold`: Maximum distance for fairlet formation

---

## Configuration and Customization

### Experiment Configuration

Edit `facroc_experiments.py` to:
- Select different datasets (uncomment desired experiments)
- Modify protected group definitions
- Adjust visualization parameters
- Change output file locations

### Fair Clustering Parameters

Modify `fair_clustering.py` to adjust:
- Cluster counts (`k`) for different datasets
- Fairness ratios (`t`) for desired balance
- Distance thresholds for fairlet formation
- Distance metrics (Euclidean, Manhattan, etc.)

---

## Interpreting Results

### FACROC Values
- **0.0**: Perfect fairness (identical clustering quality for both groups)
- **0.5**: Maximum unfairness (completely different clustering quality)
- **< 0.1**: Generally considered fair
- **> 0.3**: Significant fairness concerns

### AUCC Values  
- **0.5**: Random clustering (baseline)
- **> 0.7**: Good clustering quality
- **> 0.8**: Excellent clustering quality
- **1.0**: Perfect clustering (theoretical maximum)

### Visualization
- **ROC Curves**: Show clustering performance for each group
- **Gray Area**: Represents FACROC value (smaller = more fair)
- **Curve Separation**: Indicates disparity in clustering quality

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
