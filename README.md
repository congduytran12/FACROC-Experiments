# FACROC: Fair Clustering through ROC Curves

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

FACROC is a fairness evaluation metric for clustering algorithms that measures fairness by comparing clustering quality between protected and non-protected groups using ROC curve analysis.

**Key Features:**
- 🎯 FACROC metric calculation with ROC curve visualization
- 🧮 AUCC (Area Under the Clustering Curve) computation
- ⚖️ Fair clustering algorithms with MCF fairlet decomposition
- 📊 Experiments on real-world datasets (Adult, COMPAS, Credit, German, Student)
- � Comprehensive evaluation metrics (balance, silhouette scores)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/congduytran12/FACROC-Experiments
cd FACROC-Experiments

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.7+, numpy, pandas, matplotlib, scipy, scikit-learn, networkx

---

## Quick Start

Run FACROC experiments on available datasets:

```bash
python facroc_experiments.py
```

This will:
1. Load preprocessed datasets and clustering results
2. Compute AUCC for protected and non-protected groups
3. Calculate FACROC fairness metric
4. Generate ROC curve visualizations in `results/`

**Example Output:**
```
Results for student_mat dataset:
  FACROC: 0.0847
  AUCC: 0.6087
  Balance: 0.7243
  Silhouette: 0.2341
```

---

## Usage

### 1. Calculate FACROC Metric

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
    showPlot=True,
    filename="results/fairness_analysis.pdf"
)
```

### 2. Fair Clustering

```python
from fair_clustering_base import fair_clustering_dataset

# Generate fair clustering results
results = fair_clustering_dataset(
    input_file="data-encoded/dataset-encode.csv",
    output_file="clustering/dataset-clustering.csv",
    k=3,                    # Number of clusters
    t=2,                    # Fairness parameter (1:t ratio)
    distance_threshold=50   # Distance threshold for fairlet formation
)
```

### 3. Run Custom Experiments

```python
from facroc_experiments import facroc_experiment

# Analyze custom dataset
results = facroc_experiment(
    dataset="data-encoded/your-dataset-encode.csv",
    clustering_result="clustering/your-dataset-clustering.csv", 
    figure_out="results/your-dataset.facroc.pdf",
    protected_attr="protected_column",
    protected_group="minority_value",
    non_protected_group="majority_value"
)
```

---

## Datasets

The repository includes experiments on real-world datasets:

| Dataset | Protected Attribute | Samples |
|---------|---------------------|---------|
| Student Performance (Math & Portuguese) | Gender | 395/649 |
| German Credit | Gender | 1,000 |
| COMPAS | Race | 6,172 |
| Adult Census | Gender | 32,561 |
| Credit Card | Gender | 30,000 |

### Data Format

**Input** (`data-encoded/`): CSV with numeric features + protected attribute column  
**Clustering** (`clustering/`): CSV with columns: `id`, `cluster_id`, `protected_attribute`  
**Results** (`results/`): Generated ROC curve visualizations (PDF)

---

## Understanding Results

### FACROC Score
- **0.0** - Perfect fairness (identical quality for both groups)
- **< 0.1** - Fair clustering
- **0.1-0.3** - Moderate fairness concerns
- **> 0.3** - Significant fairness issues

### AUCC Score
- **0.5** - Random clustering (baseline)
- **0.6-0.7** - Acceptable quality
- **0.7-0.8** - Good quality
- **> 0.8** - Excellent quality

### Balance Ratio
- **0.6-1.0** - Good balance
- **0.4-0.6** - Moderate balance
- **< 0.4** - Poor balance

### Silhouette Score
- **0.5-1.0** - Good to excellent structure
- **0.1-0.5** - Reasonable structure
- **< 0.1** - Weak structure


---

## Project Structure

```
FACROC-Experiments/
├── aucc.py                      # AUCC metric computation
├── facroc.py                    # FACROC metric calculation
├── facroc_experiments.py        # Main experiment script
├── utils.py                     # Utility functions
├── fair_clustering_base.py      # MCF fairlet decomposition
├── fairlet_decomposition.py     # Fairlet implementation
├── scalable_fair_clustering.py  # Scalable algorithms
├── kcenters.py                  # K-centers clustering
├── kmeans.py                    # K-means clustering
├── threshold_analysis.py        # Threshold analysis tools
├── data_loader.py               # Dataset loading
├── data/                        # Raw datasets
├── data-encoded/                # Preprocessed datasets
├── clustering/                  # Clustering results
└── results/                     # Generated visualizations
```

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

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

- **Repository**: [github.com/congduytran12/FACROC-Experiments](https://github.com/congduytran12/FACROC-Experiments)
- **Issues**: Please open an issue on GitHub for bugs or feature requests

