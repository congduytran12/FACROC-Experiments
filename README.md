## Implementation of paper "FACROC: a fairness measure for FAir Clustering through ROC curves"

### Overview
This repository provides an implementation of the FACROC metric, as described in the paper:

> "FACROC: a fairness measure for FAir Clustering through ROC curves"

FACROC is a fairness evaluation metric for clustering algorithms, quantifying the difference in clustering quality between protected and non-protected groups using ROC curves and the AUCC (Area Under the Clustering Curve).

---

### Repository Structure

- `aucc.py` — Computes the AUCC metric and ROC curve for a clustering solution.
- `facroc.py` — Functions for calculating and plotting the FACROC metric, including ROC curve alignment and visualization.
- `facroc_experiments.py` — Scripts for running experiments on real datasets, comparing clustering fairness between groups.
- `requirements.txt` — Python dependencies.
- `data/` — Raw datasets (e.g., student, credit, adult, compas, german credit).
- `data-encoded/` — Preprocessed/encoded versions of datasets for clustering.
- `clustering/` — Clustering results for each dataset.
- `results/` — Output plots and FACROC results.

---

### Installation

1. Clone the repository:
   ```cmd
   git clone https://github.com/congduytran12/FACROC-Experiments
   cd FACROC-Experiments
   ```
2. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```

---

### Usage

You can run the main experiment script to compute and visualize FACROC for a dataset. For example, to run on the student-por dataset:

```cmd
python facroc_experiments.py
```

Edit the `facroc_experiments.py` file to select the dataset and clustering result you want to analyze. Example configuration:

```python
facroc_student_por = facroc_experiment(
    dataset="data-encoded/student-por-encode.csv",
    clustering_result="clustering/student-por-clustering.csv",
    figure_out="results/student-por.facroc.pdf",
    protected_attr="gender",
    protected_group="F",
    non_protected_group="M",
    protected_label="Female",
    non_protected_label="Male"
)
```

The script will print AUCC and FACROC values and save a plot to the `results/` directory.

---

### Datasets

- Place your raw data in the `data/` folder.
- Encoded (numeric) versions should be in `data-encoded/`.
- Clustering results (with columns like `id`, `cluster_id`, `protected_attribute`) go in `clustering/`.

---

### Main Functions

- **AUCC**: Computes the Area Under the Clustering Curve for a group, based on pairwise distances and cluster assignments.
- **FACROC**: Measures the area between ROC curves of protected and non-protected groups, quantifying fairness.

---

### Example Output

- AUCC and FACROC values printed to the console.
- PDF plots of ROC curves and FACROC area saved in `results/`.

---

### Requirements

- Python 3.7+
- numpy
- pandas
- matplotlib
- scipy
- scikit-learn
