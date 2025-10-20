# CICIoT2023 Clustering Analysis

> **Data Mining & Clustering Research Project**  
> Comparative analysis of K-Means and Fuzzy C-Means clustering algorithms on the CICIoT2023 dataset

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Visualizations](#visualizations)
- [Save/Load Checkpoints](#saveload-checkpoints)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

This project implements a comprehensive machine learning pipeline for clustering analysis on the **CICIoT2023** (Canadian Institute for Cybersecurity IoT) dataset. The analysis compares two popular clustering algorithms:

- **K-Means Clustering**
- **Fuzzy C-Means (FCM) Clustering**

The pipeline includes:
✅ Data cleaning and preprocessing  
✅ Feature selection using Information Gain  
✅ Train/Validation/Test split (64%/16%/20%)  
✅ Multiple clustering algorithms  
✅ Comprehensive performance metrics  
✅ Publication-quality visualizations

**Target Audience:** Researchers, data scientists, and students working on IoT security, network traffic analysis, or clustering algorithms.

---

## 📊 Dataset

**CICIoT2023** - A comprehensive IoT network traffic dataset for intrusion detection and attack classification.

- **Source:** Canadian Institute for Cybersecurity
- **Domain:** IoT Network Traffic
- **Size:** Multiple CSV files (millions of records)
- **Features:** Network flow statistics, packet information, protocol details
- **Classes:** Multiple attack types and benign traffic

> ⚠️ **Note:** Due to the large size, the dataset is not included in this repository. Download it from the official source.

---

## ✨ Features

### 🔧 **Data Processing**

- Memory-efficient data loading with sampling
- Automatic missing value handling
- Duplicate removal
- Infinite value detection and correction
- Stratified train/validation/test splitting

### 🎯 **Feature Selection**

- Information Gain (Mutual Information) based selection
- Visualization of feature importance
- Configurable feature count (default: top 30)

### 🤖 **Clustering Algorithms**

- **K-Means:** Standard and MiniBatch variants
- **Fuzzy C-Means:** With membership analysis
- Elbow method for optimal cluster selection
- Silhouette analysis

### 📈 **Performance Metrics**

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score
- Fuzzy Partition Coefficient (FPC)
- Cluster distribution analysis

### 🎨 **Visualizations** (15 types)

1. PCA 2D/3D cluster visualization
2. t-SNE non-linear visualization
3. Cluster distribution (bar/pie charts)
4. Performance metrics comparison
5. Radar chart (multi-metric)
6. Confusion matrix (cluster-class mapping)
7. Feature importance plots
8. Silhouette analysis plots
9. Box plots (feature distribution)
10. Correlation heatmap
11. Pair plots (K-Means & FCM)
12. Dendrogram (hierarchical view)
13. Summary dashboard
14. Elbow curves
15. Training time comparison

### 💾 **Save/Load Functionality**

- Complete checkpoint system
- Individual component saving
- Resume work from any point
- Export to Weka (ARFF/CSV)

---

## 🚀 Installation

### Prerequisites

- Python 3.12+
- 24GB RAM recommended (or use sampling)
- macOS, Linux, or Windows

### Setup

```bash
# Clone the repository
git clone https://github.com/rnihesh/dm_cbp.git
cd dm_cbp

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn scikit-fuzzy

# Optional: Install Jupyter if not already installed
pip install jupyter
```

---

## 💻 Usage

### Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate  # macOS/Linux

# Open Jupyter Notebook
jupyter notebook test.ipynb
```

### Step-by-Step Workflow

1. **Configure Parameters** (Cell 2)

   ```python
   SAMPLE_FRACTION = 0.05  # Use 5% of data (adjust based on RAM)
   ```

2. **Run Analysis Cells** (in order)

   - Data loading and cleaning
   - Feature selection
   - Data splitting
   - K-Means clustering
   - Fuzzy C-Means clustering
   - Performance evaluation

3. **Generate Visualizations**

   - Run visualization cells (15 types)
   - All saved as high-resolution PNG files

4. **Save Your Work**

   ```python
   # Run the "Save Everything" cell
   # Creates checkpoint in saved_models/
   ```

5. **Resume Later**
   ```python
   # Run "Load Checkpoint" cell
   # Restores all variables and models
   ```

### Memory Management

For systems with limited RAM:

```python
# Adjust these parameters in the notebook:
SAMPLE_FRACTION = 0.05      # Use 5-10% of data
elbow_sample_size = 5000    # Reduce sample for elbow method
fcm_sample_size = 5000      # Reduce sample for FCM
viz_sample_size = 3000      # Reduce sample for visualizations
```

---

## 🔬 Methodology

### 1. Data Preprocessing

- **Sampling:** Random sampling (5% default) for memory efficiency
- **Cleaning:** Remove duplicates, handle missing values
- **Normalization:** StandardScaler (zero mean, unit variance)

### 2. Feature Selection

- **Method:** Information Gain (Mutual Information)
- **Threshold:** Top 30 features (configurable)
- **Validation:** Feature importance visualization

### 3. Data Split

- **Training:** 64% (model training)
- **Validation:** 16% (hyperparameter tuning)
- **Testing:** 20% (final evaluation)
- **Stratification:** Maintains class distribution

### 4. Clustering

- **K-Means:**
  - Elbow method for optimal k
  - MiniBatch variant for large datasets
  - n_init=5, max_iter=300
- **Fuzzy C-Means:**
  - Fuzziness parameter m=2
  - Error threshold=0.005
  - max_iter=1000

### 5. Evaluation

- Multiple metrics for comprehensive comparison
- Per-cluster analysis
- Cross-algorithm comparison

---

## 📊 Results

### Performance Comparison (Example)

| Metric                  | K-Means | Fuzzy C-Means |
| ----------------------- | ------- | ------------- |
| Silhouette Score        | 0.XXXX  | 0.XXXX        |
| Davies-Bouldin Index    | X.XXXX  | X.XXXX        |
| Calinski-Harabasz Score | XXX.XX  | XXX.XX        |
| Training Time (s)       | XX.XX   | XX.XX         |
| FPC                     | N/A     | 0.XXXX        |

> Results will vary based on sampling and data characteristics.

### Key Findings

✅ Both algorithms successfully identify cluster patterns  
✅ K-Means is faster but produces hard assignments  
✅ FCM provides fuzzy membership with higher computational cost  
✅ Feature selection reduces dimensionality by ~XX%  
✅ Sampling (5%) maintains statistical validity

---

## 🎨 Visualizations

All visualizations are saved as **300 DPI** publication-quality PNG files:

```
viz_1_pca_2d_comparison.png          # PCA 2D clusters
viz_2_pca_3d_comparison.png          # PCA 3D clusters
viz_3_tsne_comparison.png            # t-SNE visualization
viz_4_cluster_distribution.png       # Cluster sizes
viz_5_metrics_comparison.png         # Performance metrics
viz_6_radar_chart.png                # Multi-metric comparison
viz_7_confusion_matrix.png           # Cluster-class mapping
viz_8_feature_importance.png         # Information Gain
viz_9_silhouette_plot.png            # Per-cluster quality
viz_10_boxplots.png                  # Feature distributions
viz_11_correlation_heatmap.png       # Feature correlations
viz_12_pairplot_kmeans.png           # K-Means relationships
viz_13_pairplot_fcm.png              # FCM relationships
viz_14_dendrogram.png                # Hierarchical view
viz_15_summary_dashboard.png         # Complete overview
```

### Paper Sections Guide

- **Introduction:** `viz_11`, `viz_8`
- **Methodology:** `viz_4`, `viz_14`
- **Results:** `viz_1-3`, `viz_5`, `viz_9`
- **Discussion:** `viz_6`, `viz_7`, `viz_15`
- **Appendix:** `viz_10`, `viz_12-13`

---

## 💾 Save/Load Checkpoints

### Saving Your Work

```python
# Complete checkpoint (everything)
# Run "Save Everything" cell
# Output: saved_models/complete_checkpoint_YYYYMMDD_HHMMSS.pkl
```

### Loading Previous Work

```python
# List available checkpoints
# Run "List Checkpoints" cell

# Load most recent checkpoint
# Run "Load Checkpoint" cell
# All variables and models restored!
```

### What Gets Saved

- ✅ Processed datasets (train/val/test)
- ✅ Scaled data
- ✅ Trained models (K-Means, Scaler)
- ✅ Clustering results
- ✅ Performance metrics
- ✅ Feature information
- ✅ Configuration parameters

---

## 📁 Project Structure

```
dm_cbp/
├── test.ipynb                    # Main Jupyter notebook
├── README.md                     # This file
├── .gitignore                    # Git ignore rules
├── .venv/                        # Virtual environment (excluded)
├── saved_models/                 # Checkpoints & models (excluded)
│   ├── complete_checkpoint_*.pkl
│   ├── kmeans_final.joblib
│   ├── scaler_final.joblib
│   ├── datasets_*.pkl
│   ├── feature_info_*.pkl
│   └── performance_metrics_*.json
├── viz_*.png                     # Generated visualizations
├── feature_information_gain.csv  # Feature scores
├── clustering_comparison.csv     # Algorithm comparison
└── ciciot2023_*.arff/.csv        # Exported Weka files (excluded)
```

---

## 📦 Requirements

### Python Libraries

```txt
pandas>=2.3.3
numpy>=2.3.4
scikit-learn>=1.7.2
matplotlib>=3.10.7
seaborn>=0.13.2
scikit-fuzzy>=0.5.0
joblib>=1.4.2
```

### System Requirements

- **Minimum:** 8GB RAM (with 5% sampling)
- **Recommended:** 24GB RAM (for 10% sampling)
- **Optimal:** 64GB+ RAM (for full dataset)
- **Storage:** ~5GB for checkpoints and visualizations

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- Additional clustering algorithms (DBSCAN, Hierarchical, etc.)
- GPU acceleration support
- Real-time visualization dashboards
- Automated hyperparameter tuning
- Additional datasets integration
- Performance optimizations

---

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@misc{dm_cbp_2025,
  author = {Your Name},
  title = {CICIoT2023 Clustering Analysis: Comparative Study of K-Means and Fuzzy C-Means},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/rnihesh/dm_cbp}
}
```

### Dataset Citation

```bibtex
@misc{ciciot2023,
  title={CICIoT2023: A Real-Time Dataset for IoT Intrusion Detection},
  author={Canadian Institute for Cybersecurity},
  year={2023},
  url={https://www.unb.ca/cic/datasets/}
}
```

---

## 📝 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2025 Nihesh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- **CIC (Canadian Institute for Cybersecurity)** - For the CICIoT2023 dataset
- **scikit-learn** - Machine learning library
- **scikit-fuzzy** - Fuzzy logic library
- **Open Source Community** - For continuous support and development

---

## 📧 Contact

**Project Maintainer:** Nihesh  
**GitHub:** [@rnihesh](https://github.com/rnihesh)  
**Repository:** [dm_cbp](https://github.com/rnihesh/dm_cbp)

---

## 🔄 Version History

- **v1.0.0** (2025-10-21) - Initial release
  - Complete clustering pipeline
  - 15 visualization types
  - Save/load functionality
  - Memory optimization
  - Weka export support

---

## 🚀 Future Roadmap

- [ ] Add DBSCAN clustering
- [ ] Implement hierarchical clustering
- [ ] GPU acceleration (RAPIDS)
- [ ] Interactive dashboards (Plotly/Dash)
- [ ] Automated hyperparameter tuning
- [ ] Deep learning-based clustering
- [ ] Real-time streaming analysis
- [ ] Docker containerization
- [ ] Web API for model deployment
- [ ] Comparative analysis with more datasets

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

Made with ❤️ for the Data Science Community

</div>
