# GradientMatching

## Overview

This repository explores gradient matching techniques to reduce dataset size while maintaining good accuracy across machine learning models. Specifically, the project focuses on:

1. Evaluating synthetic set initialization methods for Convolutional Neural Networks (CNNs).
2. Analyzing model convergence by incorporating higher-order gradient terms, such as the Hessian.

The project supports various machine learning models, including:
- Multi-Layer Perceptrons (MLPs)
- Convolutional Neural Networks (CNNs)
- LeNet

## Goals

The primary objectives of this project are:
- **Dataset Reduction:** Reduce dataset size while preserving model performance.
- **Gradient Matching Techniques:** Implement and evaluate gradient matching methods.
- **Hessian Analysis:** Investigate the impact of adding higher-order gradient terms on model convergence.

## Repository Structure

```
GradientMatching/
├── data/               # Sample datasets for experimentation
├── networks.py             # Implementation of MLP, CNN, and LeNet models
├── gm_utils_v2.py/              # Helper functions and utilities
├── results_t_1/            # Experiment 1 results and visualizations
├── results_t_2/            # Experiment 3 results and visualizations
├── results_t_4/            # Experiment 3 results and visualizations
├── README.md           # Repository documentation (this file)
```
Each experiement is runin a notebook file (EXP1, EXP2, EXP3). I ran these in google colab but you can download this repo and run with jupyter notebook as well. 
## Getting Started

### Prerequisites

To run the code in this repository, you'll need:
- Python 3.8+
- CUDA (for GPU support, optional but recommended)


## Results

Key results from this project include:
- Significant dataset size reduction without major performance degradation.
- Improved convergence for CNN models with higher-order gradient terms.
- Insights into the initialization techniques for synthetic datasets.

Visualizations and detailed analyses are available in the `results/` directory.

## Contact

For questions or feedback, feel free to reach out to:
- **Author:** Ben Cartwright
- **Email:** bennygoes@example.com

