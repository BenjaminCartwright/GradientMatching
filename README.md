# GradientMatching

## Overview
Data distillation (also called data condensation) is the process of compressing a large dataset into a smaller, synthetic dataset that preserves essential information for training machine learning models with comparable performance. Gradient matching is a technique for dataset distillation where the synthetic set is designed to mimic the loss dynamics of a certain model on the original data by treating the synthetic set as a learnable parameter and minimizing model loss gradient discrepancies between the original and synthetic sets. This repository explores how the following effect the gradient matching process:
1. Model Selection: We conduct gradient matching using MLP, CNN and LeNet models
2. Synthetic Set Initialization: We initialize our synthetic data as pure random, full white, full black, grey and per class samples
3. Higher Order Gradient Terms: We introduce higher order derivative terms (Hessian) into our loss functions 


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

