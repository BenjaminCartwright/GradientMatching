# GradientMatching

## Overview
Data distillation (also called data condensation) is the process of compressing a large dataset into a smaller, synthetic dataset that preserves essential information for training machine learning models with comparable performance. Gradient matching is a technique for dataset distillation where the synthetic set is designed to mimic the loss dynamics of a certain model on the original data by treating the synthetic set as a learnable parameter and minimizing model loss gradient discrepancies between the original and synthetic sets. This repository explores how the following effect the gradient matching process:
1. Model Selection: We conduct gradient matching using MLP, CNN and LeNet models
2. Synthetic Set Initialization: We initialize our synthetic data as pure random, full white, full black, grey and per class samples
3. Higher Order Gradient Terms: We introduce higher order derivative terms (Hessian) into our loss functions 

A write-up of the experiement results and further explanantion of gradient matching and the general methodologies used can be found in the attached paper here. The experiments in this repository are conducted on the MNIST dataset but FashionMNIST, SVHN, CIFAR10, CIFAR100 and TinyImageNet are supported.  

## Repository Structure

```
GradientMatching/
├── data/                   # Sample datasets for experimentation
├── write_up.pdf            # The report for this project submitted as my Computational Statistics final
├── networks.py             # Implementation of MLP, CNN, and LeNet models
├── gm_utils_v2.py          # Gradient matching functions and utilities
├── helpers.py              # General helpers and utilities 
├── results_t_1/            # Experiment 1 results and visualizations
├── results_t_2/            # Experiment 3 results and visualizations
├── results_t_3/            # Experiment 3 results and visualizations
├──Initial_Model_Eval.ipynb # Notebook used to evaluate our models on the original dataset
├──EXP1.ipynb               # Notebook for experiment 1
├──EXP2.ipynb               # Notebook for experiment 2
├──EXP3.ipynb               # Notebook for experiment 3
├──image_joining.ipynb      # Notebook used to combine images from original and synthetic set for display purposes 
├── README.md               # Repository documentation (this file)
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
- **Email:** benjaminccartwright@gmail.com

