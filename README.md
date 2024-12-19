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
├── models/             # Implementation of MLP, CNN, and LeNet models
├── synthetic/          # Methods for synthetic set initialization
├── experiments/        # Experiment scripts and analysis
├── utils/              # Helper functions and utilities
├── results/            # Results and visualizations
├── README.md           # Repository documentation (this file)
└── requirements.txt    # Python dependencies
```

## Getting Started

### Prerequisites

To run the code in this repository, you'll need:
- Python 3.8+
- CUDA (for GPU support, optional but recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bennygoes/GradientMatching.git
   cd GradientMatching
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Dataset Preparation:**
   - Place your dataset in the `data/` directory or use one of the provided datasets.

2. **Running Experiments:**
   - To train a model with a reduced dataset, use:
     ```bash
     python experiments/train_model.py --model cnn --dataset your_dataset_name --gradient_matching
     ```
   - To analyze the impact of the Hessian term:
     ```bash
     python experiments/train_model.py --model cnn --dataset your_dataset_name --use_hessian
     ```

3. **Visualizing Results:**
   - Use the provided scripts in `results/` to visualize training performance and dataset reduction impact.

### Example

Here is an example of training a CNN model with synthetic set initialization and Hessian-based convergence analysis:
```bash
python experiments/train_model.py \
  --model cnn \
  --dataset mnist \
  --gradient_matching \
  --use_hessian \
  --output_dir results/mnist_experiment
```

## Results

Key results from this project include:
- Significant dataset size reduction without major performance degradation.
- Improved convergence for CNN models with higher-order gradient terms.
- Insights into the initialization techniques for synthetic datasets.

Visualizations and detailed analyses are available in the `results/` directory.

## Contributing

Contributions are welcome! If you'd like to contribute:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, feel free to reach out to:
- **Author:** Ben Cartwright
- **Email:** bennygoes@example.com

