import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import os
#from gradient_matching_helpers import gradient_matching, get_network, get_daparam, evaluate_synset, save_image, match_loss
from gm_utils import gradient_matching, get_network, get_daparam, evaluate_synset, save_image, match_loss, get_dataset
from helpers import plot_gradient_loss_by_iteration_by_class_and_total


# Define arguments
class Args:
    method = "DC"
    dataset = "FashionMNIST"
    model =  "ConvNet"
    ipc = 1  # Images per class
    num_exp = 1  # Single experiment
    num_eval = 1  # Number of evaluations per experiment
    Iteration = 1  # Training iterations
    lr_img = 0.1  # Learning rate for synthetic images
    lr_net = 0.01  # Learning rate for network
    batch_real = 128  # Batch size for real data
    batch_train = 128  # Batch size for synthetic data training
    init = "noise"  # Initialize synthetic data from noise
    eval_mode = "S"
    data_path = "./data"
    save_path = "./results"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epoch_eval_train = 1  # Number of epochs to train evaluation models
    outer_loop = 1  # Number of outer-loop updates for synthetic data\
    dis_metric = 'ours'

args = Args()

channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
'''
# Prepare dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform)

dst_train = [(img, label) for img, label in zip(train_dataset.data.float() / 255.0, train_dataset.targets)]
testloader = DataLoader(test_dataset, batch_size=args.batch_real, shuffle=False)

num_classes = 10
channel = 1
im_size = (28, 28)
mean = [0.1307]  # Mean of MNIST
std = [0.3081]  # Std of MNIST
'''
# Evaluation iterations
eval_it_pool = [0, args.Iteration]
model_eval_pool = ["ConvNet"]
print("Running Gradient Matching")
# Call gradient matching function
accs_all_exps, data_save, gradient_losses, class_gradient_losses = gradient_matching(
    args=args,
    dst_train=dst_train,
    num_classes=num_classes,
    im_size=im_size,
    channel=channel,
    std=std,
    mean=mean,
    testloader=testloader,
    get_network=get_network,
    evaluate_synset=evaluate_synset,
    save_image=save_image,
    match_loss=match_loss,
    get_daparam=get_daparam
)

# Print results
'''
print("Final accuracy metrics:")
for model, accs in accs_all_exps.items():
    print(f"Model: {model}, Mean Accuracy: {np.mean(accs):.4f}, Std: {np.std(accs):.4f}")
print('\n')
print(accs_all_exps)
print('\n')
print(gradient_losses)
print('\n')
print(class_gradient_losses)
# Create a new ConvNet model instance
'''
plot_gradient_loss_by_iteration_by_class_and_total(class_gradient_losses, gradient_losses, save_path="loss_plot")