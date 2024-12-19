import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader, TensorDataset

def plot_gradient_loss_by_iteration_by_class_and_total(class_gradient_losses, gradient_losses, save_path=None):
    """
    Visualize and save gradient matching loss by iteration for each class and the total gradient loss.

    Parameters:
    - class_gradient_losses: Dictionary where keys are class indices and values are lists of gradient losses for each iteration.
    - gradient_losses: List of total gradient matching loss values for each iteration.
    - save_path: Base path to save the plots. Each plot will have `class#` or `total` appended to the filename.
    """
    # Plot for each class
    for c, losses in class_gradient_losses.items():
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(losses)), losses, marker='o', linestyle='-', label=f'Class {c}')
        plt.xlabel('Iteration')
        plt.ylabel('Gradient Loss')
        plt.title(f'Gradient Matching Loss by Iteration (Class {c})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        class_save_path = f"{save_path}_class{c}.png" if save_path else None
        if class_save_path:
            plt.savefig(class_save_path)
            print(f"Gradient loss by iteration plot for Class {c} saved to {class_save_path}")
        plt.close()

    # Plot for total gradient loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(gradient_losses)), gradient_losses, marker='o', linestyle='-', label='Total Gradient Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Loss')
    plt.title('Total Gradient Matching Loss by Iteration')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    total_save_path = f"{save_path}_total.png" if save_path else None
    if total_save_path:
        plt.savefig(total_save_path)
        print(f"Total gradient loss by iteration plot saved to {total_save_path}")
    plt.close()


def plot_gradient_loss_by_iteration_by_class(class_gradient_losses, save_path=None):
    """
    Visualize gradient matching loss by iteration for each class.

    Parameters:
    - class_gradient_losses: Dictionary where keys are class indices and values are lists of gradient losses for each iteration.
    - save_path: Optional path to save the plot as an image.
    """
    plt.figure(figsize=(12, 8))
    for c, losses in class_gradient_losses.items():
        plt.plot(range(len(losses)), losses, marker='o', linestyle='-', label=f'Class {c}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Loss')
    plt.title('Gradient Matching Loss by Iteration (Per Class)')
    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Gradient loss by iteration by class plot saved to {save_path}")
    plt.show()


def plot_gradient_loss_by_iteration(gradient_losses, save_path=None):
    """
    Visualize total gradient matching loss by iteration.

    Parameters:
    - gradient_losses: List of total gradient matching loss values for each iteration.
    - save_path: Optional path to save the plot as an image.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(gradient_losses)), gradient_losses, marker='o', linestyle='-', label='Total Gradient Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Total Gradient Loss')
    plt.title('Gradient Matching Loss by Iteration')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Gradient loss by iteration plot saved to {save_path}")
    plt.show()


def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Test the model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


def display_category_image(category, X_train, y_train, synthetic_X, synthetic_y, is_conv=True):
    """Display an image from the specified category in both X_train and the synthetic dataset."""
    plt.figure(figsize=(10, 5))

    # Find an image in X_train
    train_indices = torch.where(y_train == category)[0]
    if len(train_indices) > 0:
        train_image = X_train[train_indices[0]].squeeze()
        plt.subplot(1, 2, 1)
        if is_conv:
            train_image = train_image.view(28, 28)
        else:
            train_image = train_image.view(28, 28)
        plt.imshow(train_image.cpu().numpy(), cmap="gray")
        plt.title("Original Data", fontsize=16)
    else:
        print("No image found in X_train for this category.")

    # Find an image in synthetic_X
    synthetic_indices = torch.where(synthetic_y == category)[0]
    if len(synthetic_indices) > 0:
        synthetic_image = synthetic_X[synthetic_indices[0]].squeeze()
        plt.subplot(1, 2, 2)
        if is_conv:
            synthetic_image = synthetic_image.view(28, 28)
        else:
            synthetic_image = synthetic_image.view(28, 28)
        plt.imshow(synthetic_image.detach().cpu().numpy(), cmap="gray")
        plt.title("Synthetic Data", fontsize=16)
    else:
        print("No image found in synthetic_X for this category.")

    plt.show()

def display_synthetic_image_by_index(index, synthetic_X, is_conv=True):
    """Display an image from synthetic_X given its index."""
    plt.figure(figsize=(5, 5))
    synthetic_image = synthetic_X[index].squeeze()
    if is_conv:
        synthetic_image = synthetic_image.view(28, 28)
    else:
        synthetic_image = synthetic_image.view(28, 28)
    plt.imshow(synthetic_image.detach().cpu().numpy(), cmap="gray")
    plt.title(f"Synthetic Image at Index {index}", fontsize=16)
    plt.axis("off")
    plt.show()


