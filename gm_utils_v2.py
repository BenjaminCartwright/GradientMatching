import torch
import torch.nn as nn
import time
import numpy as np
from torchvision import datasets, transforms
import os
import torch.optim as optim
from networks import MLP, ConvNet, LeNet, ResNet18
from torchvision.utils import save_image as torchvision_save_image
import copy
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
def gradient_matching(
    args,
    dst_train,
    num_classes,
    im_size,
    channel,
    std,
    mean,
    eval_it_pool,
    testloader,
    get_network,
    evaluate_synset,
    save_image,
    match_loss,
    get_daparam,
    DiffAugment=None,
):
    model_eval_pool = [args.model]
    accs_all_exps = {key: [] for key in model_eval_pool}  # Performance tracking
    gradient_losses = []  # Track gradient matching loss per iteration
    class_gradient_losses = {c: [] for c in range(num_classes)}  # Per-class losses
    data_save = []  # For saving final synthetic data

    # Initialize synthetic data and class indices
    image_syn, label_syn, indices_class, images_all = init_synset(dst_train, num_classes, channel, im_size, args)
    optimizer_img = optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    criterion = nn.CrossEntropyLoss().to(args.device)

    print(f'{get_time()} Training begins')
    for it in range(args.Iteration + 1):
        print(f'Iteration# {it}')
        total_gradient_loss = 0.0  # Accumulate total gradient loss for the iteration

        # Evaluate synthetic data
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                print(f'Evaluating model {model_eval} at iteration {it}...')
                accs = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
                    #image_syn_eval = copy.deepcopy(image_syn.detach())
                    #label_syn_eval = copy.deepcopy(label_syn.detach())
                    image_syn_eval = copy.deepcopy(image_syn.detach()).to(args.device)
                    label_syn_eval = copy.deepcopy(label_syn.detach()).to(args.device)

                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                    accs.append(acc_test)
                accs_all_exps[model_eval] += accs

            # Save visualization
            save_name = os.path.join(
                args.save_path,
                f'vis_{args.method}_{args.dataset}_{args.model}_{args.ipc}ipc_single_iter{it}.png',
            )

            image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
            for ch in range(channel):
                image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
            image_syn_vis[image_syn_vis < 0] = 0.0
            image_syn_vis[image_syn_vis > 1] = 1.0
            save_image(image_syn_vis, save_name, nrow=args.ipc)

        # Train synthetic data
        net = get_network(args.model, channel, num_classes, im_size).to(args.device)
        net_parameters = list(net.parameters())
        optimizer_net = optim.SGD(net.parameters(), lr=args.lr_net)

        for ol in range(args.outer_loop):
            loss = torch.tensor(0.0).to(args.device)
            class_losses = []  # Track per-class loss for this outer loop
            for c in range(num_classes):
                # Pass indices_class and images_all to get_images
                #img_real = get_images(c, args.batch_real, indices_class, images_all)
                #lab_real = torch.ones(len(img_real), device=args.device, dtype=torch.long) * c
                img_real = get_images(c, args.batch_real, indices_class, images_all).to(args.device)
                lab_real = torch.ones(len(img_real), device=args.device, dtype=torch.long) * c


                #img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc]
                #lab_syn = torch.ones(args.ipc, device=args.device, dtype=torch.long) * c
                img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc].to(args.device)
                lab_syn = torch.ones(args.ipc, device=args.device, dtype=torch.long) * c


                output_real = net(img_real)
                loss_real = criterion(output_real, lab_real)
                gw_real = torch.autograd.grad(loss_real, net_parameters, retain_graph=True)

                output_syn = net(img_syn)
                loss_syn = criterion(output_syn, lab_syn)
                gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                if args.hessian:
                    class_loss = match_loss_with_hessian(gw_syn, gw_real, net_parameters, loss_real, args)
                    class_losses.append(class_loss.item())  # Track class-specific gradient loss
                else:
                    class_loss = match_loss(gw_syn, gw_real, args)
                    class_losses.append(class_loss.item())  # Track class-specific gradient loss
                loss += class_loss

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()

            # Track losses
            total_gradient_loss += loss.item()
            for c, class_loss in enumerate(class_losses):
                class_gradient_losses[c].append(class_loss)

        # Save total gradient loss for the iteration
        gradient_losses.append(total_gradient_loss)
        print(f"Iteration {it}: Total Gradient Loss = {total_gradient_loss:.4f}")

    data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])

    # Return all tracked metrics
    return accs_all_exps, data_save, gradient_losses, class_gradient_losses


def init_synset(dst_train, num_classes, channel, im_size, args):
    """
    Initialize synthetic dataset with specified initialization method.

    Parameters:
    - dst_train: Real dataset for training.
    - num_classes: Number of classes in the dataset.
    - channel: Number of channels in the images.
    - im_size: Tuple specifying the image size (height, width).
    - args: Namespace of experiment arguments.

    Returns:
    - image_syn: Synthetic images.
    - label_syn: Corresponding synthetic labels.
    - indices_class: List of indices for each class.
    - images_all: Tensor of all images.
    """
    images_all = torch.cat([torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))], dim=0).to(args.device)
    labels_all = torch.tensor([dst_train[i][1] for i in range(len(dst_train))], dtype=torch.long, device=args.device)
    indices_class = [[] for _ in range(num_classes)]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)

    print('Class distribution:')
    for c in range(num_classes):
        print(f'Class {c}: {len(indices_class[c])} images')

    # Initialize synthetic data
    if args.init == 'noise':
        print('Initializing synthetic data from random noise...')
        image_syn = torch.randn(
            size=(num_classes * args.ipc, channel, im_size[0], im_size[1]),
            dtype=torch.float,
            requires_grad=True,
            device=args.device,
        )
    elif args.init == 'black':
        print('Initializing synthetic data with black images...')
        image_syn = torch.zeros(
            size=(num_classes * args.ipc, channel, im_size[0], im_size[1]),
            dtype=torch.float,
            requires_grad=True,
            device=args.device,
        )
    elif args.init == 'white':
        print('Initializing synthetic data with white images...')
        image_syn = torch.ones(
            size=(num_classes * args.ipc, channel, im_size[0], im_size[1]),
            dtype=torch.float,
            requires_grad=True,
            device=args.device,
        )
    elif args.init == 'grey':
        print('Initializing synthetic data with grey images...')
        image_syn = torch.ones(
            size=(num_classes * args.ipc, channel, im_size[0], im_size[1]),
            dtype=torch.float,
            requires_grad=True,
            device=args.device,
        )
        image_syn.data.mul_(0.5)  # Scale the tensor to grey values

    elif args.init == 'real':
        print('Initializing synthetic data from random real images...')
        image_syn = torch.empty(
            size=(num_classes * args.ipc, channel, im_size[0], im_size[1]),
            dtype=torch.float,
            requires_grad=True,
            device=args.device,
        )
        for c in range(num_classes):
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc, indices_class, images_all).detach().data
    else:
        raise ValueError(f"Unsupported initialization method: {args.init}")

    label_syn = torch.tensor(
        np.array([np.ones(args.ipc) * i for i in range(num_classes)]),
        dtype=torch.long,
        device=args.device,
    ).view(-1)

    return image_syn, label_syn, indices_class, images_all

def hessian_vector_product(loss, params, vec):
    """
    Compute the Hessian-vector product (HVP).

    Args:
        loss (torch.Tensor): The scalar loss value.
        params (list): List of model parameters with respect to which gradients are computed.
        vec (list): List of tensors (same shape as params) representing the vector to multiply with the Hessian.

    Returns:
        hvp (list): Hessian-vector product, one tensor for each parameter.
    """
    # First-order gradients
    grad1 = torch.autograd.grad(loss, params, create_graph=True)
    
    # Compute Hessian-vector product (HVP)
    hvp = torch.autograd.grad(
        grad1, params, grad_outputs=vec, retain_graph=True
    )
    return hvp

def match_loss_with_hessian(gw_syn, gw_real, params, loss_real, args):
    """
    Compute the gradient matching loss with Hessian terms.

    Args:
        gw_syn (list): Gradients from synthetic data.
        gw_real (list): Gradients from real data.
        params (list): Model parameters.
        loss_real (torch.Tensor): Real loss value for Hessian computation.
        args: Argument namespace for additional parameters.

    Returns:
        loss (torch.Tensor): Gradient matching loss with Hessian terms.
    """
    loss = 0.0

    for g_syn, g_real in zip(gw_syn, gw_real):
        if args.dis_metric == 'ours':
            # Gradient Matching Loss (L2 norm)
            loss += torch.sum((g_syn - g_real) ** 2)
        elif args.dis_metric == 'cos':
            # Cosine Similarity Loss
            loss += 1 - torch.sum(g_syn * g_real) / (torch.norm(g_syn) * torch.norm(g_real) + 1e-6)

    # Add Hessian term
    vec = [(g_real - g_syn).detach() for g_real, g_syn in zip(gw_real, gw_syn)]
    hvp = hessian_vector_product(loss_real, params, vec)
    hvp = [h.detach() for h in hvp]

    for v, h in zip(vec, hvp):
        loss += torch.sum(v * h)  # Add the Hessian-related term

    return loss


def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis


def match_loss(gw_syn, gw_real, args):
    dis = torch.tensor(0.0).to(args.device)

    if args.dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('unknown distance function: %s'%args.dis_metric)

    return dis

def get_images(c, n, indices_class, images_all):
    """
    Retrieve a random set of images for a specific class.

    Parameters:
    - c: Class index.
    - n: Number of images to retrieve.
    - indices_class: List of indices for each class.
    - images_all: Tensor of all images.

    Returns:
    - selected_images: Tensor of selected images for class `c`.
    """
    idx_shuffle = np.random.permutation(indices_class[c])[:n]
    selected_images = images_all[idx_shuffle]
    # Ensure shape is (batch_size, channel, height, width)
    if selected_images.ndim == 3:  # For grayscale images
        selected_images = selected_images.unsqueeze(1)
    return selected_images


def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling



def get_network(model, channel, num_classes, im_size=(32, 32)):
        torch.random.manual_seed(int(time.time() * 1000) % 100000)
        net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

        if model == 'MLP':
            net = MLP(channel=channel, num_classes=num_classes)
        elif model == 'ConvNet':
            net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
        elif model == 'LeNet':
            net = LeNet(channel=channel, num_classes=num_classes)
        elif model == 'ResNet18':
            net = ResNet18(channel=channel, num_classes=num_classes)
        else:
            net = None
            exit('unknown model: %s'%model)

        gpu_num = torch.cuda.device_count()
        if gpu_num>0:
            device = 'cuda'
            if gpu_num>1:
                net = nn.DataParallel(net)
        else:
            device = 'cpu'
        net = net.to(device)

        return net

def evaluate_synset(it_eval, net_eval, image_syn, label_syn, testloader, args):
    # Train net_eval on synthetic data
    synthetic_dataset = TensorDataset(image_syn, label_syn)
    trainloader = DataLoader(synthetic_dataset, batch_size=args.batch_train, shuffle=True)
    optimizer = torch.optim.Adam(net_eval.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Train for a fixed number of epochs
    net_eval.train()
    for epoch in range(args.epoch_eval_train):
        for inputs, labels in trainloader:
            inputs = inputs.to(args.device)  # <-- Ensure inputs are on the correct device
            labels = labels.to(args.device)
            optimizer.zero_grad()
            outputs = net_eval(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate on real test data
    net_eval.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(args.device)  # <-- Ensure inputs are on the correct device
            labels = labels.to(args.device)
            outputs = net_eval(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc_test = correct / total
    return None, None, acc_test
    

def train_and_test_model(model, train_loader, test_loader, num_epochs, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # Testing
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    return test_accuracy


def save_image(images, filename, nrow):
    """
    Save a grid of images to a file.

    Parameters:
    - images (torch.Tensor): Tensor of images to save, shape (N, C, H, W).
    - filename (str): Path to save the image grid.
    - nrow (int): Number of images in each row of the grid.
    """
    # Check if images are in a valid range (e.g., 0-1 for normalized images)
    if images.min() < 0 or images.max() > 1:
        print("Warning: Images are not in the range [0, 1]. Clipping values.")
        images = torch.clamp(images, 0, 1)

    # Save the image grid
    torchvision_save_image(images, filename, nrow=nrow)
    print(f"Image grid saved to {filename}")


def get_daparam(dataset, model_train, model_eval, ipc):
    # Dummy augmentation parameters
    return {"strategy": "none"}

def get_time():
    """
    Returns the current timestamp as a formatted string.
    """
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

def get_dataset(dataset, data_path):
    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'FashionMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.2861]
        std = [0.3530]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'SVHN':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.SVHN(data_path, split='train', download=True, transform=transform)  # no augmentation
        dst_test = datasets.SVHN(data_path, split='test', download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'TinyImageNet':
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data = torch.load(os.path.join(data_path, 'tinyimagenet.pt'), map_location='cpu')

        class_names = data['classes']

        images_train = data['images_train']
        labels_train = data['labels_train']
        images_train = images_train.detach().float() / 255.0
        labels_train = labels_train.detach()
        for c in range(channel):
            images_train[:,c] = (images_train[:,c] - mean[c])/std[c]
        dst_train = TensorDataset(images_train, labels_train)  # no augmentation

        images_val = data['images_val']
        labels_val = data['labels_val']
        images_val = images_val.detach().float() / 255.0
        labels_val = labels_val.detach()

        for c in range(channel):
            images_val[:, c] = (images_val[:, c] - mean[c]) / std[c]

        dst_test = TensorDataset(images_val, labels_val)  # no augmentation

    else:
        exit('unknown dataset: %s'%dataset)


    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader
