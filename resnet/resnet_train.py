## Classifying Ants and Bees - 120 train and 75 Validation images for each class
## Download & extract the dataset(very small subset of imagenet) into current dir:
## <https://download.pytorch.org/tutorial/hymenoptera_data.zip>

# ref : https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# Author : Shubhajit Das

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time 
import os
import copy

from resnet import resnet_models

# Load Data
# Data Augemntation and Normalization for Training and Validation data
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet_stats
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet_stats
    ])
    }

data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x), data_transforms[x])
                    for x in ['train', 'val']}

data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, 
                                                num_workers=4) 
                                                for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Visualize a few images
def show_images(normalized_image_tensor, title=None):
    """ Imshow for Tensor """
    inp = normalized_image_tensor.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp  + mean
    inp = inp.clip(0,1)
    plt.imshow(inp)

    if title is not None:
        plt.title(title)
        
    plt.pause(0.001)  # pause a bit so that plots are updated

# get a batch of training data
inputs, classes = next(iter(data_loaders['train']))

# make grid of inputs
out = torchvision.utils.make_grid(inputs)
show_images(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch : {epoch+1}/{num_epochs}')
        print("-" * 2)

        for phase in ["train", "valid"]:
            if phase == "train":
                scheduler.step()
                model.train() # set model to training mode
            else:
                model.eval() # set model to evaluate mode
            
            running_loss = 0
            running_corrects = 0 

            # Iterate over data
            for inputs, labels in data_loaders(phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
            
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                # track history if and only if train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase=="train":
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss /dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}')

            # deep copy the model 
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print()
    time_elapsed = time.time() - start_time
    print(f'training completed in {time_elapsed // 60}m {time_elapsed % 60}')
    print(f'Best validation accuracy: {best_acc:.4f}')

    # load best_model weights
    model.load_state_dict(best_model_wts)
    return model

    


# Build the model
# Specify Optimizer and Loss function
# Train and Validate the model