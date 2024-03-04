#!/usr/bin/env python
# coding: utf-8


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

import wandb
import os
from argparse import ArgumentParser


# PyTorch offers domain-specific libraries such as `TorchText`, `TorchVision`, and `TorchAudio`, all of which include datasets. For this tutorial, we'll be using a TorchVision dataset.
# 
# The ``torchvision.datasets`` module contains ``Dataset`` objects for many real-world vision datasets, such as CIFAR and COCO. In this tutorial, we'll use the **FashionMNIST** dataset. Every TorchVision ``Dataset`` includes two arguments: ``transform`` and ``target_transform`` to modify the samples and labels respectively.



# --------------------------------------------------------------------
#                          Parse Arguements
#
#     Using Arguments to run your training script is mandatory to use 
#     wandb sweeps. In this section, you add all hyperparameters that
#     needed to tune your model. 
# --------------------------------------------------------------------
arg_parser = ArgumentParser()
arg_parser.add_argument("--lr", type=float, default=1e-2) # Here we specified the flag, type of the input, and the default value given (optional).
arg_parser.add_argument("--epochs", type=int, default=50)
arg_parser.add_argument("--batch_size", type=int, default=64)

# Wandb param (optional)
arg_parser.add_argument("--disable_wandb", dest="enable_wandb", action="store_false") # If you want enable and disable wandb monitoring easily
arg_parser.set_defaults(enable_wandb=True)

config = arg_parser.parse_args().__dict__

wandb_log = config["enable_wandb"]


##_________________________________________________________________________________________________________##
##_________________________________________________________________________________________________________##




# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


# We pass the ``Dataset`` as an argument to ``DataLoader``. This wraps an iterable over our dataset, and supports automatic batching, sampling, shuffling, and multiprocess data loading. Here we define a batch size of 64, where each element in the dataloader iterable will return a batch of 64 features and labels.



batch_size = config['batch_size']

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
    
# Display sample data
figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    idx = torch.randint(len(test_data), size=(1,)).item()
    img, label = test_data[idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


# ## Creating models
# To define a neural network in PyTorch, we create a class that inherits from `nn.Module`. We define the layers of the network in the ``__init__`` function and specify how data will pass through the network in the ``forward`` function. To accelerate operations in the neural network, we move it to the GPU if available.
# 



# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))



# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


# ## Optimizing the model parameters
# To train a model, we need a loss function and an optimizer.  We'll be using `nn.CrossEntropyLoss` for loss and `Stochastic Gradient Descent` for optimization.




loss_fn = nn.CrossEntropyLoss()
learning_rate = config['lr']
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# In a single training loop, the model makes predictions on the training dataset (fed to it in batches), and back-propagates the prediction error to adjust the model's parameters. 
# 
# 


# Monitor training using Weight & Biases to get insights
# First thing is to initiate a project using `wandb.init()`



if wandb_log:
    run = wandb.init(project='wandb_test',
                    name=f"Cloths Classificatione {learning_rate}", # optional: you can name your experiment run here (You can add insights on your hyperparameters to easily distinguish between you different runs)
                    config=config
                    )


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            ##___________________________________________________________________##
            if wandb_log:
                wandb.log({"Train Loss":loss,"Train Accuracy":100*(current/size)})
            ##___________________________________________________________________##





# We can also check the model's performance against the test dataset to ensure it is learning.
# 



def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    ##___________________________________________________________________##
    if wandb_log:
        wandb.log({"Test Loss":test_loss,"Test Accuracy":100*correct})
    ##___________________________________________________________________##


# The training process is conducted over several iterations (*epochs*). During each epoch, the model learns parameters to make better predictions. We print the model's accuracy and loss at each epoch; we'd like to see the accuracy increase and the loss decrease with every epoch.
# 
# 


epochs = config['epochs']

start_epoch = 0

if not os.path.exists("/home/anass.grini/anass.grini/HPC_training/checkpoints"):
    os.mkdir("/home/anass.grini/anass.grini/HPC_training/checkpoints") 


if os.path.exists("/home/anass.grini/anass.grini/HPC_training/checkpoints/saved_model_checkpoint.pt"): # check if there's any checkpoints saved and load it
    print("Checkpoint found! Continue training....")
    checkpoint = torch.load("/home/anass.grini/anass.grini/HPC_training/checkpoints/saved_model_checkpoint.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] # Save the epoch number where the model's saved


for t in range(start_epoch,epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)

    if epochs%10==0:
        torch.save({'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch':t
        },
        "/home/anass.grini/anass.grini/HPC_training/checkpoints/saved_model_checkpoint.pt")

print("Done!")


# The accuracy will initially not be very good (that's OK!). Try running the loop for more `epochs` or adjusting the `learning_rate` to a bigger number. It might also be the case that the model configuration we chose might not be the optimal one for this kind of problem (it isn't). Later courses will delve more into the model shapes that work for vision problems.

# Saving Models
# -------------
# A common way to save a model is to serialize the internal state dictionary (containing the model parameters).
# 
# 

# In[ ]:


torch.save(model.state_dict(), "/home/anass.grini/anass.grini/HPC_training/data/model.pth")
print("Saved PyTorch Model State to model.pth")


# Loading Models
# ----------------------------
# 
# The process for loading a model includes re-creating the model structure and loading
# the state dictionary into it. 
# 
# 


model = NeuralNetwork()
model.load_state_dict(torch.load("/home/anass.grini/anass.grini/HPC_training/data/model.pth"))


# This model can now be used to make predictions.
# 
# 

# In[12]:


classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')


# Congratulations! You have completed the PyTorch beginner tutorial! We hope this tutorial has helped you get started with deep learning on PyTorch.
