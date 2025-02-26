# -*- coding: utf-8 -*-
"""

# Homework 4: Image Classification on CIFAR-10
  🛫 🚘 🐦 🐱 🦌 🐶 🐸 🐴 🚢 🛻

## Information before starting

In this problem, we will explore different deep learning architectures for image classification on the CIFAR-10 dataset. Make sure that you are familiar with torch `Tensor`s, two-dimensional convolutions (`nn.Conv2d`) and fully-connected layers (`nn.Linear`), ReLU non-linearities (`F.relu`), pooling (`nn.MaxPool2d`), and tensor reshaping (`view`). **Make sure to read through all instructions in both this notebook and in the PDF while completing this problem!**


### Problem Introduction

You've already had some practice using the PyTorch library in HW3, but this problem dives into training more complex deep learning models.

The specific task we are trying to solve in this problem is image classification. We're using a common dataset called CIFAR-10 which has 60,000 images separated into 10 classes:
* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck


### Enabling GPU

We are using Google Colab because it has free GPU runtimes available. GPUs can accelerate training times for this problem by 10-100x when compared to using CPU. To use the GPU runtime on Colab, make sure to **enable** the runtime by going to **Runtime -> Change runtime type -> Select T4 GPU under "Hardware accelerator"**.

Note that GPU runtimes are *limited* on Colab. We recommend limiting your training to short-running jobs (under 15 minutes each) and spread your work over time, if possible. Colab *will* limit your usage of GPU time, so plan ahead and be prepared to take breaks during training. If you have used up your quota for GPU, check back in a day or so to be able to enable GPU again.

Your code will still run on CPU, so if you are just starting to implement your code or have been GPU limited by Colab, you can still make changes and run your code - it will just be quite a bit slower. You can also choose to download your notebook and run locally if you have a personal GPU or have a faster CPU than the one Colab provides. If you choose to do this, you may need to install the packages that this notebook depends on to your `cse446` conda environment or to another Python environment of your choice.

To check if you have enabled GPU, run the following cell. If `device` is `cuda`, it means that GPU has been enabled successfully.

EXAMPLE:
"""


"""### Submitting your assignment

Once you are done with the problem, make sure to put all of your necessary figures into your PDF submission. Then, download this notebook as a Python file (`.py`) by going to **File -> Download -> Download `.py`**. Rename this file as `hw4-a3.py` and upload to the Gradescope submission for HW4 code.

"""


"""

## Your Turn!

The rest is yours to code. You are welcome to structure the code any way you would like.

We do advise making using code cells and functions (train, search, predict etc.) for each subproblem, since they will make your code easier to debug.

Also note that several of the functions above can be reused for the various different models you will implement for this problem; i.e., you won't need to write a new `evaluate()`. Before you reuse functions though, make sure they are compatible with what the assignment is asking for.

### Submitting Code

And as a last reminder, once you are done with the problem, make sure to put all of your necessary figures into your PDF submission. Then, download this notebook as a Python file (`.py`) by going to **File -> Download -> Download `.py`**. Rename this file as `hw4-a3.py` and upload to the Gradescope submission for HW4 code.
"""

# Commented out IPython magic to ensure Python compatibility.
import torch
from torch import nn
import numpy as np

from typing import Tuple, Union, List, Callable
from torch.optim import SGD
import torchvision
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# %matplotlib inline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)  # this should print out CUDA

train_dataset = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())



SAMPLE_DATA = False
# batch size
batch_size = 128

if SAMPLE_DATA:
  train_dataset, _ = random_split(train_dataset, [int(0.1 * len(train_dataset)), int(0.9 * len(train_dataset))]) # get 10% of train dataset and "throw away" the other 90%

train_dataset, val_dataset = random_split(train_dataset, [int(0.9 * len(train_dataset)), int( 0.1 * len(train_dataset))])

# Create separate dataloaders for the train, test, and validation set
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
)


def linear_model(d:int) -> nn.Module:
    """Instantiate a linear model and send it to device."""
    model =  nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, d),
            nn.ReLU(),
            nn.Linear(d, 10)

         )
    return model.to(DEVICE)


def convolution_model(filters:int, kernal:int, pool:int) -> nn.Module:
    # filters = number of filters/ output
    # kernal = kernal size
    # pool = maxpool size
    model =  nn.Sequential(
            nn.Conv2d(3, filters, kernel_size=kernal),
            nn.ReLU(),
            # Max pooling layer with
            nn.MaxPool2d(pool),
            nn.Flatten(),
            nn.Linear(filters * int(np.floor((33 - kernal)/pool)) ** 2, 10)
         )
    return model.to(DEVICE)



def train(
    model: nn.Module, optimizer: SGD,
    train_loader: DataLoader, val_loader: DataLoader,
    epochs: int = 20
    )-> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Trains a model for the specified number of epochs using the loaders.

    Returns:
    Lists of training loss, training accuracy, validation loss, validation accuracy for each epoch.
    """

    loss = nn.CrossEntropyLoss()
    train_losses = []
    train_accuracies = []
    val_losses = []
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, momentum = 0.9)
    val_accuracies = []
    for e in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        # Main training loop; iterate over train_loader. The loop
        # terminates when the train loader finishes iterating, which is one epoch.
        for (x_batch, labels) in train_loader:
            x_batch, labels = x_batch.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            labels_pred = model(x_batch)
            batch_loss = loss(labels_pred, labels)
            train_loss = train_loss + batch_loss.item()

            labels_pred_max = torch.argmax(labels_pred, 1)
            batch_acc = torch.sum(labels_pred_max == labels)
            train_acc = train_acc + batch_acc.item()

            batch_loss.backward()
            optimizer.step()
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc / (batch_size * len(train_loader)))
        print("train accuracy = ", train_acc / (batch_size * len(train_loader)))

        # Validation loop; use .no_grad() context manager to save memory.
        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for (v_batch, labels) in val_loader:
                v_batch, labels = v_batch.to(DEVICE), labels.to(DEVICE)
                labels_pred = model(v_batch)
                v_batch_loss = loss(labels_pred, labels)
                val_loss = val_loss + v_batch_loss.item()

                v_pred_max = torch.argmax(labels_pred, 1)
                batch_acc = torch.sum(v_pred_max == labels)
                val_acc = val_acc + batch_acc.item()
            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(val_acc / (batch_size * len(val_loader)))
            print("val accuracy = ", val_acc / (batch_size * len(val_loader)))
    #print("train accuracy = ", train_accuracies[11])
    #print("val accuracy = ", val_accuracies[11])
    torch.save(model, 'model_saved.pt')
    return train_losses, train_accuracies, val_losses, val_accuracies


def evaluate(
    model: nn.Module, loader: DataLoader
) -> Tuple[float, float]:
    """Computes test loss and accuracy of model on loader."""
    loss = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for (batch, labels) in loader:
            batch, labels = batch.to(DEVICE), labels.to(DEVICE)
            y_batch_pred = model(batch)
            batch_loss = loss(y_batch_pred, labels)
            test_loss = test_loss + batch_loss.item()

            pred_max = torch.argmax(y_batch_pred, 1)
            batch_acc = torch.sum(pred_max == labels)
            test_acc = test_acc + batch_acc.item()
        test_loss = test_loss / len(loader)
        test_acc = test_acc / (batch_size * len(loader))
        return test_loss, test_acc

def parameter_search_partA(train_loader: DataLoader,
                     val_loader: DataLoader,
                     model_fn:Callable[[], nn.Module]) -> None:
    """
    Parameter search for our linear model using SGD.

    Args:
    train_loader: the train dataloader.
    val_loader: the validation dataloader.
    model_fn: a function that, when called, returns a torch.nn.Module.

    Returns:
    The learning rate with the least validation loss.
    NOTE: you may need to modify this function to search over and return
     other parameters beyond learning rate.
    """
    print("here 1")
    num_iter = 40
    best_loss = torch.tensor(np.inf)
    lr = 0.001
    print("here 2")
    #lrs = torch.linspace(10 ** (-6), 10 ** (-1), num_iter)
    M = torch.arange(100, 1001)
    possible_lr = torch.tensor([10 ** (-3), 10 ** (-2)])

    #using random search

    for i in range(num_iter):
        print("\n\n\n")
        print("iter start: ", i)
        # Generate a random index
        random_index = torch.randint(0, len(M), (1,)).item()
        random_index_lr = torch.randint(0, len(possible_lr), (1,)).item()

        # Select the value at the random index
        random_value = M[random_index]
        random_value_lr = possible_lr[random_index_lr]

        print(f"trying M value {random_value}")
        print(f"trying lr value {random_value_lr}")
        model = model_fn(random_value)
        # 0.9 represents momentum
        optim = SGD(model.parameters(), random_value_lr, 0.9)
        #print("iter mid: ", i)
        train_loss, train_acc, val_loss, val_acc = train(
            model,
            optim,
            train_loader,
            val_loader,
            epochs=12
            )

        if min(val_loss) < best_loss:
            best_loss = min(val_loss)
        #print("iter end: ", i)



def parameter_search_partB(train_loader: DataLoader,
                     val_loader: DataLoader,
                     model_fn:Callable[[], nn.Module]) -> None:
    """
    Parameter search for our linear model using SGD.

    Args:
    train_loader: the train dataloader.
    val_loader: the validation dataloader.
    model_fn: a function that, when called, returns a torch.nn.Module.

    Returns:
    The learning rate with the least validation loss.
    NOTE: you may need to modify this function to search over and return
     other parameters beyond learning rate.
    """
    print("here 1")
    num_iter = 40
    best_loss = torch.tensor(np.inf)
    print("here 2")
    # output channels
    filters = torch.arange(50, 500)
    #kernal size
    k = torch.arange(3, 11)
    pool = torch.arange(5,20)
    # learning rate/step size
    possible_lr = torch.tensor([0.001, 0.01])

    #using random search

    for i in range(num_iter):
        print("\n\n\n")
        print("iter start: ", i)
        # Generate a random index
        random_index_filter = torch.randint(0, len(filters), (1,)).item()
        random_index_lr = torch.randint(0, len(possible_lr), (1,)).item()
        random_index_k = torch.randint(0, len(k), (1,)).item()
        random_index_pool = torch.randint(0, len(pool), (1,)).item()

        # Select the value at the random index
        random_filter = int(filters[random_index_filter])
        random_lr = float(possible_lr[random_index_lr])
        random_k = int(k[random_index_k])
        random_pool = int(pool[random_index_pool])

        print(f"trying filter value {random_filter}")
        print(f"trying lr value {random_lr}")
        print(f"trying kernal value {random_k}")
        print(f"trying pool value {random_pool}")
        model = model_fn(random_filter, random_k, random_pool)
        # 0.9 represents momentum
        optim = SGD(model.parameters(), random_lr, 0.9)
        #print("iter mid: ", i)
        train_loss, train_acc, val_loss, val_acc = train(
            model,
            optim,
            train_loader,
            val_loader,
            epochs=12
            )


        if min(val_loss) < best_loss:
            best_loss = min(val_loss)
        #print("iter end: ", i)





######## Part A ############
# uncomment for code
#best_m = parameter_search_partA(train_loader, val_loader, linear_model)
######## Part A ############

######## Part B ############
# uncomment for code
#parameter_search_partB(train_loader, val_loader, convolution_model)
######## Part B ############

# Part A
'''
epoch = 35
lr = 0.01
model = linear_model(604)
optim = SGD(model.parameters(), lr, 0.9)
train_loss, train_acc, val_loss, val_acc = train(
            model,
            optim,
            train_loader,
            val_loader,
            epochs=epoch
            )

print("604\n\n\n\n\n")
best_m_model_one = torch.load('model_saved.pt')
test_loss, test_acc = evaluate(best_m_model_one, test_loader)
print(f"Test Accuracy: {test_acc}")
plt.plot(np.arange(epoch), train_acc, label = "Training Accuracy M = 604, lr = 0.01", color = "blue")
plt.plot(np.arange(epoch), val_acc, label = "Validation Accuracy M = 604 lr = 0.01", color = "blue", linestyle=':')



model = linear_model(710)
optim = SGD(model.parameters(), lr, 0.9)
train_loss, train_acc, val_loss, val_acc = train(
            model,
            optim,
            train_loader,
            val_loader,
            epochs=epoch
            )
print("710\n\n\n\n\n")
best_m_model_two = torch.load('model_saved.pt')
test_loss, test_acc = evaluate(best_m_model_two, test_loader)
print(f"Test Accuracy: {test_acc}")
plt.plot(np.arange(epoch), train_acc, label = "Training Accuracy M = 710, lr = 0.01", color = "red")
plt.plot(np.arange(epoch), val_acc, label = "Validation Accuracy M = 710, lr = 0.01", color = "red", linestyle=':')

model = linear_model(783)
optim = SGD(model.parameters(), lr, 0.9)
train_loss, train_acc, val_loss, val_acc = train(
            model,
            optim,
            train_loader,
            val_loader,
            epochs=epoch
            )
print("861\n\n\n\n\n")
best_m_model_three = torch.load('model_saved.pt')
test_loss, test_acc = evaluate(best_m_model_three, test_loader)
print(f"Test Accuracy: {test_acc}")
plt.plot(np.arange(epoch), train_acc, label = "Training Accuracy M = 861, lr = 0.01", color = "green")
plt.plot(np.arange(epoch), val_acc, label = "Validation Accuracy M = 861, lr = 0.01", color = "green", linestyle=':')

line = np.repeat(0.50, epoch)
plt.plot(np.arange(epoch), line, label = "Threshold", color = "orange")

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Fully-connected output, 1 fully-connected hidden layer")
plt.show()
'''



# Part B

epoch = 35
model = convolution_model(170, 8, 5)
optim = SGD(model.parameters(), 0.009999999776482582, 0.9)
train_loss, train_acc, val_loss, val_acc = train(
            model,
            optim,
            train_loader,
            val_loader,
            epochs=epoch
            )

print("M = 170, kernal = 8, pool = 5\n\n\n\n\n")
part_b_model_one = torch.load('model_saved.pt')
test_loss, test_acc = evaluate(part_b_model_one, test_loader)
print(f"Test Accuracy: {test_acc}")
print(f"Test Loss: {test_loss}")
plt.plot(np.arange(epoch), train_acc, label = "Training Accuracy M = 170, k = 8, N = 5, lr = 0.009999999776482582, ", color = "blue")
plt.plot(np.arange(epoch), val_acc, label = "Validation Accuracy M = 170, k = 8, N = 5, lr = 0.009999999776482582", color = "blue", linestyle=':')



model = convolution_model(301, 4, 6)
optim = SGD(model.parameters(), 0.009999999776482582, 0.9)
train_loss, train_acc, val_loss, val_acc = train(
            model,
            optim,
            train_loader,
            val_loader,
            epochs=epoch
            )
print("M = 301, kernal = 4, pool = 6\n\n\n\n\n")
part_b_model_two = torch.load('model_saved.pt')
test_loss, test_acc = evaluate(part_b_model_two, test_loader)
print(f"Test Accuracy: {test_acc}")
print(f"Test Loss: {test_loss}")
plt.plot(np.arange(epoch), train_acc, label = "Training Accuracy M = 352, k = 7, N = 11, lr = 0.009999999776482582", color = "red")
plt.plot(np.arange(epoch), val_acc, label = "Validation Accuracy M = 352, k = 7, N = 11, lr = 0.009999999776482582", color = "red", linestyle=':')



model = convolution_model(352, 7, 11)
optim = SGD(model.parameters(), 0.009999999776482582, 0.9)
train_loss, train_acc, val_loss, val_acc = train(
            model,
            optim,
            train_loader,
            val_loader,
            epochs=epoch
            )
print("M = 352, kernal = 7, pool = 11\n\n\n\n\n")
part_b_model_three = torch.load('model_saved.pt')
test_loss, test_acc = evaluate(part_b_model_three, test_loader)
print(f"Test Accuracy: {test_acc}")
print(f"Test Loss: {test_loss}")
plt.plot(np.arange(epoch), train_acc, label = "Training Accuracy M = 861, lr = 0.01", color = "green")
plt.plot(np.arange(epoch), val_acc, label = "Validation Accuracy M = 861, lr = 0.01", color = "green", linestyle=':')

line = np.repeat(0.50, epoch)
plt.plot(np.arange(epoch), line, label = "Threshold", color = "orange")

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Convolutional layer with max-pool and fully-connected output: ")
plt.show()
