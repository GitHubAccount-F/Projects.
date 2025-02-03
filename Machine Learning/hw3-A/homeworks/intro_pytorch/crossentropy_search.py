if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from losses import CrossEntropyLossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from .optimizers import SGDOptimizer
    from .losses import CrossEntropyLossLayer
    from .train import plot_model_guesses, train

from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)


@problem.tag("hw3-A")
def crossentropy_parameter_search(
    dataset_train: TensorDataset, dataset_val: TensorDataset
) -> Dict[str, Any]:
    """
    Main subroutine of the CrossEntropy problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers
    NOTE: Each model should end with a Softmax layer due to CrossEntropyLossLayer requirement.

    Notes:
        - When choosing the number of epochs, consider effect of other hyperparameters on it.
            For example as learning rate gets smaller you will need more epochs to converge.

    Args:
        dataset_train (TensorDataset): Dataset for training.
        dataset_val (TensorDataset): Dataset for validation.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
            You are free to employ any structure of this dictionary, but we suggest the following:
            {
                name_of_model: {
                    "train": Per epoch losses of model on train set,
                    "val": Per epoch losses of model on validation set,
                    "model": Actual PyTorch model (type: nn.Module),
                }
            }
    """

    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=32, shuffle=True) 
    result = {'One':{'train':[], 'val':[], 'model':None},  
              'Two':{'train':[], 'val':[], 'model':None}, 
              'Three':{'train':[], 'val':[], 'model':None},
              'Four':{'train':[], 'val':[], 'model':None},
              'Five':{'train':[], 'val':[], 'model':None}}
    first_batch = next(iter(train_loader))
    # get input dimension
    input_dim = first_batch[0].shape[1] 
    # get output dimension
    output_dim = first_batch[0].shape[1] 
    hidden_dim = 2
    # number of epochs
    epochs = 32
    model_one = torch.nn.Sequential(
        LinearLayer(input_dim, output_dim, RNG),
        SoftmaxLayer()
        )
    train_one = train(train_loader, model_one, CrossEntropyLossLayer, SGDOptimizer, val_loader, epochs)
    result['One']['train'] = train_one['train']
    result['One']['val'] = train_one['val']
    result['One']['model'] = torch.load('model_saved.pt')

    # Second model
    model_two = torch.nn.Sequential(
        LinearLayer(input_dim, hidden_dim, RNG),
        SigmoidLayer(),
        LinearLayer(hidden_dim, output_dim, RNG),
        SoftmaxLayer()
    )
    train_two = train(train_loader, model_two, CrossEntropyLossLayer, SGDOptimizer, val_loader, epochs)
    result['Two']['train'] = train_two['train']
    result['Two']['val'] = train_two['val']
    result['Two']['model'] = torch.load('model_saved.pt')

    # Third model
    model_three = torch.nn.Sequential( #fix
        LinearLayer(input_dim, hidden_dim, RNG),
        ReLULayer(),
        LinearLayer(hidden_dim, output_dim, RNG),
        SoftmaxLayer()
    )
    #SGDOptimizer(params=model_three.parameters(), lr=0.001)
    train_three = train(train_loader, model_three, CrossEntropyLossLayer, SGDOptimizer, val_loader, epochs)
    result['Three']['train'] = train_three['train']
    result['Three']['val'] = train_three['val']
    result['Three']['model'] = torch.load('model_saved.pt')

    # Fourth model
    model_four = torch.nn.Sequential(
        LinearLayer(input_dim, hidden_dim, RNG),
        SigmoidLayer(),
        LinearLayer(hidden_dim, hidden_dim, RNG),
        ReLULayer(),
        LinearLayer(hidden_dim, output_dim, RNG),
        SoftmaxLayer()
    )
    train_four = train(train_loader, model_four, CrossEntropyLossLayer, SGDOptimizer, val_loader, epochs)
    result['Four']['train'] = train_four['train']
    result['Four']['val'] = train_four['val']
    result['Four']['model'] = torch.load('model_saved.pt')

    # Fifth model
    model_five = torch.nn.Sequential(
        LinearLayer(input_dim, hidden_dim, RNG),
        ReLULayer(),
        LinearLayer(hidden_dim, hidden_dim, RNG),
        SigmoidLayer(),
        LinearLayer(hidden_dim, output_dim, RNG),
        SoftmaxLayer()
    )
    train_five = train(train_loader, model_five, CrossEntropyLossLayer, SGDOptimizer, val_loader, epochs)
    result['Five']['train'] = train_five['train']
    result['Five']['val'] = train_five['val']
    result['Five']['model'] = torch.load('model_saved.pt')

    return result


@problem.tag("hw3-A")
def accuracy_score(model, dataloader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for CrossEntropy.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is an integer representing a correct class to a corresponding observation.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to MSE accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    total = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            # getting the correct class
            output = model(x)
            output = torch.max(output, 1)
            total += len(y)

            count += len(y) - torch.count_nonzero(y - output.indices)
            #count += (output.indices == torch.tensor(y)).sum().item()
    accuracy = count / total
    return accuracy


@problem.tag("hw3-A", start_line=7)
def main():
    """
    Main function of the Crossentropy problem.
    It should:
        1. Call crossentropy_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me Crossentropy loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    dataset_train = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y))
    dataset_val = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    dataset_test = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))

    test_data = DataLoader(dataset_test, batch_size=32, shuffle=True)

    epochs = 32
    ce_configs = crossentropy_parameter_search(dataset_train, dataset_val)

    # Find minimum out of all validation sets
    print(torch.min(torch.tensor(ce_configs['One']['val'])))
    print(torch.min(torch.tensor(ce_configs['Two']['val'])))
    print(torch.min(torch.tensor(ce_configs['Three']['val'])))
    print(torch.min(torch.tensor(ce_configs['Four']['val'])))
    print(torch.min(torch.tensor(ce_configs['Five']['val'])))

    plt.plot(np.arange(epochs), ce_configs['One']['train'], label = "Model one - Train", color = "blue")
    plt.plot(np.arange(epochs), ce_configs['Two']['train'], label = "Model two - Train", color = "red")
    plt.plot(np.arange(epochs), ce_configs['Three']['train'], label = "Model three - Train", color = "green")
    plt.plot(np.arange(epochs), ce_configs['Four']['train'], label = "Model four - Train", color = "orange")
    plt.plot(np.arange(epochs), ce_configs['Five']['train'], label = "Model five - Train", color = "purple")
    plt.plot(np.arange(epochs), ce_configs['One']['val'], label = "Model one - Validation", color = "cyan")
    plt.plot(np.arange(epochs), ce_configs['Two']['val'], label = "Model two - Validation", color = "pink")
    plt.plot(np.arange(epochs), ce_configs['Three']['val'], label = "Model three - Validation", color = "lime")
    plt.plot(np.arange(epochs), ce_configs['Four']['val'], label = "Model four - Validation", color = "black")
    plt.plot(np.arange(epochs), ce_configs['Five']['val'], label = "Model five - Validation", color = "peru")
    plt.legend()
    plt.xlabel("Epochs")
    plt.title('Cross Entropy Search Plot')
    plt.ylabel("Cross Entropy Loss")
    plt.show()

    # part c
    # best loss was found in model 3
    print("accuracy")
    print(accuracy_score(ce_configs['Five']['model'], test_data))
    

    plot_model_guesses(test_data, ce_configs['Five']['model'], 'Network with two hidden layers and ReLU, Sigmoid activation')


if __name__ == "__main__":
    main()
