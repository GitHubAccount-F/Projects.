if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer
    from losses import MSELossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer
    from .optimizers import SGDOptimizer
    from .losses import MSELossLayer
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
def accuracy_score(model: nn.Module, dataloader: DataLoader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for MSE.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is also a 2-d vector of floats, but specifically with one being 1.0, while other is 0.0.
            Index of 1.0 in target corresponds to the true class.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to CrossEntropy accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    total = 0
    count = 0
    # come back to
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            # getting the correct class
            y = torch.argmax(y, dim=1)
            total += len(y)

            y_pred = torch.argmax(model(x), dim=1)

            count += len(y) - torch.count_nonzero(y - y_pred)
    accuracy = count / total
    return accuracy
        




@problem.tag("hw3-A")
def mse_parameter_search(
    dataset_train: TensorDataset, dataset_val: TensorDataset
) -> Dict[str, Any]:
    """
    Main subroutine of the MSE problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers

    Notes:
        - When choosing the number of epochs, consider effect of other hyperparameters on it.
            For example as learning rate gets smaller you will need more epochs to converge.

    Args:
        dataset_train (TensorDataset): Training dataset.
        dataset_val (TensorDataset): Validation dataset.

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
    output_dim = first_batch[1].shape[1] 
    hidden_dim = 2
    # number of epochs
    epochs = 32
    model_one = torch.nn.Sequential(LinearLayer(input_dim, output_dim, RNG))
    train_one = train(train_loader, model_one, MSELossLayer, SGDOptimizer, val_loader, epochs=epochs)
    result['One']['train'] = train_one['train']
    result['One']['val'] = train_one['val']
    result['One']['model'] = torch.load('model_saved.pt')

    # Second model
    model_two = torch.nn.Sequential(
        LinearLayer(input_dim, hidden_dim, RNG),
        SigmoidLayer(),
        LinearLayer(hidden_dim, output_dim, RNG)
    )
    train_two = train(train_loader, model_two, MSELossLayer, SGDOptimizer, val_loader, epochs=epochs)
    result['Two']['train'] = train_two['train']
    result['Two']['val'] = train_two['val']
    result['Two']['model'] = torch.load('model_saved.pt')

    # Third model
    model_three = torch.nn.Sequential( #fix
        LinearLayer(input_dim, hidden_dim, RNG),
        ReLULayer(),
        LinearLayer(hidden_dim, output_dim, RNG)
    )
    #SGDOptimizer(params=model_three.parameters(), lr=0.001)
    train_three = train(train_loader, model_three, MSELossLayer, SGDOptimizer, val_loader, epochs=epochs)
    result['Three']['train'] = train_three['train']
    result['Three']['val'] = train_three['val']
    result['Three']['model'] = torch.load('model_saved.pt')

    # Fourth model
    model_four = torch.nn.Sequential(
        LinearLayer(input_dim, hidden_dim, RNG),
        SigmoidLayer(),
        LinearLayer(hidden_dim, hidden_dim, RNG),
        ReLULayer(),
        LinearLayer(hidden_dim, output_dim, RNG)
    )
    train_four = train(train_loader, model_four, MSELossLayer, SGDOptimizer, val_loader, epochs=epochs)
    result['Four']['train'] = train_four['train']
    result['Four']['val'] = train_four['val']
    result['Four']['model'] = torch.load('model_saved.pt')

    # Fifth model
    model_five = torch.nn.Sequential(
        LinearLayer(input_dim, hidden_dim, RNG),
        ReLULayer(),
        LinearLayer(hidden_dim, hidden_dim, RNG),
        SigmoidLayer(),
        LinearLayer(hidden_dim, output_dim, RNG)
    )
    train_five = train(train_loader, model_five, MSELossLayer, SGDOptimizer, val_loader, epochs=epochs)
    result['Five']['train'] = train_five['train']
    result['Five']['val'] = train_five['val']
    result['Five']['model'] = torch.load('model_saved.pt')

    return result


    
    


@problem.tag("hw3-A", start_line=11)
def main():
    """
    Main function of the MSE problem.
    It should:
        1. Call mse_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me MSE loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    dataset_train = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(to_one_hot(y)))
    dataset_val = TensorDataset(
        torch.from_numpy(x_val).float(), torch.from_numpy(to_one_hot(y_val))
    )
    dataset_test = TensorDataset(
        torch.from_numpy(x_test).float(), torch.from_numpy(to_one_hot(y_test))
    )

    # building data loader
    test_data = DataLoader(dataset_test, batch_size=32, shuffle=True)
    

    epochs = 32
    mse_configs = mse_parameter_search(dataset_train, dataset_val)
    # Find minimum out of all validation sets
    print(torch.min(torch.tensor(mse_configs['One']['val'])))
    print(torch.min(torch.tensor(mse_configs['Two']['val'])))
    print(torch.min(torch.tensor(mse_configs['Three']['val'])))
    print(torch.min(torch.tensor(mse_configs['Four']['val'])))
    print(torch.min(torch.tensor(mse_configs['Five']['val'])))

    plt.plot(np.arange(epochs), mse_configs['One']['train'], label = "Model one - Train", color = "blue")
    plt.plot(np.arange(epochs), mse_configs['Two']['train'], label = "Model two - Train", color = "red")
    plt.plot(np.arange(epochs), mse_configs['Three']['train'], label = "Model three - Train", color = "green")
    plt.plot(np.arange(epochs), mse_configs['Four']['train'], label = "Model four - Train", color = "orange")
    plt.plot(np.arange(epochs), mse_configs['Five']['train'], label = "Model five - Train", color = "purple")
    plt.plot(np.arange(epochs), mse_configs['One']['val'], label = "Model one - Validation", color = "cyan")
    plt.plot(np.arange(epochs), mse_configs['Two']['val'], label = "Model two - Validation", color = "pink")
    plt.plot(np.arange(epochs), mse_configs['Three']['val'], label = "Model three - Validation", color = "lime")
    plt.plot(np.arange(epochs), mse_configs['Four']['val'], label = "Model four - Validation", color = "black")
    plt.plot(np.arange(epochs), mse_configs['Five']['val'], label = "Model five - Validation", color = "peru")
    plt.legend()
    plt.xlabel("Epochs")
    plt.title('MSE Search Plot')
    plt.ylabel("MSE Loss")
    plt.show()

    # part c
    # best loss was found in model 5
    print("accuracy")
    print(accuracy_score(mse_configs['Five']['model'], test_data))

    plot_model_guesses(test_data, mse_configs['Five']['model'], 'Model - 5: Network with Two Hidden Layers, Relu then Sigmoid')
    









def to_one_hot(a: np.ndarray) -> np.ndarray:
    """Helper function. Converts data from categorical to one-hot encoded.

    Args:
        a (np.ndarray): Input array of integers with shape (n,).

    Returns:
        np.ndarray: Array with shape (n, c), where c is maximal element of a.
            Each element of a, has a corresponding one-hot encoded vector of length c.
    """
    r = np.zeros((len(a), 2))
    r[np.arange(len(a)), a] = 1
    return r


if __name__ == "__main__":
    main()
