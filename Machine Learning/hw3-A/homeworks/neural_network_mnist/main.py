# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        a_zero = 1 / math.sqrt(d) # for weight/bias 0
        a_one = 1 / math.sqrt(h) # for weight/bias 1
        uniform_zero = Uniform(-a_zero, a_zero)
        uniform_one = Uniform(-a_one, a_one)

        # initialize weights
        self.weight_zero = Parameter(uniform_zero.sample((h,d)))
        self.weight_one = Parameter(uniform_one.sample((k,h)))

        # initialize bias
        self.bias_zero = Parameter(uniform_zero.sample((h, 1)))
        self.bias_one = Parameter(uniform_one.sample((k, 1)))




        
    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        '''
        output = self.linear_0(x)
        output = relu(output)
        output = self.linear_1(output)

        return self.weight_one @ (relu(self.weight_zero @ x + self.bias_zero)) + self.bias_one
        '''
        output = self.weight_zero @ x.t() + self.bias_zero # h x n 
        output = relu(output) # h x n
        output = self.weight_one @ output + self.bias_one
        return output.t() # n x k
        



class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """

        # fix 
        super().__init__()
        a_zero = 1 / math.sqrt(d)
        a_one = 1 / math.sqrt(h0)
        a_two = 1 / math.sqrt(h1)
        uniform_zero = Uniform(-a_zero, a_zero)
        uniform_one = Uniform(-a_one, a_one)
        uniform_two = Uniform(-a_two, a_two)

        #initialize weights
        self.weight_zero = Parameter(uniform_zero.sample((h0,d)))
        self.weight_one = Parameter(uniform_one.sample((h1,h0)))
        self.weight_two = Parameter(uniform_two.sample((k,h1)))


        #initialize bias
        self.bias_zero = Parameter(uniform_zero.sample((h0, 1)))
        self.bias_one = Parameter(uniform_one.sample((h1, 1)))
        self.bias_two = Parameter(uniform_one.sample((k, 1)))


    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        output = self.weight_zero @ x.t() + self.bias_zero # h x n 
        output = relu(output) # h x n
        output = self.weight_one @ output + self.bias_one
        output = relu(output)
        output = self.weight_two @ output + self.bias_two
        return output.t() # n x k    


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.
    
    avg training losses?

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    optimizer = Adam(model.parameters(), lr=0.001)
    result = []
    max_epoch = 100
    for epoch in range(max_epoch):
        print(epoch)
        total = 0
        count = 0
        total_loss = 0
        for x, y in train_loader:
            
            output = model(x)
            #print(output)
            #print(len(y))
            #print(y)
            optimizer.zero_grad()

            loss = cross_entropy(output, y)            

            total_loss += loss
            
            loss.backward()
            optimizer.step()

            # calculate accuracy
            argmax_indices = torch.argmax(output, dim=1)
            #print(argmax_indices)
            #print(y)
            total += len(y)
            #count += torch.sum(torch.all(argmax_indices == y, dim=0)).item()
            #count += torch.sum((argmax_indices == y)).item()
            count += torch.sum(torch.argmax(output, dim=1) == y).item()
            #print(count)
        result.append((total_loss.data / len(train_loader)).item())
        accuracy = count / total
        if accuracy >= 0.99:
            break
    return result






@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    dataset_train = TensorDataset(x, y)

    # Create a DataLoader
    dataloader = DataLoader(dataset_train, batch_size=32, shuffle=True)


    model_one = F1(64, 784, 10)
    result_one = train(model_one, Adam, dataloader)
    plt.plot([i for i in range(len(result_one))], result_one)
    plt.xlabel("Epochs")
    plt.title('F1')
    plt.ylabel("Loss")
    plt.show()
        
    # perform on test data
    output_one = model_one(x_test)
    # Calculate loss
    test_loss = cross_entropy(output_one, y_test)
    print("Model One Loss:")
    print(test_loss)

    #calculate accuracy
    accuracy = torch.sum(torch.argmax(output_one, dim=1) == y_test).item() / len(y_test)
    print("Model One Accuracy:")
    print(accuracy)
    print("Parameters in model:")
    print(sum(p.numel() for p in model_one.parameters()))
    
    


    model_two = F2(32, 32, 784, 10)
    result_two = train(model_two, Adam, dataloader)
    plt.plot([i for i in range(len(result_two))], result_two)
    plt.xlabel("Epochs")
    plt.title('F2')
    plt.ylabel("Loss")
    plt.show()

    # perform on test data
    output_two = model_two(x_test)
    # Calculate loss
    test_loss = cross_entropy(output_two, y_test)
    print("Model One Loss:")
    print(test_loss)

    #calculate accuracy
    accuracy = torch.sum(torch.argmax(output_two, dim=1) == y_test).item() / len(y_test)
    print("Model One Accuracy:")
    print(accuracy)

    print("Parameters in model:")
    print(sum(p.numel() for p in model_two.parameters()))
    



if __name__ == "__main__":
    main()
