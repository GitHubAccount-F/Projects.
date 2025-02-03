from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

from utils import problem


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float, eta: float
) -> Tuple[np.ndarray, float]:
    """Single step in ISTA algorithm.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        bias (float): Bias returned from the step before.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.
        eta (float): Step-size. Determines how far the ISTA iteration moves for each step.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
    
    """

    biasSummation = np.sum(((np.matmul(X,weight) + bias) - y))
    biasTemp = bias - (2 * eta * biasSummation) # new bias value

    # weight
    weightSummation = np.zeros(weight.shape)
    # gives an n x 1
    summationInner = (np.matmul(X,weight) - y) + bias
    # should give a d x 1 matrix
    summationOuter = np.matmul(np.transpose(X), summationInner)



    weightSummation = weight - 2 * eta * summationOuter  

    
    # update weight vector
    conditions = [weightSummation < (-2 * eta * _lambda), 
                 weightSummation > (2 * eta * _lambda), 
                np.logical_and(-2 * eta * _lambda <= weightSummation, weightSummation <= 2 * eta * _lambda)]
    functions =  [lambda x: x + 2*eta*_lambda, 
                lambda x: x - 2*eta*_lambda,
                lambda x: 0]
    weightSummation = np.piecewise(weightSummation, conditions, functions)

    return (weightSummation, biasTemp)


   


@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized SSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    sseLoss = np.sum(((np.matmul(X, weight) - y) + bias) ** 2)
    lambdaSum = _lambda * (np.sum(np.abs(weight)))

    return sseLoss + lambdaSum





@problem.tag("hw2-A", start_line=5)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    eta: float = 0.00001,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
    start_bias: float = None
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        eta (float): Step size.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.
        start_bias (float, optional): Bias for hot-starting model.
            If None, defaults to zero. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You will also have to keep an old copy of bias for convergence criterion function.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
        start_bias = 0
    old_w: Optional[np.ndarray] = np.copy(start_weight)
    old_b: float = np.copy(start_bias)
    newWeight, newBias = step(X, y, start_weight, start_bias, _lambda, eta)
    while (not convergence_criterion(newWeight, old_w, newBias, old_b, convergence_delta)):
        old_w: Optional[np.ndarray] = np.copy(newWeight)
        old_b: float = np.copy(newBias)
        newWeight, newBias = step(X, y, newWeight, newBias, _lambda, eta)
    
    return (newWeight, newBias)



   


@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, bias: float, old_b: float, convergence_delta: float
) -> bool:
    """Function determining whether weight and bias has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compare it to convergence delta.
    It should also calculate the maximum absolute change between the bias and old_b, and compare it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of gradient descent.
        old_w (np.ndarray): Weight from previous iteration of gradient descent.
        bias (float): Bias from current iteration of gradient descent.
        old_b (float): Bias from previous iteration of gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight and bias has not converged yet. True otherwise.
    """
    maxWeight = np.max(np.abs(np.subtract(weight, old_w)))
    maxBias =  np.max(abs(bias - old_b))

    return (maxWeight < convergence_delta and maxBias < convergence_delta)
    
    


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    mu, sigma = 0, 0.1 # mean and standard deviation
    gaussian = np.random.normal(mu, sigma, 500)
    # create weight
    # up to k = 100
    weight = np.zeros(1000)
    for i in range(100):
        weight[i] = i/100
    # create samples of x
    X = np.random.randn(500, 1000)
    # this gives a 500 x 1000 matrix
    # standardize 
    mean_standard = np.mean(X, axis = 0)
    std_dev = np.std(X, axis = 0)

    X_standardized = (X - mean_standard) / std_dev

    # generate the y vector

    y = np.matmul(X_standardized[:,:100], weight[:100]) + gaussian
    # start with lambda max
    avgY = np.average(y)
    lambda_max = np.max(2 * np.abs(np.matmul(np.transpose(X_standardized), (y - avgY))))

    

    lambda_ = lambda_max
    #print(lambda_)
    # first column represents lambda, second column represents number of features
    lambdasStore = np.array([])
    countStore = np.array([])
    count = 0
    FDR = np.array([])
    TPR = np.array([])
    while(count < 1000):
        #print(count)
        new_weight, new_bias = train(X_standardized, y, _lambda = lambda_)
        count = np.count_nonzero(new_weight)
        countStore = np.append(countStore, count)
        lambdasStore = np.append(lambdasStore, lambda_)
        lambda_ *= 0.5
        if (count == 0):
            FDR = np.append(FDR, 0)
            TPR = np.append(TPR, 0)
        else:
            FDR = np.append(FDR, np.count_nonzero(new_weight[100:])/count)
            TPR = np.append(TPR, np.count_nonzero(new_weight[:100])/100)

    



    #print("here")
    # create plot
    plt.plot(lambdasStore, countStore)
    plt.grid('on')
    plt.xlabel('lambda:')
    plt.ylabel('Featuers that are nonzero:')
    plt.xscale('log')
    plt.show()


    plt.plot(FDR, TPR)
    plt.xlabel('FDR:')
    plt.ylabel('TPR:')
    plt.grid('on')
    plt.show()
    


if __name__ == "__main__":
    main()

