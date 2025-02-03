"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple

import numpy as np

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=5)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        # Fill in with matrix with the correct shape
        self.weight: np.ndarray = None  # type: ignore
        self.mean: np.ndarray = None
        self.std: np.ndarray = None
        self.target: np.ndarray = None
        # You can add additional fields
        #raise NotImplementedError("Your Code Goes Here")

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """
        exp = np.arange(1, degree + 1)
        result = np.zeros((len(X), degree))
        for row in range(len(X)):
            temp = X[row]
            repeat = np.repeat(temp, degree) # Creates an array of just X_i
            list = np.power(repeat, exp)
            np.copyto(result[row],list)
        return result

    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You will need to apply polynomial expansion and data standardization first.
        """
        # Store target 
        self.target = y
        # Polyfeature() then standardize
        input = self.polyfeatures(X, self.degree)
        #stdList = [] # std for each col
        #meanList = [] # mean for each col
        #np.std(input, 0, None, stdList)
        #np.mean(input, 0, None, stdList)
        self.std = np.std(input, axis=0)
        self.mean = np.mean(input, axis=0)
        input = (input - self.mean) / self.std
        n = len(X)

        # add 1s column
        X_ = np.c_[np.ones([n, 1]), input]

        n, d = X_.shape
        # remove 1 for the extra column of ones we added to get the original num features
        d = d - 1

        # construct reg matrix
        reg_matrix = np.eye(d + 1) * self.reg_lambda
        reg_matrix[0, 0] = 0

        # (X'X + regMatrix)^-1 X' y
        self.weight = (np.matmul(np.linalg.inv(np.add(np.matmul(np.transpose(X_),X_), reg_matrix)),np.matmul(np.transpose(X_),y)))


    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        # Polyfeature() then standardize
        input = self.polyfeatures(X, self.degree)
        input = (input - self.mean) / self.std

        n = len(X)

        # add 1s column
        X_ = np.c_[np.ones([n, 1]), input]

        # predict
        return np.matmul(X_, self.weight)


@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    # find mean squared error for 'a'
    result: float = np.sum(np.power(np.subtract(a,b), 2)) / len(a)
    return result




@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)
    # Fill in errorTrain and errorTest arrays

    model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
    for i in range(1,n):
        # Create subsets of the original train data
        listX = np.array(Xtrain[:i + 1])
        listY = np.array(Ytrain[:i + 1])
        # create model and find mean square error
        model.fit(listX, listY)
        # Predict train data
        predictA = model.predict(listX)
        errorTrain[i] = mean_squared_error(predictA, listY)
        # Work on errorTrain
        # Predict test data based on training data weight
        predictB = model.predict(Xtest)
        errorTest[i] = mean_squared_error(predictB, Ytest)
    return (errorTrain, errorTest)

        


    
 
        
