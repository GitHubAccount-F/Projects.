from typing import List, Tuple

import numpy as np

from utils import problem


@problem.tag("hw4-A")
def calculate_centers(
    data: np.ndarray, classifications: np.ndarray, num_centers: int
) -> np.ndarray:
    """
    Sub-routine of Lloyd's algorithm that calculates the centers given datapoints and their respective classifications/assignments.
    num_centers is additionally provided for speed-up purposes.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        classifications (np.ndarray): Array of shape (n,) full of integers in range {0, 1, ...,  num_centers - 1}.
            Data point at index i is assigned to classifications[i].
        num_centers (int): Number of centers for reference.
            Might be usefull for pre-allocating numpy array (Faster that appending to list).

    Returns:
        np.ndarray: Array of shape (num_centers, d) containing new centers.
    """
    #count = np.zeros(num_centers)
    #sum = np.zeros(num_centers)
    # output matrix
    output = np.zeros((num_centers, len(data[0])))
    for i in range(num_centers):
        # get indexes
        indexes = np.where(classifications == i)[0]
        #print(indexes)
        # get total count
        count = len(indexes) 
        if (count == 0):
            count = 1
            print("count == 0")
            #output[i] = np.zeros(len(data[0]))
            #continue
            
        # get sum
        elements = np.sum(data[indexes], axis=0)
        #print(elements)
        #output[i] = np.vstack((output, elements / count))
        output[i] = elements / count
    return output
    

    
    


@problem.tag("hw4-A")
def cluster_data(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Sub-routine of Lloyd's algorithm that clusters datapoints to centers given datapoints and centers.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        centers (np.ndarray): Array of shape (k, d). Each row is a center to which a datapoint can be clustered.

    Returns:
        np.ndarray: Array of integers of shape (n,), with each entry being in range {0, 1, 2, ..., k - 1}.
            Entry j at index i should mean that j^th center is the closest to data[i] datapoint.
    """
    
    result = np.zeros(len(data))
    prev = np.linalg.norm(data - centers[0], 2, axis = 1) ** 2
    for i in range(1, len(centers)):
        output = np.linalg.norm(data - centers[i], 2, axis = 1) ** 2
        indexes = np.where(output < prev)[0]
        result[indexes] = np.int(i)
        prev[indexes] = output[indexes]
    return result
    '''
    norms = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
    
    # Assign each data point to the closest center
    centers = np.argmin(norms, axis=1)
    
    return centers
    '''


    


def calculate_error(data: np.ndarray, centers: np.ndarray) -> float:
    """ This method has been implemented for you.
    
    Calculates error/objective function on a provided dataset, with trained centers.

    Args:
        data (np.ndarray): Array of shape (n, d). Dataset to evaluate centers on.
        centers (np.ndarray): Array of shape (k, d). Each row is a center to which a datapoint can be clustered.
            These should be trained on training dataset.

    Returns:
        float: Single value representing mean objective function of centers on a provided dataset.
    """
    distances = np.zeros((data.shape[0], centers.shape[0]))
    for idx, center in enumerate(centers):
        distances[:, idx] = np.sqrt(np.sum((data - center) ** 2, axis=1))
    return np.mean(np.min(distances, axis=1))


@problem.tag("hw4-A")
def lloyd_algorithm(
    data: np.ndarray, num_centers: int, epsilon: float = 10e-3
) -> Tuple[np.ndarray, List[float]]:
    """Main part of Lloyd's Algorithm.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        num_centers (int): Number of centers to train/cluster around.
        epsilon (float, optional): Epsilon for stopping condition.
            Training should stop when max(abs(centers - previous_centers)) is smaller or equal to epsilon.
            Defaults to 10e-3.

    Returns:
        np.ndarray: Tuple of 2 numpy arrays:
            Element at index 0: Array of shape (num_centers, d) containing trained centers.
            Element at index 1: List of floats of length # of iterations
                containing errors at the end of each iteration of lloyd's algorithm.
                You should use the calculate_error() function that has been implemented for you.

    Note:
        - For initializing centers please use the first `num_centers` data points.
    """
    error = np.array([])
    test = 1
    centers = data[:num_centers]
    error = np.append(error, calculate_error(data, centers))
    while (test):
        cluster = cluster_data(data, centers)
        new_centers = calculate_centers(data, cluster, num_centers)
        error = np.append(error, calculate_error(data, new_centers))
        #print(len(error))
        #print(np.max(np.abs(new_centers - centers)))
        if (np.max(np.abs(new_centers - centers)) <= epsilon):
            test = 0
        centers = new_centers
    return (centers, error)
        


    
