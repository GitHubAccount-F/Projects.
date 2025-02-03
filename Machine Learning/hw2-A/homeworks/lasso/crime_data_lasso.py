if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")
    y_train = df_train['ViolentCrimesPerPop'].values # first column stores y
    y_test = df_test['ViolentCrimesPerPop'].values # first column stores y
    X_train = df_train.drop('ViolentCrimesPerPop', axis = 1).values
    X_test = df_test.drop('ViolentCrimesPerPop', axis = 1).values

    #avg y
    avgY = np.average(y_train)
    lambda_max = np.max(2 * np.abs(np.matmul(np.transpose(X_train), (y_train - avgY))))


    #initialize weight to 0
    weight = np.zeros(len(X_train))

    # Create space to store values
    lambdasStore = np.array([])
    countStore = np.array([])
    agePct12t29 = np.array([])
    pctWSocSec = np.array([])
    pctUrban = np.array([])
    agePct65up = np.array([])
    householdsize = np.array([])
    mse_train = np.array([])
    mse_test = np.array([])
    
    weight, bias = train(X_train, y_train, _lambda = lambda_max)
    lambdasStore = np.append(lambdasStore, lambda_max)
    countStore = np.append(countStore, np.count_nonzero(weight))

    agePct12t29_index = df_train.columns.get_loc('agePct12t29') - 1
    pctWSocSec_index = df_train.columns.get_loc('pctWSocSec') - 1
    pctUrban_index = df_train.columns.get_loc('pctUrban') - 1
    agePct65up_index = df_train.columns.get_loc('agePct65up') - 1
    householdsize_index = df_train.columns.get_loc('householdsize') - 1


    #part D
    agePct12t29 = np.append(agePct12t29, weight[agePct12t29_index])
    pctWSocSec = np.append(pctWSocSec, weight[pctWSocSec_index])
    pctUrban = np.append(pctUrban, weight[pctUrban_index])
    agePct65up = np.append(agePct65up, weight[agePct65up_index])
    householdsize = np.append(householdsize, weight[householdsize_index])

    #part e
    mse_train = np.append(mse_train, np.sum((np.matmul(X_train, weight) - y_train) ** 2) / len(X_train))
    mse_test = np.append(mse_test, np.sum((np.matmul(X_test, weight) - y_test) ** 2) / len(X_test))


    
    lambda_ = lambda_max / 2
    while (lambda_ >= 0.01):
        #print(lambda_)
        weight, bias = train(X_train, y_train, _lambda = lambda_, start_weight = weight, start_bias = bias)
        lambdasStore = np.append(lambdasStore, lambda_)
        countStore = np.append(countStore, np.count_nonzero(weight))

        #part d
        agePct12t29 = np.append(agePct12t29, weight[agePct12t29_index])
        pctWSocSec = np.append(pctWSocSec, weight[pctWSocSec_index])
        pctUrban = np.append(pctUrban, weight[pctUrban_index])
        agePct65up = np.append(agePct65up, weight[agePct65up_index])
        householdsize = np.append(householdsize, weight[householdsize_index])
     
        lambda_ = lambda_ / 2

        #part e
        mse_train = np.append(mse_train, np.sum((np.matmul(X_train, weight) - y_train) ** 2) / len(X_train))
        mse_test = np.append(mse_test, np.sum((np.matmul(X_test, weight) - y_test) ** 2) / len(X_test))


    
    # part C
    plt.plot(lambdasStore, countStore)
    plt.xlabel('lambda:')
    plt.ylabel('# of nonzero weights:')
    plt.xscale('log')
    plt.grid('on')
    plt.show()

    #Part D
    plt.plot(lambdasStore, agePct12t29, label='agePct12t29')  # Plot the first line
    plt.plot(lambdasStore, pctWSocSec, label='pctWSocSec')  # Plot the second line 
    plt.plot(lambdasStore, pctUrban, label='pctUrban')  # Plot the 3rd line
    plt.plot(lambdasStore, agePct65up, label='agePct65up')  # Plot the 4th line 
    plt.plot(lambdasStore, householdsize, label='householdsize')  # Plot the 5th line
    plt.xlabel('lambda:')
    plt.ylabel('coefficients:')
    plt.grid('on')
    plt.xscale('log')
    plt.legend()  # Display legend with labels for each line
    plt.show()
    
    #Part e
    plt.plot(lambdasStore, mse_train, label='train error')  # Plot the first line
    plt.plot(lambdasStore, mse_test, label='test error')  # Plot the second line 
    plt.xlabel('lambda:')
    plt.ylabel('Error:')
    plt.grid('on')
    plt.xscale('log')
    plt.legend()  # Display legend with labels for each line
    plt.show()


    #part f
    lambda_ = 30
    weight, bias = train(X_train, y_train, _lambda = lambda_)
    maxValue = np.max(weight)
    maxIndex = np.argmax(weight)
    minValue = np.min(weight)
    minIndex = np.argmin(weight)
    # comment out
    #print(maxValue)
    #print(df_test.columns[maxIndex + 1])
    #print(minValue)
    #print(df_test.columns[minIndex + 1])


    




if __name__ == "__main__":
    main()
