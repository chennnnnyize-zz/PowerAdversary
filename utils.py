from numpy import shape
import numpy as np



def reorganize(X_train, Y_train, seq_length):
    # Organize the input and output to feed into RNN model
    x_data = []
    for i in range(len(X_train) - seq_length):
        x_new = X_train[i:i + seq_length]
        x_data.append(x_new)

    # Y_train
    y_data = Y_train[seq_length:]
    y_data = y_data.reshape((-1, 1))

    return x_data, y_data


def check_control_constraint(X, dim, uppper_bound, lower_bound):
    for i in range(0, shape(X)[0]):
        for j in range(0, shape(X)[0]):
            for k in range(0, dim):
                if X[i, j, k] >= uppper_bound[i, j, k]:
                    X[i, j, k] = uppper_bound[i, j, k] - 0.01
                if X[i, j, k] <= lower_bound[i, j, k]:
                    X[i, j, k] = lower_bound[i, j, k] + 0.01
    return X