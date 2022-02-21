import numpy as np 
import random


def mixup(X_train, Y_train, class_n,
                 ratio=0.5,
                 alpha=1.0,
                 beta=1.0):
    new_data_n = int(X_train.shape[0] * ratio)
    new_X = []
    new_Y = []
    count = 0
    idxset = set(range(X_train.shape[0]))
    while count < new_data_n:
        idx1, idx2 = random.sample(idxset, 2)
        l = np.random.beta(alpha, beta)
        new_x = l * X_train[idx1] + (1-l) * X_train[idx2]
        new_y = l * Y_train[idx1] + (1-l) * Y_train[idx2]
        new_X.append(new_x)
        new_Y.append(new_y)
        count += 1

    X_train = np.concatenate((X_train, np.array(new_X)))
    Y_train = np.concatenate((Y_train, np.array(new_Y)))
    print(X_train.shape, Y_train.shape)
    return X_train, Y_train
        