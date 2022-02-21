import random


def random_sampling(X, Y, ratio):
    total = X.shape[0]
    n = int(total * ratio)
    random_idx_list = random.sample(list(range(total)), n)
    x = X[random_idx_list]
    y = Y[random_idx_list]
    return x, y


def random_split(X, Y, ratio):
    total = X.shape[0]
    n = int(total * ratio)

    idx_list = list(range(total))
    random.shuffle(idx_list)

    X_shuffled = X[idx_list]
    Y_shuffled = Y[idx_list]

    return X_shuffled[:n], Y_shuffled[:n], X_shuffled[n:], Y_shuffled[n:]


def random_split_list_inputs(X_list, Y, ratio):
    total = X_list[0].shape[0]
    n = int(total * ratio)

    idx_list = list(range(total))
    random.shuffle(idx_list)

    X_train = [X[idx_list][:n] for X in X_list]
    X_test = [X[idx_list][n:] for X in X_list]

    Y_shuffled = Y[idx_list]

    return X_train, Y_shuffled[:n], X_test, Y_shuffled[n:]