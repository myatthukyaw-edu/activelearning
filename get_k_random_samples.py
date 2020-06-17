from sklearn.utils import check_random_state
import numpy as np

def get_k_random_samples(trainset_size, initial_labeled_samples, X_train_full, y_train_full):
    random_state = check_random_state(0)
    permutation = np.random.choice(trainset_size, initial_labeled_samples, replace=False)

    print ('Initial random chosen samples', permutation.shape)
    X_train = X_train_full[permutation]
    y_train = y_train_full[permutation]
    #X_train = X_train.reshape((X_train.shape[0], -1))
    y_train_bin = y_train.reshape(initial_labeled_samples,)
    bin_count = np.bincount(y_train_bin.astype('int64'))
    unique = np.unique(y_train)
    print ( 'initial train set:', X_train.shape, y_train.shape,
        '\nlabels count:', bin_count, unique,
        )
    return (permutation, X_train, y_train)