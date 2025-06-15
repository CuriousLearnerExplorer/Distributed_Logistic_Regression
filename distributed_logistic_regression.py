import numpy as np

# training data, import as in:
# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
from torchvision import datasets

if __name__ == '__main__':
    # Data Import
    data = datasets.MNIST('../data', train = True, download = True)
    data_test = datasets.MNIST('../data', train = False, download = True)
    
    # Preprocessing: Retain only 0-1-data
    # Inspired by approach in:
    # https://stackoverflow.com/questions/75034387/remove-digit-from-mnist-pytorch
    indices = (data.targets == 0) | (data.targets == 1) 
    indices_test = (data_test.targets == 0) | (data_test.targets == 1)

    # Training data
    data.data = data.data[indices]
    data.targets = data.targets[indices]

    # Test data
    data_test.data = data_test.data[indices_test]
    data_test.targets = data_test.targets[indices_test]
    
    # N Samples
    n_samples = 5000
    data.data = data.data[0:n_samples]
    data.targets = data.targets[0:n_samples]

    # For numerical reasons scale the pixel values to be in [0,1]
    data.data = data.data / 255
    # also scale test data
    data_test.data = data_test.data / 255

    # Features in numpy matrix format, each row is a data entry with 28x28 pixels
    dim = 28*28

    features_train = data.data.numpy().reshape((n_samples, dim))

    features_test = data_test.data.numpy()
    features_test = features_test.reshape((features_test.shape[0], dim))

    # Add 1-column to feature to include intercept into the model
    features_train = np.insert(features_train, dim, 1, axis = 1)
    features_test = np.insert(features_test, dim, 1, axis = 1)

    # Labels
    labels_train = data.targets.numpy()
    labels_test = data_test.targets.numpy()
    
    # w_0
    w = np.zeros(dim+1)
    # mu_0: estimate for Lagrange multiplier
    mu_0 = np.zeros(dim+1)