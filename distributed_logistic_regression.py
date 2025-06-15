import numpy as np

# training data, import as in:
# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
from torchvision import datasets

# from multiprocessing import Pool, cpu_count # for multiprocessing
import time # to measure duration of code

# Predict 0/1 for a given model parameter w, as done in classification through logistic regression
def predict(w, features):
    """
    w: parameter vector of logistic classifier
    features: the features of the data
    """

    # vectorized evaluation of model at features
    t = sigmoid_vec(features @ w)

    # prediction is 1 if our model gives > 0.5
    t[t > 0.5] = 1
    # else it is 0
    t[t <= 0.5] = 0

    return t

# Objective and augmented Lagrangian
# Define logistic function
def sigmoid(x):
    """
    x: input to the sigmoid function
    """
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        t = np.exp(x)
        return t/(t+1)

# Vectorize
sigmoid_vec = np.vectorize(sigmoid)
    
# Define cost function for logistic regression (weighted)
def loss(w, features, labels):
    """
    features: Array of x_i (features of data)
    labels: y_i in {0,1}
    w: current iterate
    """
    sigmoids = sigmoid_vec(features @ w)
    # print(features @ w)
    # print(labels)
    # Cross-entropy loss
    value = np.sum(labels*np.log(sigmoids) + (1 - labels)*np.log(1 - sigmoids))
    
    return - value / features.shape[0]

# Define the consensus constraints
def constraint(n_nodes, amount, beta, w, mu):
    """
    n_nodes: number of nodes
    amount: Amount of data points each node can access
    beta: hyperparamter of augmented Lagrangian
    w: current iterate
    mu: estimate of Lagrange multiplier
    """
    residual = 0

    # Circular graph of communication

    # For each node, add extra term of augmented Lagrangian
    for i in range(n_nodes-1):
        cons = w[i*amount:(i+1)*amount] - w[(i+1)*amount:(i+2)*amount]
        residual += np.dot(mu[i*amount:(i+1)*amount], cons) + beta/2 * np.linalg.norm(cons)**2
    
    # for n -> 1 (circular graph)
    cons = w[(n_nodes-1)*amount:n_nodes*amount] - w[0:amount]
    return residual + np.dot(mu[(n_nodes-1)*amount:n_nodes*amount], cons) + beta/2 * np.linalg.norm(cons)**2

# Define augmented lagrangian
def augmented_lagrangian(n_nodes, amount, features, labels, beta, w, mu):
    """
    n_nodes: number of nodes
    amount: Amount of data points each node can access
    features, labels: all features and labels
    beta: hyperparameter of L_beta
    w: current iterate
    mu: estimate of Lagrange multiplier
    """
    
    residual = constraint(n_nodes, amount, w, mu)
    objective = loss(features, labels, w) + residual

    return objective

# setup the distributed system
def setup_distributed_system(n_nodes, n_samples, features, labels, estimate, mu_0, beta, lamb):
    """
    n_nodes: number of nodes
    n_samples: number of samples
    features: features stored in this node
    labels: labels stored in this node
    estimate: estimate of this node
    mu_0: first guess at corresponding Lagrange multiplier
    beta: hyperparameter of augmented Lagrangian
    lamb: lambda for LASSO-Regularization (l1)
    """

    # Instantiation of workers
    # pool = Pool(n_nodes)
    pool = list()

    # Distributing work among nodes 1,...,n_nodes
    # For now: Node i has data [amount*i, amount*(i+1) )
    # work = [ for i in range(amount)]
    amount = int(n_samples / n_nodes)

    # Instantiate nodes and append to pool
    for node in range(n_nodes):
        # assign data points to correspinding node
        indices = [node*amount + i for i in range(amount)]
        # Take prev = estimate
        pool.append(Node(features[indices], labels[indices], estimate, mu_0, estimate.copy(), estimate.copy(), estimate.copy(), beta, lamb, estimate.copy()))
    # Return node pool
    return pool

# Proximal Operator LASSO-regularization thorugh l1 (soft-thresholding)
def prox_l1(iterate, param):
    """
    iterate: value at which we want to evaluate the proximal operator
    param: hyperparameter of soft-thresholding
    """

    # replace negative values in max in formula of proximal operator
    modified = np.abs(iterate) - param
    modified[modified < 0] = 0

    # formula for proximal operator of l1
    optimum = modified * np.sign(iterate)
    return optimum

# One step of ADMM in star graph
# TO-DO: Create parallelized version
def distributed_optimizer_step_star(pool, n_iter_grad, beta, lamb):
    """
    pool: Pool of nodes with corresponding data
    n_iter_grad: number of iterations of gradient descent
    beta: hyperparameter of augmented Lagrangian
    lamb: hyperparameter of l1-regularization
    """
    results = []
    multipliers = []
    # For the z update, see Boyd's manuscript on ADMM, page 52
    dim = pool[0].get_estimate().shape[0]
    avg_x = np.zeros(dim)
    avg_mu = np.zeros(dim)
    for i in range(0, len(pool)):
        # optimize locally
        results.append(pool[i].optimize(n_iter_grad))
        multipliers.append(pool[i].get_mu())

        avg_x += results[i] / n_nodes
        avg_mu += multipliers[i] / n_nodes


    intermediate = avg_x + avg_mu / beta
    # update centre variable that means l1(centre) + quadratic penalties    
    centre = prox_l1(intermediate , lamb/(n_nodes * beta))
    
    # Communicate centre to each node
    for i in range(0, len(pool)):
        # optimize locally
        pool[i].update_centre(centre)

# One step of ADMM in cyclic graph
def distributed_optimizer_step_cyclic(pool, n_iter_grad):
    """
    pool: Pool of nodes with corresponding data
    n_iter_grad: number of iterations of gradient descent
    """
    for i in range(0, len(pool)):
        # optimize locally
        result = pool[i].optimize(n_iter_grad)
        # communicate result and multiplier part to previous node
        pool[(i-1) % len(pool)].update_next(result)
        pool[(i-1) % len(pool)].update_next_mu(pool[i].get_mu())
        # communicate result to next node
        pool[(i+1) % len(pool)].update_prev(result)

# Coordinate whole ADMM
def distributed_optimizer(pool, n_iter, n_iter_grad, beta, lamb):
    """
    pool: Pool of nodes with corresponding data
    n_iter: Number of iterations of outer loop of ADMM
    n_iter_grad: Number of iterations for GD in subproblems
    beta: hyperparameter of augmented Lagrangian
    lamb: hyperparameter of l1-regularization
    """

    # To-DO: Allow to choose topology easier, see also at bottom of code
    for iter in range(n_iter):
        # cyclic topology
        # distributed_optimizer_step_cyclic(pool, n_iter_grad)

        # star topology
        distributed_optimizer_step_star(pool, n_iter_grad, beta, lamb)

# To-DO: depending on graph topology, only take necessary attributes for that
class Node:

    def __init__(self, features, labels, estimate, mu, prev, next, next_mu, beta, lamb, centre):
        """
        features: features stored in this node
        labels: labels stored in this node
        estimate: estimate of this node
        mu: estimate of Lagrange multiplier part w.r.t equality constraint
        prev: currently communicated guess from previous node (circular topology)
        next: currently communicated guess from next node (circular topology)
        next_mu: estimate of Lagrange multiplier part w.r.t equality constraint of next node
        beta: hyperparameter of augmented Lagrangian
        lamb: lambda for LASSO-Regularization (l1)

        centre: shared veriable in star based graph topology
        """
        self.features = features
        self.labels = labels
        self.estimate = estimate
        self.mu = mu
        self.prev = prev
        self.next = next
        self.next_mu = next_mu
        self.beta = beta
        self.lamb = lamb

        self.centre = centre
    
    # Getters
    def get_estimate(self):
        return self.estimate

    def get_mu(self):
        return self.mu
    
    def get_prev(self):
        return self.prev
    
    def get_next(self):
        return self.prev
    
    def get_next_mu(self):
        return self.next_mu
    
    def get_centre(self):
        return self.centre

    # Setters 

    def update_estimate(self, estimate):
        """
        estimate: estimate of this node
        """
        self.estimate = estimate
    
    def update_mu(self, update):
        """
        update: additive update of estimate of Lagrange mulitplier
        """
        self.mu += update

    def update_prev(self, prev):
        """
        prev: currently communicated guess from previous node (circular topology)
        """
        # Update prev after getting information from previous node
        self.prev = prev

    def update_next(self, next):
        """
        prev: currently communicated guess from previous node (circular topology)
        """
        # Update next after getting information from next node
        self.next = next

    def update_next_mu(self, next_mu):
        """
        next_mu: estimate of Lagrange multiplier part w.r.t equality constraint of next node
        """
        self.next_mu = next_mu

    def update_centre(self, centre):
        self.centre = centre
    
    # Gradient of loss function on this node
    def loss_gradient(self):
        # Calculating gradient of objective loss
        gradient = 1 / self.features.shape[0] * np.dot(self.features.T, sigmoid_vec(self.features @ self.get_estimate()) - self.labels)
        return gradient

    def optimize(self, n_iter):
        return self.optimize_cyclic(n_iter)
    
    # GD to solve subproblem in star based graph
    def optimize_star(self, n_iter):
        """
        n_iter: Number of iterations of gradient descent (GD)
        """
        # x is storing the iterates of GD
        x = self.estimate.copy()
        start_rate = 0.05
        centre = self.get_centre()
        
        # Minimize by gradient descent
        for iter in range(n_iter):
            # print(loss(x, features, labels))
            # Stepsize
            s = start_rate# / (iter + 1)**0.5
            # gradient descent update with gradient of augmented lagrangian
            # print(np.linalg.norm(x))
            x += (-s)*(self.loss_gradient() + self.mu + (self.beta)*(x - centre))
        
        # Update estimate
        self.update_estimate(x)

        # Update estimate of Lagrange multiplier
        self.update_mu(self.beta*(x - centre))
        
        # print loss after GD
        # print(loss(x, features, labels))
        # print(self.loss_gradient())

        return x

    
    # (Sub)Gradient Descent to solve subproblem in circular graph
    def optimize_cyclic(self, n_iter):
        """
        n_iter: Number of iterations of gradient descent (GD)
        """
        # x is storing the iterates of GD
        x = self.estimate.copy()
        start_rate = 0.05
        prev = self.get_prev()
        next = self.get_next()
        next_mu = self.get_next_mu()
        # Minimize by gradient descent
        for iter in range(n_iter):
            # print(loss(x, features, labels))
            # Stepsize
            s = start_rate# / (iter + 1)**0.5
            # gradient descent update with gradient of augmented lagrangian
            # print(np.linalg.norm(x))
            x += (-s)*(self.loss_gradient()+ (self.mu - next_mu) + (self.beta)*(2*x - (prev + next)) + self.lamb/self.features.shape[0] * np.sign(x))
        
        # Update estimate
        self.update_estimate(x)

        # Update estimate of Lagrange multiplier
        self.update_mu(1000*self.beta*(2*(x/1000) - (prev + next)/1000))

        return x

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

    # Initalize topology of communication
    # To-DO: Implement mechanism to choose topology
    # For now: fixed to star graph

    # Number of nodes
    n_nodes = 20

    # Initialize distributed system
    beta = 5
    lamb = 10
    pool = setup_distributed_system(n_nodes, n_samples, features_train, labels_train, w.copy(), mu_0.copy(), beta, lamb)

    # Starting loss and gradient norm
    print(f"Starting loss: {loss(w, features_train, labels_train) + lamb * np.linalg.norm(w,1)}")
    print(np.linalg.norm(pool[0].loss_gradient()))

    # Now optimization process

    # Number of iterations of outer loop of ADMM
    n_iter = 500
    # Number of iterations for GD in subproblems
    n_iter_grad = 50

    # measure time to train model
    start_time = time.time()

    # update
    distributed_optimizer(pool, n_iter, n_iter_grad, beta, lamb)

    # stop time to train model
    end_time = time.time()

    # Take estimate of the last node
    w_guess = pool[-1].get_estimate()

    # get predictions of the trained model
    prediction_train = predict(w_guess, features_train)
    prediction_test = predict(w_guess, features_test)
    
    # training accuracy, i.e. correctly specifies samples
    accuracy_train = 1 - np.sum(np.abs(labels_train - prediction_train)) / n_samples
    accuracy_test = 1 - np.sum(np.abs(labels_test - prediction_test)) / len(indices_test)

    # Summary statistics
    
    print(f"Training took {end_time - start_time} seconds.")
    print(f"End loss: {loss(w_guess, features_train, labels_train) + lamb * np.linalg.norm(w_guess,1)}")
    print(np.linalg.norm(pool[0].loss_gradient()))
    print()
    print(f"Accuracy in {n_samples} training samples: {accuracy_train}")
    print(f"Accuracy in {len(indices_test)} test samples: {accuracy_test}")
