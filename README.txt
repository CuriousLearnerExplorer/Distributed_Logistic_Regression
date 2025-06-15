Thank you for your interest!

The goal of this project is to train a 0-1-classifier using the 
data set MNIST for handwritten digits (0/1) in a distributed manner,
by introducing linear constraints such that different nodes can
optimize over parts of the training data independently of the others,
and then communicating with the other nodes.
We add regularization to the problem for more robust solutions.
The algorithm we use for this is ADMM, which allows the decoupling by 
introducing auxiliary equality constraints and minimizing the
augmented Lagrangian blockwise.