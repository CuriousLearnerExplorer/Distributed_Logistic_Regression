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

Here we consider two possible system topology:
1. A cyclic structure: node_1 -> node_2 -> ... -> node_n -> node_1
2. A star structure: node_i -> centre for all i, where centre is a 
   special node; communication is allowed in the following sense: Each 
   node_i communicates their estimates to the centre, which processes 
   it (formally while handling the l1-regularization) and then 
   communicates a consensus solution uniformly back.