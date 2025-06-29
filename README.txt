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

Originally I did this in Matlab, but changed the code to fit into
Python, as I wanted to get more practise in some important frameworks 
like numpy in Python.

Here we consider two possible system topologies:
1. A cyclic structure: node_1 -> node_2 -> ... -> node_n -> node_1
2. A star structure: node_i -> centre for all i, where centre is a 
   special node; communication is allowed in the following sense: Each 
   node_i communicates their estimates to the centre, which processes 
   it (formally while handling the l1-regularization) and then 
   communicates a consensus solution uniformly back.

For now we prefer the second structure, as although we need to introduce 
one more node for the centre, the individual node_i-updates can be 
implemented in parallel.

Model: