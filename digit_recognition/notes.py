#artificial neuron
# 1.perceptron
# 2.sigmoid

#we use stochastic gradient descent algo to train neural network

##perceptron
#weight:it denotes importance of inputs to output
#threshold value
# output = 0 or 1
#         : 0 if sum(w[j]*x[j]) wrt j <= threshold
#         : 1 if sum(w[j]*x[j]) wrt j > threshold
#
# bias..(b) = -threshold

##sigmoid neuron
# Sigmoid neurons are similar to perceptrons, but modified so that small
# changes in their weights and bias cause only a small change in their output.
#  ̈
# sigma(w*x+b) = 1/(1+exp(-(w*x+b))
# sigmoid_func : f(x) = 1/(1+exp(-x))
#
# multiple layer networks are some-
# times called multilayer perceptrons or MLPs, despite being made up of sigmoid neurons,
# not perceptrons.
