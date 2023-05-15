
# Supervised Topological Maps

Supervised topological maps (STMs) are a type of machine learning algorithm that combines the strengths of self-organizing maps (SOMs) and supervised learning techniques.

## Introduction

Controlling the internal representation space of a neural network is a desirable feature because it allows for the generation of new data in a supervised manner. In a recent paper by Mannella et al. (2020), they proposed a novel approach to achieving this using a variation of the SOM algorithm, called Supervised Topological Maps (STMs).

## Training STMs

STMs are a kind of neural network that can be trained with unsupervised learning to produce a low-dimensional discretized mapping of the input space. However, the final topology of the mapping space of a SOM is not known before learning, making it difficult to interpolate new data in a supervised way. Mannella et al. (2020) proposed a variation of the SOM algorithm that constrains the update of prototypes so that it is also a function of the distance of its prototypes from extrinsically given targets in the mapping space. This allows for a supervised mapping where the position of internal representations in the mapping space is determined by the experimenter.

## the Algorithm 

1. Initialize the weights of the SOM randomly
2. For each input vector x do the following:
   a. Find the best matching unit (BMU) in the SOM for x
   b. Update the weights of the BMU and its neighbors using the following formula:

         w_i(t+1) = w_i(t) + η(t) *  h(i, trg)(t) * h(i, BMU)(t) * (x - w_i(t))
      
      where **w_i(t)** is the weight of the ith neuron at time t, **η(t)** is the learning rate at time t, **h(i, j)(t)** is the neighborhood function between the i-th neuron and the j-th neuron at time t, **x** is the input vector and **trg** is the target position in the neurons' space for the input **x**
   c. Update the learning rate and neighborhood function according to a predefined schedule
3. Repeat step 2 for a fixed number of epochs or until convergence
4. Train a classifier on the labeled data using the SOM as a feature extractor
5. Evaluate the performance of the classifier on a test set
