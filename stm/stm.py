# -*- coding: utf-8 -*-

"""
Supervised Topological Map

An organizing map whose topology can be guided by a teaching signal.
Generalizes SOMs.
    
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.engine import base_layer_utils


def radial2d(mean, sigma, size):
    """ Gives radial bases on a 2D space, flattened into 1d vectors, 
    given a list of means and a std_dev.
                
    Es.

    size: 64              ┌────────┐         ┌─────────────────────────────────────────────────────────────────┐      
    mean: (5, 4)  ----->  │........│  -----> │ ....................,+,....,oOo,...+O@O+...,oOo,....,+,.........│
    sigma: 1.5            │........│         └─────────────────────────────────────────────────────────────────┘
                          │....,+,.│
                          │...,oOo,│
                          │...+O@O+│
                          │...,oOo,│
                          │....,+,.│
                          │........│
                          └────────┘

    Args: 
        mean (list): vector of means of radians
        sigma (float): standard deviation of radiants
        size (int): dimension of the flattened gaussian (side is sqrt(dim))

    Returns:
        (np.array): each row is a flattened radial basis in 
                 the (sqrt(dim), sqrt(dim)) space
    
    """
    side = tf.sqrt(tf.cast(size, dtype="float"))
    mean = tf.stack([mean // side, mean % side])
    x  = tf.range(side, dtype="float")
    Y, X  = tf.meshgrid(x, x)
    grid_points = tf.transpose(tf.stack([tf.reshape(X, [-1]), tf.reshape(Y, [-1])]))
    diff = tf.expand_dims(grid_points, axis=2) - tf.expand_dims(mean, axis=0)
    radial_basis =  tf.exp(-0.5*tf.pow(sigma, -2)*tf.pow(tf.norm(diff, axis=1), 2))
    return tf.transpose(radial_basis)

def RBF2DInterpolation(points, sigma, size):
    side = tf.sqrt(tf.cast(size, dtype="float"))
    x  = tf.range(side, dtype="float")
    Y, X  = tf.meshgrid(x, x)
    grid_points = tf.transpose(tf.stack([tf.reshape(X, [-1]), tf.reshape(Y, [-1])]))    
    diff = tf.expand_dims(grid_points, axis=1) - tf.expand_dims(points, axis=0)
    radial_basis =  tf.exp(-0.5*tf.pow(sigma, -2)*tf.pow(tf.norm(diff, axis=2), 2))
    return tf.transpose(radial_basis)
    

class STM(keras.layers.Layer):
    """ A generic topological map
    """
    
    def __init__(self, output_size, sigma,
            learn_intrinsic_distances = True, 
            extrinsic_distances = None,
            **kwargs):
        """
        Args:
            output_size (int): number of elements in the output layer (shape will be 
                                (sqrt(output_size), sqrt(output_size)))
            sigma (float): starting value of the extension of the learning neighbourhood
                           in the output layer.
            learn_intrinsic_distances (boolean): if learning depends on the distance of prototypes from inputs.
            exrinsic_distances (Tensor): if learning depends on the distance of prototypes from targets.

        """

        self.output_size = output_size
        self._sigma = sigma
        self.learn_intrinsic_distances = learn_intrinsic_distances
        self.extrinsic_distances = extrinsic_distances

        super(STM, self).__init__(**kwargs)

    def build(self, input_shape):
         
        self.sigma = self.add_weight(
                name="sigma", shape=(), 
                initializer=tf.constant_initializer(self._sigma), 
                trainable=False)
        
        self.kernel = self.add_weight(
                name='kernel', 
                shape=(input_shape[1], self.output_size),
                initializer=tf.zeros_initializer(),
                trainable=True)

        super(STM, self).build(input_shape)  

    def call(self, x):
         
        # compute norms 
        norms = tf.norm(tf.expand_dims(x, 2) - \
                tf.expand_dims(self.kernel, 0), axis = 1)
        norms2 = tf.pow(norms, 2)
        
        # compute activation
        radials = self.get_radials(norms2) 

        return radials*norms2        

    def backward(self, radials): 
        x = tf.matmul(radials, tf.transpose(self.kernel))
        return x

    def get_radials(self, norms2):
        
        wta = tf.cast(tf.argmin(norms2, axis=1), dtype="float")
        radials = radial2d(wta, self.sigma, self.output_size)
       
        return radials


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

    def loss(self, radial_norms2, extrinsic=None):
        if extrinsic is None:
            extrinsic = tf.ones_like(radial_norms2)
        return tf.reduce_mean(radial_norms2 * extrinsic, axis=1)
     