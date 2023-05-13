
#Supervised Topological Maps#
Supervised topological maps (STMs) are a type of machine learning algorithm that combines the strengths of self-organizing maps (SOMs) and supervised learning techniques. STMs are used for data visualization, clustering, and classification tasks.

##Introduction##
Controlling the internal representation space of a neural network is a desirable feature because it allows for the generation of new data in a supervised manner. In a recent paper by Mannella et al. (2020), they proposed a novel approach to achieving this using a variation of the SOM algorithm, which they call Supervised Topological Maps (STMs).

##Training STMs##
STMs are a kind of neural network that can be trained with unsupervised learning to produce a low-dimensional discretized mapping of the input space. However, the final topology of the mapping space of a SOM is not known before learning, making it difficult to interpolate new data in a supervised way. Mannella et al. (2020) proposed a variation of the SOM algorithm that constrains the update of prototypes so that it is also a function of the distance of its prototypes from extrinsically given targets in the mapping space. This allows for a supervised mapping where the position of internal representations in the mapping space is determined by the experimenter.

