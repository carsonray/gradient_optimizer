from genetic import *
from rand_grad import *

import numpy as np
import matplotlib.pyplot as plt

def gradient_test(alg, dense, exp, poprange, popsize, epochs):
    """Runs genetic algorithms on a random gradient and finds the highest point"""

    # Sets up plots with comparison on y-axis and epochs on x-axis
    # Leaves extra spot for comparitive stat results
    f, axis = plt.subplots(epochs)

    # Creates random gradient with a size of one more than the population range (so there is no overflow)
    gradient = rand_grad(0, 1, dense, exp, size=(np.array(poprange) + 1))

    def grad_assess(epoch, points):
        """Assesses population points based off of gradient"""
        # Rounds all points
        points = np.round(points)
        points = points.astype(int)
        
        # Displays population
        axis[epoch].imshow(gradient, cmap=plt.get_cmap("terrain"))
        axis[epoch].scatter(*points.swapaxes(0,1), c="#000000")
        
        # Reshapes points so that fancy indexing per axis can be used
        values = gradient[tuple(points.swapaxes(0,1))]

        # Returns points and values
        return points, values

    # Sets up algorithm
    pop = alg.setup(grad_assess, GenAlg.mean_spread_cross(50, 0.1), range=poprange, size=popsize, gsize=2)
    
    # Runs algorithm
    alg.run(pop, epochs)

def curve_metrics(alg):
    stats = alg.stats
    plt.plot(stats["epoch"].values, stats["max"].values, label="Max")
    plt.plot(stats["epoch"].values, stats["min"].values, label="Min")
        
    plt.legend()
    plt.show()

alg = GenAlg()
gradient_test(alg, dense=0.1, exp=0.8, poprange=(40,40), popsize=40, epochs=20)
plt.show()

curve_metrics(alg)