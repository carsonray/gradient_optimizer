import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from flextensor import FlexTensor as Tensor

class GenAlg():
    def __init__(self):
        # Initializes iterations to 0
        self.epoch = 0

        # Initilizes pandas dataframe with epochs column
        self.stats = pd.DataFrame(columns=["epoch"])
    
    def log(self, vals):
        """Logs stat variables"""
        # Creates dataframe to append
        append = pd.DataFrame(vals)

        # Add epoch column with length of append
        append["epoch"] = np.repeat(self.epoch, len(append))
                
        # Appends new values by epoch
        self.stats = pd.concat([self.stats, append])
    
    def clear(self):
        """Clears stat dataframe"""
        # Resets epoch count
        self.epoch = 0

        # Saves data
        result = self.stats
        
        # Resets to blank dataFrame
        self.stats = pd.DataFrame(columns=result.columns)

        # Returns data
        return result
    
    def run(self, data, epochs):
        """Recursively runs system and passes data to next iteration"""
        # Clears log and sets epochs to 0
        self.clear()
        
        # Loops through epochs and runs function
        for epoch in range(epochs):
            self.epoch = epoch
            data = self.generation(data)

        # Returns result
        return data

    def generation(self, data):
        """Runs a generation of the genetic algorithm"""
        
        # If next generation, recursively map from results of last generation
        # Also clips population within ranges
        pop = data if self.epoch == 0 else self.clip(data)
        
        # Assesses current generation and adds to population
        data = self.add_pop(self.assess(self.epoch, pop.copy()))

        # Logs statistics
        self.log({
            "max": [data[1].max()],
            "min": [data[1].min()],
            "mean": [np.average(data[1])],
            "median": [np.median(data[1])],
            "valRange": [np.ptp(data[1])]
        })

        # Groups individuals by assessment scores
        data = self.group(data)

        # Crosses attributes of groups
        data = self.cross(data)

        # Returns resulting population and values
        return data

    def setup(self, assess, cross, range=None, size=None, gsize=2):
        """Creates random initial population"""
        # Sets assessment function to maximize
        self.assess = assess

        # Sets cross function to optimize points
        self.cross = cross

        # Sets popsize
        self.popsize = size

        # Sets group size
        self.gsize = gsize
        
        # Sets default population range from 0 to 1
        poprange = [1]*size if range is None else range
        self.poprange = np.array(poprange)

        # Creates random population and stacks it by last axis to get array of points
        self.population = np.stack([np.random.random(size)*length for length in poprange], axis=-1)

        # Returns resulting population
        return self.population

    def add_pop(self, data):
        """Selects fittest individuals to continue population size"""
        # Unpacks both points and values
        points, values = data

        if self.epoch == 0:
            # If first epoch, reset population to current members
            self.population = points

            self.values = values
        else:
            # Appends individuals to current population and values
            self.population = np.concatenate((self.population, points))

            self.values = np.concatenate((self.values, values))

        # Sorts population by values and then sorts values
        self.population = self.population[np.argsort(self.values)[::-1]]

        self.values = np.sort(self.values)[::-1]

        # Trims population to population size
        self.population = self.population[0:self.popsize]

        self.values = self.values[0:self.popsize]

        # Returns population and values
        return self.population, self.values

    def clip(self, population):
        """Clips population within ranges"""
        # Swaps axes of population for clipping
        pop = population.copy().swapaxes(0,1)

        # Clips each axis of population within range
        pop = np.clip(pop, 0, self.poprange[:, np.newaxis])

        # Returns the swapped result
        return pop.swapaxes(0,1)

    # Algorithm functions
    def group(self, data):
        # Unpacks both points and values
        points, values = data

        # Sets up points as list of points
        points = Tensor(points, "points", "coords")

        # Sets up values in same way
        values = Tensor(values, "values")

        # Number of points
        pointNum = points.size("points")

        # Finds largest number of points that can be divided into sized groups
        newNum = pointNum - (pointNum % self.gsize)

        # Finds number of groups
        groups = int(newNum / self.gsize)
            
            
        # Trims population to new number of points
        points.vals = points.vals[:newNum]
        values.vals = values.vals[:newNum]

        size = int(newNum / groups)

        # Returns reshaped population into groups (adds axis to beginning
        points = points.by([1], "points", "coords").reshape(groups, size, points.size("coords"))
        values = values.by([1], "values").reshape(groups, size)

        return points, values

    def mean_spread_cross(self, expand=2, exp=0.14):
        """Creates offspring by averaging characteristics of parent groups"""
        # Returns cross function with given parameters
        def func(data):
            # Unpacks both points and values
            points, values = data

            # Sets up points as list of groups of points
            points = Tensor(points, "groups", "points", "coords")

            # Sets up values in same way but normalizes for each group
            values = Tensor(softmax(values, axis=-1), "groups", "values")

            # Gets valRange and scales
            scale = np.ptp(values.vals)*values.size("values")
            scale = expand * exp**scale + 1

            # Weights each point by the values and sums each group to get a weighted average
            average = np.sum(points.vals * values.by("groups", "values", [1]), axis=points.ax("points"))
            average = Tensor(average, "groups", "coords")

            # Finds the distance of all points to the weighted average and averages in each group
            dists = np.average(distance(points.vals, average.by("groups", [1], "coords")), axis=points.ax("points"))
            dists = dists * scale
            
            # Loops through each group and produces random points within a circle
            # extending from the weighted averages with a radius of the distances

            for center, radius in zip(average.vals, dists):
                # Creates random points within a circle
                randPoints = rand_circ(points.size("points"), center=center, radius=radius)

                # Create ouput points if array doesn't exist
                try:
                    newPoints = np.append(newPoints, randPoints, axis=0)
                except NameError:
                    newPoints = randPoints
            

            # Returns new points
            return newPoints
        
        return func
            
    def old_mean_spread_cross(self, variance=0.0001):
        """Creates offspring by averaging characteristics of parent groups"""
        # Returns cross function with given parameters
        def func(data):
            # Unpacks both points and values
            points, values = data

            # Sets up points as list of groups of points
            points = Tensor(points, "groups", "points", "coords")

            # Sets up values in same way
            values = Tensor(values, "groups", "values")

            # Creates array of weight adjustments and scales to particular variance
            # Each group has different weight adjustments to create a new offspring
            # group of the same size
            # Adjustment is array of matrices, with each matrix being one adjustment to the weights for all groups
            adjust = np.random.random(points.shape("points", "groups", "points"))*2*variance - variance

            adjust = Tensor(adjust, "adjusts", *values.axes)

            

            # Adds adjustments to values and normalizes values for each group
            values = softmax(np.squeeze(adjust.by("adjusts", "groups", "values") + values.by([1], "groups", "values")), axis=-1)
            values = Tensor(values, *adjust.axes)

            # Multiplies each point by the weights and sums each group.
            points = np.sum((values.by("adjusts", "groups", "values", [1]) * points.by([1], "groups", "points", "coords")), axis=2)
            points = Tensor(points, "adjusts", "groups", "coords")

            # Returns points and coords
            return points.vals.reshape(points.size("adjusts", "groups"), points.size("coords"))
        
        return func
        

# Utility functions
def rand_circ(num, center=[0, 0], radius=1):
    """Creates random points within a circle/sphere in multiple dimensions"""
    # Dimensions are equal to the length of the center point
    center = np.array(center)
    dims = len(center)
    
    # Creates array of random angles between 0 and 2pi
    angles = np.random.random((num, dims-1))*2*np.pi
    
    # Creates random magnitudes within radius
    mags = np.random.random((num, 1))*radius
    
    # Gets cosine of all angles and adds magnitudes to the beginning
    mags = np.append(mags, np.cos(angles), axis=-1)
    
    # Cumulatively sums cosines to get progressive magnitudes
    mags = np.cumprod(mags, axis=-1)
    
    # Multiplies pointwise by sines of angles and lets the last magnitude pass
    # through as the first coordinate
    points = mags * np.append(np.sin(angles), [[1]]*num, axis=-1)
    
    # Returns the points adjusted by the center
    return points + center

def distance(one, other):
    """Calculates vectorized distance between multidimensional points"""
    # Finds differences between arrays, squares them, and sums them and takes a square root
    # This completes the pythagorean theorem
    return np.sqrt(np.sum((one - other)**2, axis=-1))
        
def cartesian(arrays):
    """Returns the cartesian product of the arrays"""

    # Creates a meshgrid of each array and then flattens them
    axes = [axis.flatten() for axis in np.meshgrid(*arrays)]

    # Stacks the arrays along the last axis to create a list
    # of all possible combinations
    return np.stack(axes, axis=-1)

def scaleTo(array, min, max):
    """Scales array between min and max"""
    # Makes sure array is numpy array
    array = np.array(array)

    # Scales array
    return (array - array.min()) * (max - min) / np.ptp(array) + min


def softmax(array, axis=-1):
    """Runs softmax normalization along axis"""
    # Exponentializes array by e
    exp = np.exp(array)
    
    # Gets sum of exponentializes array along an axis
    # and then replaces the axis for broadcasting
    expSum = np.moveaxis(np.sum(exp, axis=axis)[np.newaxis, ...], 0, axis)
    
    # Returns exp array normalized by sum
    return (exp / expSum)
