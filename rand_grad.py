from genetic import *

def point_grad(points, values, exp=0.9, size=None):
    """Creates a gradient between points in a dimensional space"""
    # Creates points tensor and converts values to np array
    points = Tensor(points, "coords", "points")
    values = np.array(values)
    
    # Creates list of ranges representing all possible coordinates in gradient
    ranges = [np.arange(length) for length in size]
    
    # Creates the cartesian product of all possible coordinates within size
    coords = Tensor(cartesian(ranges), "points", "coords")


    # Swaps axes of points so that it is a list of coordinates rather than a list of axes
    # Adds an axis to points so it is by the opposite orientation from the coords

    # Calculates distance between all possible coords and the selected gradient points
    dist = distance(coords.by("points", [1], "coords"), points.by([1], "points", "coords"))
    dist = Tensor(dist, "allPoints", "guidePoints")


    # The now two-dimensional array has the distance to each point in each column,
    # and the rows represent each possible point

    # The distances are decayed exponentially so that more distance has less weight
    # and negated so that more distance has less weight.
    # They are then weighted with the values array and averaged for each possible point
    newValues = np.average((exp**dist.by("allPoints", "guidePoints")) * values, axis=dist.ax("guidePoints"))

    # The result is a one-dimensional array of all of the possible points
    # with their corresponding values
    # This is reshaped back to the gradient space and returned
    return newValues.reshape(size)

def rand_grad(low, high, dense=0.01, exp=None, size=None):
    """Creates a random gradient by using a set of random points
    in the pointGrad function"""

    # Number of reference points is the total points in gradient times density
    pointNum = int(np.prod(size) * dense)

    # Creates random points by creating list of random coordinates for each axis
    # For each axis creates random intergers within the size of the axis
    points = np.stack([np.random.randint(length, size=pointNum) for length in size])

    # Creates random values for each point
    values = np.random.random(pointNum)*2 - 1

    # Creates gradient between the random points that is scaled to the end result
    return scaleTo(point_grad(points, values, exp=exp, size=size), low, high)

if __name__ == "__main__":
    numPlots = 2

    f, axis = plt.subplots(1, numPlots)

    for i in range(numPlots):
        grad = rand_grad(-1, 1, dense=0.1, exp=0.9, size=(100,100))
        img = axis[i].imshow(grad, cmap=plt.get_cmap("terrain"))

    plt.colorbar(img, ax=axis)
    plt.show()
