import matplotlib.pyplot as plt
import numpy as np
import math


class Point:
    def __init__(self, data, label, weight):
        self.data = data
        self.label = label
        self.weight = weight

# 1D Splitter object to represent a simple straight line classifier
class Splitter:
    def __init__(self, value, dimension, positive_for_less_than):
        self.value = value                                      # Value to split at
        self.dimension = dimension                              # Dimension number to split on
        self.positive_for_less_than = positive_for_less_than    # True if (data < value) -> (label = 1)

    def predict(self, point):
        if point.data[self.dimension] < self.value:
            return 1 if self.positive_for_less_than else -1
        else:
            return -1 if self.positive_for_less_than else 1


# Any generic classifier that can make a prediction for a point
class Classifier:
    def __init__(self, func):
        self.func = func

    def predict(self, point):
        return self.func(point)


# Returns a Classifier representing the best 1D Splitter object for the data points & ranges provided
# ranges = list of ranges, one for each dimension of the data
def one_dimension_split(points, ranges):
    min_error = 10000.0
    result = Classifier(lambda x: 1)

    # Check the classification error for every combination of 1D line in each dimension,
    # for each range value provided, for each positive_for_less_than value.
    dimension_number = 0
    for r in ranges:
        for value in r:
            for pflt in [True, False]:
                splitter = Splitter(value, dimension_number, pflt)
                classifier = Classifier(splitter.predict)
                error = classification_error(classifier, points)
                if error < min_error:
                    min_error = error
                    result = classifier
        dimension_number += 1

    return result


def classification_error(classifier, points):
    error = 0.0
    for point in points:
        if classifier.predict(point) != point.label:
            error += point.weight
    return error


def contribution(error):
    if error == 0:
        # Return a very high contribution if this simple predictor was perfect
        return 100

    return 0.5 * math.log((1-error) / error, 2)


def update_weights(classifier, contr, points):
    total_weight = 0.0
    for point in points:
        point.weight *= math.exp(-1 * contr * point.label * classifier.predict(point))
        total_weight += point.weight

    # Normalize the point weights
    for point in points:
        point.weight /= total_weight


def adaboost(points, classifier_generator, num_iterations=10):
    h_and_b = []

    for t in xrange(num_iterations):
        h = classifier_generator(points)
        error = classification_error(h, points)
        b = contribution(error)
        update_weights(h, b, points)

        h_and_b.append([h, b])
        print 'b=', b

    def final_predictor(x):
        s = 0.0
        for t in xrange(num_iterations):
            s += h_and_b[t][0].predict(x) * h_and_b[t][1]
        return 1 if s > 0 else -1

    return final_predictor


def main():
    X = np.append(np.append(np.append(
        np.random.normal(30, scale=5, size=30),
        np.random.normal(50, scale=15, size=60)),
        np.random.normal(70, scale=5, size=30)),
        np.random.normal(50, scale=5, size=80))
    Y = np.append(np.append(np.append(np.append(
        np.random.normal(50, scale=15, size=30),
        np.random.normal(30, scale=5, size=30)),
        np.random.normal(70, scale=5, size=30)),
        np.random.normal(50, scale=15, size=30)),
        np.random.normal(50, scale=5, size=80))
    L = np.append(
        -1 * np.ones(120),
        np.ones(80))

    points = []
    for i in xrange(200):
        points.append(Point([X[i], Y[i]], L[i], 1.0/200))

    predictor = adaboost(
        points,
        lambda p: one_dimension_split(p, [xrange(100), xrange(100)]),
        num_iterations=60)

    # Plot the decision boundary
    for x in xrange(0, 101, 5):
        for y in xrange(0, 101, 5):
            plt.scatter(x, y, marker='s',
                        color=('mistyrose' if predictor(Point([x, y], 1, 1)) == -1 else 'lightsteelblue'),
                        s=250)

    # Plot the data points, scaled by final weights
    for i in xrange(200):
        plt.scatter(points[i].data[0], points[i].data[1],
                    color=('red' if points[i].label == -1 else 'blue'),
                    s=2000*points[i].weight)

    plt.show()


if __name__ == '__main__':
    main()
