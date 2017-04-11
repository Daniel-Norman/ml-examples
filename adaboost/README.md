## AdaBoost Classification
Uses AdaBoost to create a complex decision boundary from extremely simple classifiers.

Each simple classifier is a 1D straight line that divides the data in half.
For 2D data, this is represented as either a vertical line (at any x) or horizontal line (at any y).

Example of a single classifier on random data (AdaBoost with # Iterations = 1):

![alt tag](http://i.imgur.com/6Wt4aQM.png)

This classifier minimized weighted error by splitting the data into a left and right half.
Each data point is drawn scaled by its weight after running the algorithm.


Result from running again on similarly distributed data, using 30 simple classifiers:

![alt tag](http://i.imgur.com/LxSK2so.png)

Notice the decision boundary surrounding the blue data points, and the large weight given to red data points within
the blue boundary that are constantly misclassified.
