__author__ = 'Alan Shen'

import numpy

def readMatrix(fileName):

    f = open(fileName, "r")

    X = numpy.zeros(shape=(0,3))
    Y = numpy.zeros(shape=(0,1))
    for line in f:
        line = line.replace("\n","");
        x,y,z = line.split(",");
        X = numpy.vstack([X, [1, float(x), float(y)]])
        Y = numpy.vstack([Y, [float(z)]])

    f.close()
    return X,Y


def normalEquation(X, Y):

    #theta = pinv(X'*X)*X'*y;

    xT = numpy.transpose(X)

    var1 = xT.dot(X)

    var3 = numpy.linalg.pinv(var1);

    var4 = xT.dot(Y)

    theta = var3.dot(var4)

    return theta;

def main():
    # Create matrix from file. Add x0 as all 1s to X so that Θ0 can be used as feature
    X, Y = readMatrix('ex1data2.txt')

    ## theta = pinv(X'*X)*X'*y;
    theta = normalEquation(X, Y)

    ## now that we have theta...we can start predicting
    ## Given predict the price of a house with 1650 square feet and 3 bedrooms

    var1 = numpy.matrix([1, 1650, 3])

    predict_y = var1.dot(theta)

    print("Predicted price of a 1650 sq-ft, 3 br house is:   ", predict_y)

if __name__ == '__main__':
  main()
