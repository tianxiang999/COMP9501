__author__ = 'Alan Shen'

import numpy,math

def readMatrix(fileName):
    lineNum = 0
    f = open(fileName, "r")

    X1 = numpy.zeros(shape=(0, 3))
    Y1 = numpy.zeros(shape=(0, 1))
    X2 = numpy.zeros(shape=(0, 3))
    Y2 = numpy.zeros(shape=(0, 1))
    next(f)
    for line in f:
        lineNum +=1
        if lineNum <= 1000:
            line.replace("\n", "");
            x, y, z = line.split(" ");
            X1 = numpy.vstack([X1, [1, float(x), float(y)]])
            Y1 = numpy.vstack([Y1, [float(z)]])
        else:
            line.replace("\n", "");
            x, y, z = line.split(" ");
            X2 = numpy.vstack([X2, [1, float(x), float(y)]])
            Y2 = numpy.vstack([Y2, [float(z)]])

    f.close()
    return X1,Y1,X2,Y2


def normalEquation(X, Y):

    #theta = pinv(X'*X)*X'*y;
    xT = numpy.transpose(X)
    var1 = xT.dot(X)
    var3 = numpy.linalg.pinv(var1);
    var4 = xT.dot(Y)

    theta = var3.dot(var4)
    return theta;

def RSS_traingset(X,Y):
    var = normalEquation(X, Y)
    sum = 0
    f = [lambda i=i: math.pow(Y[i]-numpy.matrix(X[i]).dot(var),2) for i in range(1000)]
    for i in range(1000):
        sum +=f[0](i)
    return sum / 1000

def main():

    train_X, train_Y, test_X, test_Y = readMatrix('HW2_linear_regression.txt')

    print(RSS_traingset(train_X, train_Y))


if __name__ == '__main__':
  main()
