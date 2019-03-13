__author__ = 'Alan Shen'

import numpy,math

def readMatrix_NE(fileName):
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

def readMatrix_SGD_RGD(fileName):
    lineNum = 0
    f = open(fileName, "r")

    X1 = numpy.zeros(shape=(0, 2))
    Y1 = []
    X2 = numpy.zeros(shape=(0, 2))
    Y2 = []
    next(f)
    for line in f:
        lineNum +=1
        if lineNum <= 1000:
            line.replace("\n", "");
            x, y, z = line.split(" ");
            X1 = numpy.vstack([X1, [float(x), float(y)]])
            Y1.append(float(z))
        else:
            line.replace("\n", "");
            x, y, z = line.split(" ");
            X2 = numpy.vstack([X2, [float(x), float(y)]])
            Y2.append(float(z))

    f.close()
    return X1,Y1,X2,Y2

def SGD_get_weights(x, y, verbose = 0,step_size=0.01, max_iter_count=10000):
    shape, dim = x.shape
    print(shape,dim)
    w = numpy.ones((dim,), dtype=numpy.float32)
    print(w)
    loss = 10
    iteration = 0

    while loss > 0.001 and iteration < max_iter_count:
        loss = 0
        error = numpy.ones((dim,), dtype=numpy.float32)
        for i in range(shape):
            predict_y = numpy.dot(w.T, x[i])
            for j in range(dim):
                error[j] += (y[i] - predict_y) * x[i][j]
                print("error= ",error[j])
                w[j] += step_size * error[j] / shape
                #print ("j = ",j,"w[j]= ",w[j])
        for i in range(shape):
            predict_y = numpy.dot(w.T, x[i])
            error = (1 / (shape * dim)) * numpy.power((predict_y - y[i]), 2)
            loss += error

        print("iter_count: ", iteration, "the loss:", loss)
        iteration += 1

    return w

def RGD_get_weights(x, y, verbose = 0,step_size=0.01, max_iter_count=10000):
    shape = x.shape
    x = numpy.insert(x, 0, 1, axis=1)
    w = numpy.ones((shape[1]+1,))
    print(w)
    weights = []

    learning_rate = 10
    iteration = 0
    loss = None
    while iteration <= 1000 and loss != 0:
        for ix, i in enumerate(x):
            pred = numpy.dot(i,w)
            if pred > 0: pred = 1
            elif pred < 0: pred = -1
            if pred != y[ix]:
                w = w - learning_rate * pred * i
            weights.append(w)

        loss = numpy.dot(x, w)
        loss[loss<0] = -1
        loss[loss>0] = 1
        loss = numpy.sum(loss - y )

        if verbose == 1:
            print(numpy.sum(loss - y ))
        if iteration%10 == 0:
            learning_rate = learning_rate / 2
        iteration += 1

    print('Weights: ', w)
    print('Loss: ', loss)
    return w, weights

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

def RSS_testingset(X,Y):
    var = normalEquation(X, Y)
    sum = 0
    f = [lambda i=i: math.pow(Y[i]-numpy.matrix(X[i]).dot(var),2) for i in range(100)]
    for i in range(100):
        sum +=f[0](i)
    return sum / 100

def do_predict(X,Y):
    return numpy.matrix([1, 1, 135]).dot(normalEquation(X,Y))

def main():

    train_X, train_Y, test_X, test_Y = readMatrix_NE('HW2_linear_regression.txt')
    print(train_Y)
    print("Linear Regression")
    print("RSS for training dataset is: ",RSS_traingset(train_X, train_Y))

    print("RSS for testing dataset is: ",RSS_testingset(test_X, test_Y))

    print("Prediction for (1,135) is ",do_predict(train_X, train_Y))

    train_X2, train_Y2, test_X2, test_Y2 = readMatrix_NE('HW2_logistic_regression.txt')

    print("Logistic Regression")
    print("RSS for training dataset is: ", RSS_traingset(train_X2, train_Y2))

    print("RSS for testing dataset is: ", RSS_testingset(test_X2, test_Y2))

    print("Prediction for (1,135) is ", round(float(do_predict(train_X2, train_Y2))))

if __name__ == '__main__':
  main()
