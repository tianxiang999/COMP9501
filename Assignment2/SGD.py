import numpy

def readMatrix(fileName):
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

def get_weights(x, y, verbose = 0,step_size=0.01, max_iter_count=10000):
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
            print('------------------------------------------')
            print(numpy.sum(loss - y ))
            print('------------------------------------------')
        if iteration%10 == 0: learning_rate = learning_rate / 2
        iteration += 1

    print('Weights: ', w)
    print('Loss: ', loss)
    return w, weights

def main():

    train_X, train_Y, test_X, test_Y = readMatrix('HW2_linear_regression.txt')
    train_Y = numpy.array(train_Y)
    test_Y = numpy.array(test_Y)
    print(train_X[0][0])
    print(train_Y)
    w, all_weights = get_weights(train_X, train_Y)
    print(w)

    x2 = numpy.array([[1,1,135]])
    pred = numpy.dot(x2, w)

    print('Predictions', pred)

if __name__ == '__main__':
    main()
