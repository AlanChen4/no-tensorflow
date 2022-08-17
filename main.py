from utils import *
from keras.datasets import mnist

(train_data, train_label), (test_data, test_label) = mnist.load_data()

SCALE_FACTOR = 255
WIDTH = train_data.shape[1]
HEIGHT = train_data.shape[2]

train_data = train_data.reshape(train_data.shape[0], WIDTH * HEIGHT).T / SCALE_FACTOR
test_data = test_data.reshape(test_data.shape[0], WIDTH * HEIGHT).T / SCALE_FACTOR

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 100 == 0:
            print(f"Iteration: {i}")
            print(f"Accuracy: {get_accuracy(get_predictions(A2), Y)}")
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(train_data, train_label, 1000, 0.5)
