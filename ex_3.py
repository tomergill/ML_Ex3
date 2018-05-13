import numpy as np
from numpy import tanh
from random import shuffle
from time import time


def load_data_from_files(x_file, y_file, validation=False, validation_size=0.2, shuffle=True):
    x = np.loadtxt(x_file)
    y = np.loadtxt(y_file)

    if shuffle:
        yy = y.reshape(-1, 1)
        data = np.concatenate((x, yy), axis=1)
        np.random.shuffle(data)
        x, y = data[:, :-1], data[:, -1]

    if not validation:
        return x, y

    dev_size = int(x.shape[0] * validation_size)
    unique, times = np.unique(y, return_counts=True)
    examples_per_tag = (times * dev_size / float(y.shape[0])).astype(np.int)
    counters = [0] * 10
    train_indices, dev_indices = [], []

    for i in xrange(y.shape[0]):
        tag = int(y[i])
        if counters[tag] < examples_per_tag[tag]:
            counters[tag] += 1
            dev_indices.append(i)
        else:
            train_indices.append(i)

    train_x, dev_x = x[train_indices], x[dev_indices]
    train_y, dev_y = y[train_indices], y[dev_indices]
    return train_x, train_y, dev_x, dev_y


def initialize_weight(size1, size2=None, glorot=False):
    if glorot:
        n, m = size1, 1 if size2 is None else size2
        eps = np.sqrt(6.0 / (n + m))
        return np.random.uniform(-eps, eps, (size1, size2)) if size2 is not None else np.random.uniform(-eps, eps, size1)
    # else
    return  np.random.uniform(-0.08, 0.08, (size1, size2)) if size2 is not None else np.random.uniform(-0.08, 0.08,
                                                                                                      size1)


def softmax(x):
    """
    Computes the probablitic output function softmax:
    softmax(x[1:n])[i] = e^(x[i])/sum(e^x[j]) for 1<=j<=n
    :param x: Input vector
    :type x: np.ndarray
    :return: The probabilities vector for x
    """
    exps = np.exp(x - x.max())
    return exps / exps.sum()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def det_sigmoid(x):
    x = sigmoid(x)
    return x * (1 - x)


def ReLU(x):
    return np.maximum(x, 0)


def det_ReLU(x):
    """

    :param x:
    :type x: np.ndarray
    :return:
    """
    return (x > 0).astype(np.float)


def leaky_ReLU(x):
    return np.maximum(x, 0.01 * x)


def det_leaky_ReLU(x):
    out = np.copy(x)
    out[x > 0] = 1
    out[x <= 0] = 0.01
    return out


def det_tanh(x):
    return 1 - np.square(tanh(x))


func_to_derivative = {sigmoid: det_sigmoid, ReLU: det_ReLU, tanh: det_tanh, leaky_ReLU: det_leaky_ReLU}


def neglogloss(probs, y):
    return -np.log(probs[y])


class MLP1:
    def __init__(self, hidden_size, activation_function=ReLU, input_size=28 * 28, output_size=10, glorot=True):
        self.W1, self.b1 = initialize_weight(hidden_size, input_size, glorot=glorot), initialize_weight(hidden_size, glorot=glorot)
        self.W2, self.b2 = initialize_weight(output_size, hidden_size, glorot=glorot), initialize_weight(output_size, glorot=glorot)
        self.g = activation_function

    def forward(self, x, return_h=False):
        h = self.W1.dot(x) + self.b1
        h = self.g(h)
        out = self.W2.dot(h) + self.b2
        probs = softmax(out)

        if not return_h:
            return probs
        return probs, h

    def backprop(self, x, y, return_loss=True):
        probs, h = self.forward(x, return_h=True)
        if return_loss:
            loss = neglogloss(probs, y)

        gW2 = np.outer(probs, h)
        gW2[y, :] -= h

        gb2 = np.copy(probs)
        gb2[y] -= 1

        layer1 = self.W1.dot(x) + self.b1

        dl_dg = probs.dot(self.W2) - self.W2[y, :]
        dg_db1 = func_to_derivative[self.g](layer1)

        gb1 = dl_dg * dg_db1
        gW1 = np.outer(gb1, x)

        if return_loss:
            return loss, gW1, gb1, gW2, gb2
        return gW1, gb1, gW2, gb2

    def update_weights(self, lr, gW1, gb1, gW2, gb2):
        self.W1 -= lr * gW1
        self.b1 -= lr * gb1
        self.W2 -= lr * gW2
        self.b2 -= lr * gb2

    def set_params(self, params):
        self.W1, self.b1, self.W2, self.b2 = params


def dev_loss_and_accuracy(net, dev_x, dev_y):
    """

    :param net:
    :type net: MLP1
    :param dev_x:
    :param dev_y:
    :return:
    """
    total_loss = total = good = 0.0
    for x, y in zip(dev_x, dev_y):
        y = int(y)
        probs = net.forward(x)
        loss = neglogloss(probs, y)
        total_loss += loss
        total += 1
        good += probs.argmax() == y
    return total_loss / total, good / total


def train(net, train_x, train_y, epochs, lr, dev_x, dev_y):
    """

    :param net:
    :type net: MLP1
    :param train_x:
    :param train_y:
    :param epochs:
    :param lr:
    :param dev_x:
    :param dev_y:
    :return:
    """
    print "+-------+------------+----------+----------+------------+"
    print "| epoch | train loss | dev loss | accuracy | epoch time |"
    print "+-------+------------+----------+----------+------------+"

    best_params = []
    best_acc = 0

    indices = range(train_x.shape[0])
    for i in xrange(epochs):
        start = time()
        total_loss = 0.0

        shuffle(indices)
        for j in indices:
            x, y = train_x[j], int(train_y[j])
            loss, gW1, gb1, gW2, gb2 = net.backprop(x, y)
            total_loss += loss
            net.update_weights(lr, gW1, gb1, gW2, gb2)
        avg_loss, acc = dev_loss_and_accuracy(net, dev_x, dev_y)
        if acc >= best_acc:
            best_acc = acc
            best_params = [net.W1.copy(), net.b1.copy(), net.W2.copy(), net.b2.copy()]
        print "| {:^5} | {:010f} | {:7f} | {:6.4f}% | {:8f}s |".format(i, total_loss / len(indices), avg_loss,
                                                                       acc * 100,
                                                                   time() - start)
    print "+-------+------------+----------+----------+------------+"
    return best_params, best_acc


def predict_to_file(net, test_x, file_name):
    with open(file_name, "w") as f:
        for x in test_x:
            pred = str(net.forward(x).argmax())
            f.write(pred)
            f.write("\n")


def main():
    # Hyper-parameters
    hidden_size = 200  # 128  200 even better (89.518 over 20)
    learning_rate = 0.02
    epochs = 20
    activ_func = sigmoid

    # Load Data
    print "starting to load data..."
    t = time()
    train_x, train_y, dev_x, dev_y = load_data_from_files("train_x", "train_y", validation=True, validation_size=0.2)
    test_x = np.loadtxt("test_x", np.float)
    train_x, dev_x, test_x = train_x / 255.0, dev_x / 255.0, test_x / 255.0
    print "loaded data in {} seconds.\n".format(time() - t)

    # create neural net
    net = MLP1(hidden_size, activation_function=activ_func, glorot=True)

    # print hyper-params
    print "hidden layer size = {}".format(hidden_size)
    print "learning rate = {}".format(learning_rate)
    print "# of epochs = {}".format(epochs)
    print "activation function = {}".format(
        {sigmoid: "sigmoid", tanh: "tanh", ReLU: "ReLU", leaky_ReLU: "leaky ReLU"}[activ_func])
    print ""

    params, best_acc = train(net, train_x, train_y, epochs, learning_rate, dev_x, dev_y)
    print "The best accuracy is {}%".format(best_acc * 100)
    net.set_params(params)
    predict_to_file(net, test_x, "test.pred")


if __name__ == '__main__':
    main()
