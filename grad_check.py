import numpy as np


def gradient_check(f, x):
    """
    Gradient check for a function f
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """
    fx, grad = f(x)  # Evaluate function value at original point
    h = 1e-4

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        # modify x[ix] with h defined above to compute the numerical gradient.
        # if you change x, make sure to return it back to its original state for the next iteration.
        # YOUR CODE HERE:
        xplus = x.copy()
        xminus = x.copy()
        xplus[ix] += h
        xminus[ix] -= h
        numeric_gradient = (f(xplus)[0] - f(xminus)[0]) / (2 * h)
        # END YOUR CODE

        # Compare gradients
        reldiff = abs(numeric_gradient - grad[ix]) / max(1, abs(numeric_gradient), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numeric_gradient)
            return

        it.iternext()  # Step to next index

    print "Gradient check passed!"


def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print "Running sanity checks..."
    gradient_check(quad, np.array(123.456))  # scalar test
    gradient_check(quad, np.random.randn(3, ))  # 1-D test
    gradient_check(quad, np.random.randn(4, 5))  # 2-D test
    print ""


if __name__ == '__main__':
    # If these fail, your code is definitely wrong.
    # sanity_check()
    from ex_3 import MLP1, ReLU

    net = MLP1(7, ReLU, 3, 9)
    W1, b1, W2, b2 = net.W1, net.b1, net.W2, net.b2


    def _loss_and_W2_grad(W2):
        net.W2 = W2
        loss, gW1, gb1, gW2, gb2 = net.backprop(np.array([1, 2, 3]), 0)
        return loss, gW2


    def _loss_and_W1_grad(W1):
        net.W1 = W1
        loss, gW1, gb1, gW2, gb2 = net.backprop(np.array([1, 2, 3]), 0)
        return loss, gW1


    def _loss_and_b1_grad(b1):
        net.b1 = b1
        loss, gW1, gb1, gW2, gb2 = net.backprop(np.array([1, 2, 3]), 0)
        return loss, gb1


    def _loss_and_b2_grad(b2):
        net.b2 = b2
        loss, gW1, gb1, gW2, gb2 = net.backprop(np.array([1, 2, 3]), 0)
        return loss, gb2


    # for _ in xrange(10):
    #     W1 = np.random.randn(W1.shape[0], W1.shape[1])
    #     b1 = np.random.randn(b1.shape[0])
    #     W2 = np.random.randn(W2.shape[0], W2.shape[1])
    #     b2 = np.random.randn(b2.shape[0])
    #     loss, grads = loss_and_gradients(np.array([1, 2, 3]), 0, [W1, b1, W2, b2])

    gradient_check(_loss_and_W2_grad, W2)
    gradient_check(_loss_and_W1_grad, W1)
    gradient_check(_loss_and_b1_grad, b1)
    gradient_check(_loss_and_b2_grad, b2)