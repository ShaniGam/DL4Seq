import numpy as np

def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    # Your code should be fast, so use a vectorized implementation using numpy,
    # don't use any loops.
    # With a vectorized implementation, the code should be no more than 2 lines.
    #
    # For numeric stability, use the identify you proved in Ex 2 Q1.

    # subtract the maximum from every x to prevent large values
    ex = np.exp(x - np.max(x))
    x = ex / ex.sum(axis=0)

    return x
    

def classifier_output(x, params):
    """
    Return the output layer (class probabilities) 
    of a log-linear classifier with given params on input x.
    """
    W,b = params

    # compute the softmax with the given parameters
    probs = softmax(np.dot(x, W) + b)
    return probs

def predict(x, params):
    """
    Returnss the prediction (highest scoring class id) of a
    a log-linear classifier with given parameters on input x.
    """
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    Compute the loss and the gradients at point x with given parameters.
    y is a scalar indicating the correct label.

    returns:
        loss,[gW,gb]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    """
    # initialize the W matrix
    W,b = params
    len_row = W[:, 0].size
    len_col = W[0].size
    gW = np.zeros((len_row, len_col))

    # loss calculation
    class_result = classifier_output(x, params)
    loss = -1 * np.log(class_result[y])

    # gradient of every element in the W matrix
    for i in range(len_row):
        for j in range(len_col):
            gW[i,j] = x[i] * (-1 * ((y == j) - class_result[j]))

    # gradient of the b vactor
    y_vec = np.zeros(len(class_result))
    y_vec[y] = 1
    gb = class_result - y_vec
    return loss,[gW,gb]

def create_classifier(in_dim, out_dim):
    """
    returns the parameters (W,b) for a log-linear classifier
    with input dimension in_dim and output dimension out_dim.
    """
    W = np.zeros((in_dim, out_dim))
    b = np.zeros(out_dim)
    return [W,b]

if __name__ == '__main__':
    # Sanity checks for softmax. If these fail, your softmax is definitely wrong.
    # If these pass, it may or may not be correct.
    test1 = softmax(np.array([1,2]))
    print test1
    assert np.amax(np.fabs(test1 - np.array([0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([1001,1002]))
    print test2
    assert np.amax(np.fabs(test2 - np.array( [0.26894142, 0.73105858]))) <= 1e-6

    test3 = softmax(np.array([-1001,-1002])) 
    print test3 
    assert np.amax(np.fabs(test3 - np.array([0.73105858, 0.26894142]))) <= 1e-6

    test4 = softmax(np.array([0.1, 0.2]))
    print test4
    assert np.amax(np.fabs(test4 - np.array([0.47502081,  0.52497919]))) <= 1e-6

    test5 = softmax(np.array([-0.1, 0.2]))
    print test5
    assert np.amax(np.fabs(test5 - np.array([0.42555748,  0.57444252]))) <= 1e-6

    test6 = softmax(np.array([0.9, -10]))
    print test6
    assert np.amax(np.fabs(test6 - np.array([9.99981542e-01, 1.84578933e-05]))) <= 1e-6

    test7 = softmax(np.array([0, 10]))
    print test7
    assert np.amax(np.fabs(test7 - np.array([4.53978687e-05, 9.99954602e-01]))) <= 1e-6


    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W,b = create_classifier(3,4)

    def _loss_and_W_grad(W):
        global b
        loss,grads = loss_and_gradients([1,2,3],0,[W,b])
        return loss,grads[0]

    def _loss_and_b_grad(b):
        global W
        loss,grads = loss_and_gradients([1,2,3],0,[W,b])
        return loss,grads[1]

    for _ in xrange(10):
        W = np.random.randn(W.shape[0],W.shape[1])
        b = np.random.randn(b.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
