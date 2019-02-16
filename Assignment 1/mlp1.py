import numpy as np
from loglinear import softmax

def der_tanh(x):
    # return the derivative of tanh
    return 1 - (np.tanh(x)**2)

def classifier_output(x, params):
    """
    Return the output layer (class probabilities)
    of a mlp classifier with given params on input x.
    """
    U, W, b, b_tag = params
    # compute the softmax with the given parameters
    probs = softmax(np.tanh(x.dot(W) + b).dot(U) + b_tag)
    return probs

def predict(x, params):
    """
    Returnss the prediction (highest scoring class id) of a
    mlp classifier with given parameters on input x.
    """
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    Compute the loss and the gradients at point x with given parameters.
    y is a scalar indicating the correct label.

    returns:
        loss,[gU,gW,gb,gb_tag]

    loss: scalar
    gU: matrix, gradients of U
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gb_tag: vector, gradients of b_tag
    """
    U, W, b, b_tag = params
    x = np.array(x)
    class_result = classifier_output(x, params)

    # creates a y vector with 1 in the given y index
    y_vec = np.zeros(len(class_result))
    y_vec[y] = 1

    # the loss and gradients of the parameters
    gb_tag = class_result - y_vec
    sub_result = (class_result - y_vec).reshape(-1, 1)
    gb = np.dot(sub_result.T, U.T * (der_tanh(np.dot(x, W) + b)))[0]
    gW = np.dot(sub_result.T, U.T * (der_tanh(np.dot(x, W) + b))).T.dot(x.reshape(-1, 1).T).T
    gU = np.dot(sub_result, (np.tanh(np.dot(x, W) + b)).reshape(-1, 1).T).T
    loss = -1 * np.log(class_result[y])
    return loss, [gU, gW, gb, gb_tag]

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.
    """
    W = np.zeros((in_dim, hid_dim))
    b = np.zeros(hid_dim)
    b_tag = np.zeros(out_dim)
    U = np.zeros((hid_dim, out_dim))
    params = [U, W, b, b_tag]
    return params

if __name__ == '__main__':
    from grad_check import gradient_check

    U, W, b, b_tag = create_classifier(2,2,2)

    def _loss_and_W_grad(W):
        global U, b, b_tag
        loss,grads = loss_and_gradients([1,2],0,[U, W, b, b_tag])
        return loss,grads[1]

    def _loss_and_b_grad(b):
        global U, W, b_tag
        loss,grads = loss_and_gradients([1,2],0,[U, W, b, b_tag])
        return loss,grads[2]

    def _loss_and_U_grad(U):
        global W, b, b_tag
        loss,grads = loss_and_gradients([1,2],0,[U, W, b, b_tag])
        return loss,grads[0]

    def _loss_and_b_tag_grad(b_tag):
        global U, W, b
        loss,grads = loss_and_gradients([1,2],0,[U, W, b, b_tag])
        return loss,grads[3]

    for _ in xrange(10):
        W = np.random.randn(W.shape[0],W.shape[1])
        b = np.random.randn(b.shape[0])
        b_tag = np.random.randn(b_tag.shape[0])
        U = np.random.randn(U.shape[0], U.shape[1])
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_b_tag_grad, b_tag)
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_U_grad, U)
