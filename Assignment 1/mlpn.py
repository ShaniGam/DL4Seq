import numpy as np
import loglinear as ll
import mlp1 as ml

hidden_layers = []


def classifier_output(x, params):
    """
    Return the output layer (class probabilities)
    of a mlpn classifier with given params on input x.
    """
    temp_x = x
    params_len = len(params)
    # take all the parameters except the last two
    prams = params[:(params_len - 2)]
    # calculate the tanh of W,b and add the result to a list of hidden layers
    for W, b in zip(prams[0::2], prams[1::2]):
        temp_x = np.tanh(np.dot(temp_x, W) + b)
        hidden_layers.append(temp_x)
    # compute the softmax with the given parameters
    probs = ll.softmax(np.dot(temp_x.T, params[params_len - 2]) + params[params_len - 1])
    return probs


def predict(x, params):
    """
    Returns the prediction (highest scoring class id) of a
    mlpn classifier with given parameters on input x.
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    Compute the loss and the gradients at point x with given parameters.
    y is a scalar indicating the correct label.

    returns:
        loss, gradients
    """
    x = np.array(x)
    class_result = classifier_output(x, params)

    # creates a y vector with 1 in the given y index
    y_vec = np.zeros(len(class_result))
    y_vec[y] = 1

    # the loss with the given parameters
    loss = -1 * np.log(class_result[y])

    # creates the W matrix and the b vector
    W_matrix = []
    b_vector = hidden_layers[::-1]
    for i in range(len(params) - 2, -1, -2):
        W_matrix.append(params[i])

    gradients = []
    W_len = len(W_matrix)
    for i in range(W_len):
        # the computed y minus the true y
        sub_result = (class_result - y_vec).reshape(-1, 1)
        index = 0
        if i != index:
            sub_result = sub_result.T.dot((W_matrix[index]).T * ml.der_tanh(b_vector[index])).T
            index += 1
        # the gradient of the b vactor
        gb = sub_result
        # the gradient of the W matrix
        gW = np.dot(sub_result, b_vector[index].reshape(-1, 1).T) \
            if i != W_len - 1 \
            else np.dot(sub_result, x.reshape(-1, 1).T)
        gW = gW.T
        # add the gradients to the list
        gradients.append(gb)
        gradients.append(gW)

    gradients = gradients[::-1]
    return loss, gradients


def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """

    params = []
    dims_len = len(dims)
    # create W and b and add them to the params list
    for i in range(dims_len - 1):
        W_matrix=(np.zeros((dims[i], dims[i + 1])))
        b_vector=(np.zeros(dims[i + 1]))
        params.append(W_matrix)
        params.append(b_vector)

    return params
