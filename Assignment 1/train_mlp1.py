import mlp1 as mlp
import random
import numpy as np
from utils import F2I, L2I, TRAIN as train_data, DEV as dev_data

def feats_to_vec(features):
    # Should return a numpy vector of features.
    features_list = np.zeros(len(F2I))
    # creates list with a counter for every letter bigram in exists
    for feature in features:
        if feature in F2I:
            # increase the counter by one of the letter bigram appears on features
            features_list[F2I[feature]] += 1.0

    # normalize the features values
    features_list /= features_list.sum()
    return np.asarray(features_list)

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        x = feats_to_vec(features)
        # get the prediction for the given x and parameters
        y_hat = mlp.predict(x, params)
        if y_hat == L2I[label]:
            good += 1
        else:
            bad += 1

        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
    return good / (good + bad)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """

    for I in xrange(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = L2I[label]                  # convert the label to number if needed.

            # update the parameters according to the gradients
            # and the learning rate.
            loss, grads = mlp.loss_and_gradients(x,y,params)
            cum_loss += loss
            params_len = len(params)
            for i in range(params_len):
                params[i] -= grads[i] * learning_rate

        # calculate the loss and the accuracies.
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params

if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    hid_dim = 100
    params = mlp.create_classifier(len(F2I), hid_dim, len(L2I))
    U, W, b, b_tag = params

    # give the parameters random values to get better results
    U = np.random.randn(U.shape[0],U.shape[1])
    W = np.random.randn(W.shape[0], W.shape[1])
    b = np.random.randn(b.shape[0])
    b_tag=np.random.randn(b_tag.shape[0])

    params = [U, W, b, b_tag]
    trained_params = train_classifier(train_data, dev_data, 100, 0.1, params)
