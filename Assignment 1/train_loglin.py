import loglinear as ll
import random
import numpy as np
from utils import F2I, L2I, TRAIN as train_data, DEV as dev_data

def feats_to_vec(features):
    # Should return a numpy vector of features.

    # creates list with a counter for every letter bigram in exists
    features_list = np.zeros(len(F2I))
    for feature in features:
        if feature in F2I:
            # increase the counter by one of the letter bigram appears on features
            features_list[F2I[feature]] += 1

    # normalize the features values
    features_list /= features_list.sum()
    return np.asarray(features_list)

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        x = feats_to_vec(features)
        # get the prediction for the given x and parameters
        y_hat = ll.predict(x, params)
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
            loss, grads = ll.loss_and_gradients(x,y,params)

            # update the parameters according to the gradients
            # and the learning rate.
            cum_loss += loss
            params[0] -= grads[0] * learning_rate
            params[1] -= grads[1] * learning_rate

        # calculate the loss and the accuracies.
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params

trained_params = []

if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    params = ll.create_classifier(len(F2I), len(L2I))
    trained_params = train_classifier(train_data, dev_data, 100, 0.1, params)
