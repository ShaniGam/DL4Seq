from dynet import *
import numpy as np
import time


# create a class encapsulating the network
class OurNetwork(object):
    # The init method adds parameters to the model.
    def __init__(self, vocab_size, label_size, input_dim_lstm, vec_dim, hidden_size, layers):
        self.lookup = m.add_lookup_parameters((vocab_size, vec_dim))
        self.pW = m.add_parameters((hidden_size, vec_dim))
        self.pU = m.add_parameters((label_size, hidden_size))
        self.pB = m.add_parameters(hidden_size)
        self.pB_tag = m.add_parameters(label_size)
        self.lstm = LSTMBuilder(layers, input_dim_lstm, vec_dim, m)

    # the __call__ method applies the network to an input
    def __call__(self, sentence):
        renew_cg()
        s0 = self.lstm.initial_state()
        W = parameter(self.pW)
        b = parameter(self.pB)
        U = parameter(self.pU)
        b_tag = parameter(self.pB_tag)
        lookup = self.lookup
        sentence = [F2I[c] for c in sentence]
        s = s0
        for char in sentence:
            s = s.add_input(lookup[char])
        probs = softmax(U * tanh((W * s.output()) + b) + b_tag)
        return probs

    # function for graph creation
    def create_network_return_loss(self, sentence, expected_output):
        """
            inputs is a list of numbers
        """
        renew_cg()
        out = self(sentence)
        loss = -log(pick(out, expected_output))
        return loss


    # function for prediction
    def create_network_return_best(self, inputs):
        """
            inputs is a list of numbers
        """
        renew_cg()
        out = self(inputs)
        return np.argmax(out.npvalue())


# reads the train/test file
# file_name - the name of the file
def read_data(file_name):
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            (label, text) = line.strip().split()
            data.append((text, label))
    f.close()
    return data


# calculate the accuracy on the test file
# network - the rnn network
# data - the test data
def accuracy(network, data):
    good = bad = 0.0
    for (text, lable) in data:
        # gets the prediction from the network
        prediction = network.create_network_return_best(text)
        # compares the prediction to the given label
        if prediction == L2I[lable]:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


# train the network with the given train and test data
# iterations - number of epochs
def train_network(iterations):
    trainer = AdagradTrainer(m)
    network = OurNetwork(len(F2I), len(L2I), 50, 50, 50, 2)
    # start measuring the train time
    startTime = time.clock()
    for i in xrange(iterations):
        for (text, label) in train:
            loss = network.create_network_return_loss(text, L2I[label])
            loss_value = loss.value()
            loss.backward()
            trainer.update()
        print time.clock() - startTime
        print "test: " + str(accuracy(network, test))


if __name__ == '__main__':
    train = read_data(sys.argv[1])
    test = read_data(sys.argv[2])

    # get the vocabulary
    L2I = {l: i for i, l in enumerate(list(sorted(set([l for t, l in train]))))}
    s = set()
    for (text, label) in train:
        s |= set(text)
    F2I = {f: i for i, f in enumerate(list(sorted(s)))}

    m = Model()
    train_network(200)
