from dynet import *
import numpy as np
import collections
from tagger1 import read_data, accuracy_ner, accuracy_pos, UNIQUE, train_network, rare_words, START, END


# create a class encapsulating the network
class OurNetwork(object):
    # The init method adds parameters to the model.
    def __init__(self, m, num_of_words, num_of_labels, e_vec_size, hidden_size, vec_dim, e_array):
        self.pW = m.add_parameters((hidden_size, e_vec_size))
        self.pU = m.add_parameters((num_of_labels, hidden_size))
        self.pB = m.add_parameters(hidden_size)
        self.pB_tag = m.add_parameters(num_of_labels)
        self.lookup = m.add_lookup_parameters((num_of_words, vec_dim))
        self.lookup.init_from_array(e_array)

    # the __call__ method applies the network to an input
    def __call__(self, inputs):
        W = parameter(self.pW)
        b = parameter(self.pB)
        U = parameter(self.pU)
        b_tag = parameter(self.pB_tag)
        lookup = self.lookup
        emb_vectors = [lookup[i] for i in inputs]
        net_input = concatenate(emb_vectors)
        net_output = softmax(U * tanh((W * net_input) + b) + b_tag)
        return net_output

    # function for graph creation
    def create_network_return_loss(self, inputs, expected_output):
        """
            inputs is a list of numbers
        """
        renew_cg()
        out = self(inputs)
        loss = -log(pick(out, expected_output))
        return loss

    # function for prediction
    def create_network_return_best_with_loss(self, inputs, expected_output):
        """
            inputs is a list of numbers
        """
        renew_cg()
        out = self(inputs)
        loss = -log(pick(out, expected_output))
        return loss.value(), np.argmax(out.npvalue())

    # function for prediction
    def create_network_return_best(self, inputs):
        """
            inputs is a list of numbers
        """
        renew_cg()
        out = self(inputs)
        return np.argmax(out.npvalue())

    def return_model(self):
        return (self.pW, self.pU, self.pB, self.pB_tag, self.lookup)

    def restore_components(self, components):
        self.pW, self.pU, self.pB, self.pB_tag, self.lookup = components


# Maps every word to its vector
# vocab_file - the name of the vocabulary
# vecs - the given vectors
def read_vectors(vocab_file, vecs):
    with open(vocab_file, 'r') as vocab:
        temp = [line[:-1] for line in vocab]
        words_vecs = dict(zip(temp, vecs))
        return words_vecs


# Check for every word if it exists in the dictionary given
# F2I - the words exist in the train file
# words - the words from the vocabulary
def create_embedding(F2I, words):
    words_vec = {}
    for word in F2I:
        if word in words:
            words_vec[F2I[word]] = words[word]
        elif word.lower() in words:
            words_vec[F2I[word]] = words[word.lower()]
        else:
            words_vec[F2I[word]] = np.random.randn(50)
    return collections.OrderedDict(sorted(words_vec.items()))


# Read the data and train it
# train_path - the path of the train file
# dev_path - the path of the dev file
# vec_dim - the size of the hidden layer
# itrations - number of epochs
# accuracy_func - the function we use to calculate the accuracy
def read_and_train(train_path, dev_path, hidden_size, iterations, accuracy_func, m, trainer, test_input, test_output):
    # get the rare words from the train file
    rare = rare_words(train_path)
    train_data = read_data(train_path)
    dev_data = read_data(dev_path)
    #    dev_data = read_data("pos/dev", data)
    # label strings to IDs

    L2I = {l: i for i, l in enumerate(list(sorted(set([l for t, l in train_data if l != START and l != END]))))}

    # feature strings (bigrams) to IDs
    F2I = {f: i for i, f in enumerate(list(sorted(set([t for t, l in train_data]))))}

    num_of_words = len(F2I)
    F2I[UNIQUE] = num_of_words
    words_vec = create_embedding(F2I, words)
    num_of_labels = len(L2I)
    vec_dim = 50
    # create a network
    network = OurNetwork(m, len(F2I), num_of_labels, vec_dim * 5, hidden_size, vec_dim, np.asarray(words_vec.values()))
    train_network(train_data, dev_data, F2I, L2I, network, iterations, rare, accuracy_func, trainer, test_input, test_output, m)

if __name__ == '__main__':
    vecs = np.loadtxt("wordVectors.txt")
    words = read_vectors("vocab.txt", vecs)
    # create model
    m = Model()
    trainer = SimpleSGDTrainer(m)

    read_and_train("ner/train", "ner/dev", 100, 50, accuracy_ner, m, trainer, 'ner/test', 'test3.ner')
    read_and_train("pos/train", "pos/dev", 50, 50, accuracy_pos, m, trainer, 'pos/test', 'test3.pos')