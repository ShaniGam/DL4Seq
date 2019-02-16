import numpy as np
import collections
from tagger2 import read_vectors
from tagger1 import read_data, UNIQUE, train_network, rare_words, START, END,create_graph, read_test
from dynet import *
import random

# create a class encapsulating the network
class OurNetwork(object):
    # The init method adds parameters to the model.
    # dict - use or not use the pre-trained vectors
    def __init__(self, m, num_of_words, num_of_labels, e_vec_size, hidden_size, vec_dim, e_array, dict):
        self.pW = m.add_parameters((hidden_size, e_vec_size))
        self.pU = m.add_parameters((num_of_labels, hidden_size))
        self.pB = m.add_parameters(hidden_size)
        self.pB_tag = m.add_parameters(num_of_labels)
        self.lookup = m.add_lookup_parameters((num_of_words, vec_dim))
        if dict:
            self.lookup.init_from_array(e_array)

    # the __call__ method applies the network to an input
    def __call__(self, inputs):
        W = parameter(self.pW)
        b = parameter(self.pB)
        U = parameter(self.pU)
        b_tag = parameter(self.pB_tag)
        lookup = self.lookup
        emb_inputs = []
        for inp in inputs:
            emb_vectors = [lookup[j] for j in inp]
            ex = emb_vectors[0]
            for e in emb_vectors[1:]:
                ex += e
            emb_inputs.append(ex)
        net_input = concatenate(emb_inputs)
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

# Gets the vector of all the features of the word
# word - the current word
# F2I - the map of features
def get_vector(word, F2I):
    word_vec = []
    if word != START and word != END and len(word) >= 3:
        # gets the prefix and suffix
        prefix = 'p_' + word[:3]
        suffix = 's_' + word[(len(word) - 3):]
        if prefix in F2I:
            word_vec.append(F2I[prefix])
        if suffix in F2I:
            word_vec.append(F2I[suffix])
    if word in F2I:
        word_vec.append(F2I[word])
    return word_vec


# Train the network on the train data and calculate the accuracy using the dev data
# train_data - the data we get from the train file
# dev_data - the data we get from the dev file
# F2I - the different kinds of features
# L2I - the different kinds of labels
# network - an instance of OurNetwork
# iterations - number of epochs
# rare - rare words
# accuracy_func - the function we use to calculate the accuracy
def train_network(train_data, dev_data, F2I, L2I, network, iterations, rare, accuracy_func, trainer, test_input, test_output):
    train_len = len(train_data) - 4
    prob = 0.5
    loss_map = {}
    accuracy_map = {}
    max_accuracy = 0.0
    for epoch in xrange(iterations):
        for i in range(train_len):
            # check if the current word is one of the words we used for the padding
            if train_data[i+2][0] == START or train_data[i+2][0] == END:
                continue
            # replace the rare word in some probability
            r = random.random()
            current_word = train_data[i+2][0]
            if current_word in rare:
                if r > prob:
                    current_vec = get_vector(train_data[i+2][0], F2I)
                else:
                    current_vec = [F2I[UNIQUE]]
            else:
                current_vec = get_vector(train_data[i+2][0], F2I)
            # get the window and label of the word
            inp = [get_vector(train_data[i][0], F2I), get_vector(train_data[i+1][0], F2I), current_vec,
                   get_vector(train_data[i+3][0], F2I), get_vector(train_data[i+4][0], F2I)]
            lbl = L2I[train_data[i+2][1]]
            # train the network using SGD
            loss = network.create_network_return_loss(inp, lbl)
            loss.value()  # need to run loss.value() for the forward prop
            loss.backward()
            trainer.update()
        # get the loss and accuracy on the dev
        loss, accuracy = accuracy_func(dev_data, L2I, F2I, network)
        print accuracy
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            m.save("model", network.return_model())
        loss_map[epoch] = loss
        accuracy_map[epoch] = accuracy
    network.restore_components(m.load("model"))
    predict_test(test_input, test_output, F2I, L2I, network)
    # create a graphs representing the results
    create_graph(loss_map, test_output + '_loss')
    create_graph(accuracy_map, test_output + '_accuracy')


# Return the features vector of the word if exists in the train file and the UNIQUE word otherwise
# word - the current word
# F2I - the different kinds of features
def get_val(word, F2I):
    features = get_vector(word, F2I)
    if features == []:
        features = [F2I[UNIQUE]]
    return features


# Get the prediction of the current word on the dev data
# i - the current iteration
# dev_data - the data we get from the dev file
# F2I - the different kinds of features
# L2I - the different kinds of labels
# network - an instance of OurNetwork
def test_prediction(i, dev_data, F2I, L2I, network):
    inp = [get_val(dev_data[i][0], F2I), get_val(dev_data[i+1][0], F2I), get_val(dev_data[i+2][0], F2I),
           get_val(dev_data[i+3][0], F2I), get_val(dev_data[i+4][0], F2I)]
    lbl = L2I[dev_data[i + 2][1]]
    loss, prediction = network.create_network_return_best_with_loss(inp, lbl)
    return loss, prediction, lbl


# Calculate the accuracy on the dev_data of POS
# F2I - the different kinds of features
# L2I - the different kinds of labels
# network - an instance of OurNetwork
def accuracy_pos(dev_data, L2I, F2I, network):
    good = bad = 0.0
    dev_len = len(dev_data) - 4
    loss_sum = 0.0
    for i in range(dev_len):
        if dev_data[i + 2][0] == START or dev_data[i + 2][0] == END:
            continue
        loss, prediction, lbl = test_prediction(i, dev_data, F2I, L2I, network)
        loss_sum += loss
        # check if the prediction was correct
        if prediction == lbl:
            good += 1
        else:
            bad += 1
    loss_avg = loss_sum / (good + bad)
    print "loss: " + str(loss_avg)
    return loss_avg, (good / (good + bad))


# Calculate the accuracy on the dev_data of NER
# F2I - the different kinds of features
# L2I - the different kinds of labels
# network - an instance of OurNetwork
def accuracy_ner(dev_data, L2I, F2I, network):
    good = bad = 0.0
    dev_len = len(dev_data) - 4
    loss_sum = 0.0
    words_counter = 0
    for i in range(dev_len):
        if dev_data[i + 2][0] == START or dev_data[i + 2][0] == END:
            continue
        loss, prediction, lbl = test_prediction(i, dev_data, F2I, L2I, network)
        words_counter += 1
        # check if the prediction was correct
        if prediction == lbl:
            # check only of the label wasn't 'O'
            if lbl != L2I['O']:
                good += 1
        else:
            bad += 1
        loss_sum += loss
    loss_avg = loss_sum / words_counter
    print "loss: " + str(loss_avg)
    return loss_avg, (good / (good + bad))


# Write the predictions of the test to a file
def predict_test(input_file, output_file, F2I, L2I, network):
    data = read_test(input_file)
    new_line = True
    data_len = len(data) - 4
    with open(output_file, 'w+') as output:
        for i in range(data_len):
            if data[i + 2] == START:
                new_line = True
                continue
            if data[i + 2] == END:
                if new_line:
                    output.write('\n')
                new_line = False
                continue
            inp = [get_val(data[i], F2I), get_val(data[i + 1], F2I), get_val(data[i + 2], F2I),
                   get_val(data[i + 3], F2I), get_val(data[i + 4], F2I)]
            # get the prediction of the window of the word
            prediction = network.create_network_return_best(inp)
            # find the label of the prediction
            label = L2I.keys()[L2I.values().index(prediction)]
            output.write(label)
            output.write('\n')
    output.close()


# Check for every word if it exists in the dictionary given
# F2I - the words exist in the train file
# words - the words from the vocabulary
def create_embedding(F2I, words):
    words_vec = {}
    F2I_ex = dict(F2I)
    for word in F2I:
        if word in words:
            words_vec[F2I[word]] = words[word]
        elif word.lower() in words:
            words_vec[F2I[word]] = words[word.lower()]
        else:
            words_vec[F2I[word]] = np.random.randn(50)
        word_len = len(word)
        if len(word) >= 3:
            prefix = 'p_' + word[:3]
            suffix = 's_' + word[(word_len - 3):]
            if prefix not in words_vec:
                words_vec[prefix] = np.random.randn(50)
                F2I_ex[prefix] = len(F2I_ex)
            if suffix not in words_vec:
                words_vec[suffix] = np.random.randn(50)
                F2I_ex[suffix] = len(F2I_ex)
    return F2I_ex, collections.OrderedDict(sorted(words_vec.items()))


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
    F2I, words_vec = create_embedding(F2I, words)
    num_of_labels = len(L2I)
    vec_dim = 50
    # create a network
    network = OurNetwork(m, len(F2I), num_of_labels, vec_dim * 5, hidden_size, vec_dim, np.asarray(words_vec.values()), True)
    train_network(train_data, dev_data, F2I, L2I, network, iterations, rare, accuracy_func, trainer, test_input, 'dict_' + test_output)
    network = OurNetwork(m, len(F2I), num_of_labels, vec_dim * 5, hidden_size, vec_dim, np.asarray(words_vec.values()), False)
    train_network(train_data, dev_data, F2I, L2I, network, iterations, rare, accuracy_func, trainer, test_input, 'no_dict_' + test_output)

if __name__ == '__main__':

    vecs = np.loadtxt("wordVectors.txt")
    words = read_vectors("vocab.txt", vecs)
    # create model
    m = Model()
    trainer = SimpleSGDTrainer(m)
    read_and_train("ner/train", "ner/dev", 100, 50, accuracy_ner, m, trainer, 'ner/test', 'test4.ner')
    read_and_train("pos/train", "pos/dev", 50, 50, accuracy_pos, m, trainer, 'pos/test', 'test4.pos')
