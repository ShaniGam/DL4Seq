from dynet import *
import numpy as np
import random
import matplotlib.pyplot as plt

START = '-START-'
END = '-END-'
UNIQUE = 'UUUNKKK'


# create a class encapsulating the network
class OurNetwork(object):
    # The init method adds parameters to the model.
    def __init__(self, m, num_of_words, num_of_labels, window_vec_size, hidden_size, vec_dim):
        self.pW = m.add_parameters((hidden_size, window_vec_size))
        self.pU = m.add_parameters((num_of_labels, hidden_size))
        self.pB = m.add_parameters(hidden_size)
        self.pB_tag = m.add_parameters(num_of_labels)
        self.lookup = m.add_lookup_parameters((num_of_words, vec_dim))

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

# Find the words that appeared once in the text
# file_name - the name of the train file
def rare_words(file_name):
    words = {}
    with open(file_name, 'r') as data_file:
        for line in data_file:
            # get the word and label from the not empty lines
            try:
                text, label = line.strip().split()
            except ValueError:
                continue
            if text not in words:
                words[text] = 1
            else:
                words[text] += 1
    data_file.close()
    # return the word that appeared less than twice in the text
    return [k for k, v in words.iteritems() if v < 2]


# Read the data from the train/dev file
def read_data(file_name):
    data = []
    for i in range(2):
        data.append((START, START))
    with open(file_name, 'r') as data_file:
        for line in data_file:
            # pad every sentence with 'START' and 'END'
            if line == '\n':
                for i in range(2):
                    data.append((END, END))
                for i in range(2):
                    data.append((START, START))
                continue
            text, label = line.strip().split()
            data.append((text, label))
    data_file.close()
    return data


# Create graph from a list of x points and y points
def create_graph(info_map, name):
    keys = info_map.keys()
    values = info_map.values()
    plt.plot(keys, values)
    plt.savefig(name + '.png', bbox_inches='tight')
    plt.clf()


# Train the network on the train data and calculate the accuracy using the dev data
# train_data - the data we get from the train file
# dev_data - the data we get from the dev file
# F2I - the different kinds of features
# L2I - the different kinds of labels
# network - an instance of OurNetwork
# iterations - number of epochs
# rare - rare words
# accuracy_func - the function we use to calculate the accuracy
def train_network(train_data, dev_data, F2I, L2I, network, iterations, rare, accuracy_func, trainer, test_input, test_output, m):
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
                    current_vec = F2I[current_word]
                else:
                    current_vec = F2I[UNIQUE]
            else:
                current_vec = F2I[current_word]
            # get the window and label of the word
            inp = [F2I[train_data[i][0]], F2I[train_data[i + 1][0]], current_vec,
                           F2I[train_data[i + 3][0]], F2I[train_data[i + 4][0]]]

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


# Return the word if exists in the train file and the UNIQUE word otherwise
# feature - the current word
# F2I - the different kinds of features
def get_val(F2I, feature):
    try:
        return F2I[feature]
    except:
        return F2I[UNIQUE]


# Get the prediction of the current word on the dev data
# i - the current iteration
# dev_data - the data we get from the dev file
# F2I - the different kinds of features
# L2I - the different kinds of labels
# network - an instance of OurNetwork
def test_prediction(i, dev_data, F2I, L2I, network):
    inp = [get_val(F2I, dev_data[i][0]), get_val(F2I, dev_data[i+1][0]), get_val(F2I, dev_data[i+2][0]),
           get_val(F2I, dev_data[i+3][0]), get_val(F2I, dev_data[i+4][0])]
    lbl = L2I[dev_data[i + 2][1]]
    # get the loss and prediction of the window and label of the word
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


# Read the data from the test file
def read_test(input_file):
    data = []
    for i in range(2):
        data.append(START)
    with open(input_file, 'r') as data_file:
        for line in data_file:
            # pad every sentence with 'START' and 'END'
            if line == '\n':
                for i in range(2):
                    data.append(END)
                for i in range(2):
                    data.append(START)
                continue
            text = line.strip()
            data.append(text)
    data_file.close()
    return data


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
            inp = [get_val(F2I, data[i]), get_val(F2I, data[i + 1]), get_val(F2I, data[i + 2]),
                   get_val(F2I, data[i + 3]), get_val(F2I, data[i + 4])]
            # get the prediction of the window of the word
            prediction = network.create_network_return_best(inp)
            # find the label of the prediction
            label = L2I.keys()[L2I.values().index(prediction)]
            output.write(label)
            output.write('\n')
    output.close()


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
    num_of_labels = len(L2I)
    vec_dim = 50
    # create a network
    network = OurNetwork(m, len(F2I), num_of_labels, vec_dim * 5, hidden_size, vec_dim)
    train_network(train_data, dev_data, F2I, L2I, network, iterations, rare, accuracy_func, trainer, test_input, test_output, m)

if __name__ == '__main__':
    # create model
    m = Model()

    # create trainer
    trainer = SimpleSGDTrainer(m)
    read_and_train("ner/train", "ner/dev", 100, 50, accuracy_ner, m, trainer, 'ner/test', 'test1.ner')
    read_and_train("pos/train", "pos/dev", 50, 50, accuracy_pos, m, trainer, 'pos/test', 'test1.pos')
