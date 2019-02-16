import numpy as np


# Maps every word to its vector
# vocab_file - the name of the vocabulary
# vecs - the given vectors
def read_data(vocab_file, vecs):
    with open(vocab_file, 'r') as vocab:
        temp = [line[:-1] for line in vocab]
        words_vecs = dict(zip(temp, vecs))
        return words_vecs


# The k nearest neighbors to the word
def most_similar(word, k):
    knn = []
    for i in range(k):
        max_val = 0.0
        closest_word = words_vecs.keys()[0]
        for w in words_vecs:
            if w != word and w not in knn:
                # gets the vector of the word
                input_vec = words_vecs[word]
                w_vec = words_vecs[w]
                # calculate the cos of the word and the current feature
                dist = input_vec.dot(w_vec) / np.dot(np.sqrt(input_vec.dot(input_vec)), np.sqrt(w_vec.dot(w_vec)))
                # find max
                if dist > max_val:
                    max_val = dist
                    closest_word = w
        print closest_word, max_val
        knn.append(closest_word)
    return knn

if __name__ == '__main__':
    vecs = np.loadtxt("wordVectors.txt")
    words_vecs = read_data("vocab.txt", vecs)
    words = ['dog', 'england', 'john', 'explode', 'office']
    for word in words:
        print most_similar(word, 5)