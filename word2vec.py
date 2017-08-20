from collections import Counter

import tensorflow as tf
import numpy as np

import collections
import pickle
import random
import os

VOCABULARY_PATH = 'vocabulary.pkl'
COUNT_PATH = 'count.pkl'
DICTIONARY_PATH = 'dictionary.pkl'
REVERSED_DICTIONARY_PATH = 'reversed_dictionary.pkl'
DATA_PATH = 'data.pkl'

data_index = 0


def pickle_dump(filename, data):
    with open(filename, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)


def pickle_load(filename):
    with open(filename, 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    return data


def read_text_data(filename):
    if os.path.exists(VOCABULARY_PATH):
        data = pickle_load(VOCABULARY_PATH)
    else:
        with open(filename, 'r') as text_file:
            data = tf.compat.as_str(text_file.read()).split()

        pickle_dump(VOCABULARY_PATH, data)

    return data


def prepare_data(filename, verbose=False, num_examples=5):
    vocabulary = read_text_data('text8')

    if verbose:
        print('Number of tokens: {}'.format(len(vocabulary)))
        print('Display some examples:\n')

        for i in range(num_examples):
            print(vocabulary[i])

        print()

    return vocabulary


def create_dataset(vocabulary, num_words):
    # Count the frequency of words in our vocabulary
    count = [['UNK', -1]]
    count.extend(Counter(vocabulary).most_common(num_words - 1))

    # Word to index dict
    dictionary = dict()
    for index, freq in enumerate(count):
        dictionary[freq[0]] = index

    unk_count = 0
    data = list()

    # Turn vocabulary into a list of index
    for word in vocabulary:
        index = dictionary.get(word, 0)

        if not index:  # dictionary['UNK']
            unk_count += 1

        data.append(index)

    count[0][1] = unk_count
    # Index to word dict
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, count, dictionary, reversed_dictionary


def prepate_dataset(vocabulary, num_words):
    file_paths = [COUNT_PATH, DATA_PATH, DICTIONARY_PATH,
                  REVERSED_DICTIONARY_PATH]
    dataset_available = True

    for dataset_file in file_paths:
        if not os.path.exists(dataset_file):
            dataset_available = False
            break

    if not dataset_available:
        data, count, dictionary, reversed_dictionary = create_dataset(
            vocabulary, num_words)

        pickle_dump(DATA_PATH, data)
        pickle_dump(COUNT_PATH, count)
        pickle_dump(DICTIONARY_PATH, dictionary)
        pickle_dump(REVERSED_DICTIONARY_PATH, reversed_dictionary)
    else:
        data = pickle_load(DATA_PATH)
        count = pickle_load(COUNT_PATH)
        dictionary = pickle_load(DICTIONARY_PATH)
        reversed_dictionary = pickle_load(REVERSED_DICTIONARY_PATH)

    return data, count, dictionary, reversed_dictionary


def generate_batch(data, batch_size, num_skips, window_size):
    global data_index

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * window_size + 1  # Number of words in a context

    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0

    buffer.extend(data[data_index:data_index + span])
    data_index += span

    """
    The batch will be composed by some repeated words,
    for example, if the window size is 1, the input will appear
    in the batch two times, one for the left word and the other
    for the right word.
    """
    for i in range(batch_size // num_skips):
        target = window_size

        context_words = [i for i in range(span) if i != target]
        random.shuffle(context_words)
        targets_to_use = collections.deque(context_words)

        for j in range(num_skips):
            batch[i * num_skips + j] = buffer[target]
            context = targets_to_use.pop()
            labels[i * num_skips + j, 0] = buffer[context]

        if data_index == len(data):
            buffer[:] = data[:span]
            data_index = span
        else:
            """
            Since we are using a deque data structure with maxlen
            specified, once we add a new data, the index in the deque
            head will be popped, and the new index will be added at
            the deque tail.
            """
            buffer.append(data[data_index])
            data_index += 1

    """
    Suppose we have we have a batch_size of 2 and a num_skips
    value of 1. At the end of the for loop, data_index will be pointing
    at 4. Therefore, onde we will generate a new batch, buffer would
    look only for data[4: 4 + span], missing some word. This next line
    regularize the data_index to not miss words when the next batch is
    generated.
    """
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


def main():
    vocabulary_size = 50000
    window_size = 1
    num_skips = window_size * 2
    batch_size = 64

    vocabulary = prepare_data('text8', verbose=True)
    data, count, dictionary, reversed_dictionary = prepate_dataset(
        vocabulary, vocabulary_size)

    del vocabulary
    print('Most common words (+UNK): ', count[:5])
    print('Sample data', data[:10],
          [reversed_dictionary[i] for i in data[:10]])

    batch, labels = generate_batch(data, batch_size, num_skips, window_size)


if __name__ == '__main__':
    main()
