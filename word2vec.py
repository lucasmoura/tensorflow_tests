from collections import Counter

import tensorflow as tf
import pickle
import os

VOCABULARY_PATH = 'vocabulary.pkl'
COUNT_PATH = 'count.pkl'
DICTIONARY_PATH = 'dictionary.pkl'
REVERSED_DICTIONARY_PATH = 'reversed_dictionary.pkl'
DATA_PATH = 'data.pkl'


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


def main():
    vocabulary_size = 50000

    vocabulary = prepare_data('text8', verbose=True)
    data, count, dictionary, reversed_dictionary = prepate_dataset(
        vocabulary, vocabulary_size)


if __name__ == '__main__':
    main()
