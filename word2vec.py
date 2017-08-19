import tensorflow as tf


def read_text_data(filename):
    with open(filename, 'r') as text_file:
        data = tf.compat.as_str(text_file.read()).split()

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


def main():
    vocabulary = prepare_data('text8', verbose=True)



if __name__ == '__main__':
    main()
