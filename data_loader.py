import codecs
import collections
import csv
import json
import string
import sys

import numpy as np
from tensorflow import keras


def read_file(filename, encoding=None):
    """

    :param filename:    str
    :param encoding:    str
    :return:            str
    """

    if encoding is None:
        encoding = sys.getdefaultencoding()

    with codecs.open(filename, "r", encoding) as f:
        data = f.read()

    return data


def read_csv(filename, encoding=None, delimiter=",", quotechar="\""):
    """

    :param filename:    str
    :param encoding:    str
    :param delimiter:   str
    :param quotechar:   str
    :return:            list[list[str]]
    """
    if encoding is None:
        encoding = sys.getdefaultencoding()

    with codecs.open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
        data = [l for l in reader]

    return data


def read_json(filename, encoding=None):
    if encoding is None:
        encoding = sys.getdefaultencoding()

    with codecs.open(filename, "r", encoding) as f:
        data = json.load(f)

    return data


def write_file(filename, data, encoding=None):
    """

    :param filename:    str
    :param data:        str
    :param encoding:    str
    """
    if encoding is None:
        encoding = sys.getdefaultencoding()

    with codecs.open(filename, "w", encoding) as f:
        f.write(data)


def write_json(filename, data, encoding=None):
    write_file(filename, json.dumps(data), encoding)


def get_pad_dict():
    UNKNOWN_TOKEN = "<UNK>"
    UNKNOWN_INDEX = 0
    START_TOKEN = "<START>"
    START_INDEX = 1
    END_TOKEN = "<END>"
    END_INDEX = 2

    return {
        UNKNOWN_INDEX: UNKNOWN_TOKEN,
        START_INDEX: START_TOKEN,
        END_INDEX: END_TOKEN,
    }


def generate_indices(token_list, pad_tokens=True, max_indices=None):
    """

    :param token_list:  list
    :param pad_tokens:  bool
    :param max_indices: int
    :return:            dict[T: int], dict[int: T]
    """

    tokens_to_inds = {}
    inds_to_tokens = {}

    if pad_tokens:
        inds_to_tokens = get_pad_dict()

    counter = collections.Counter(token_list)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    pad = len(inds_to_tokens)

    for i, (t, _) in enumerate(count_pairs):
        tokens_to_inds[t] = i + pad
        inds_to_tokens[i + pad] = t

    if max_indices:
        i = max_indices
        while i in inds_to_tokens:
            token = inds_to_tokens[i]
            del tokens_to_inds[token]
            del inds_to_tokens[i]
            i += 1

    return tokens_to_inds, inds_to_tokens


def clean_data(data, lower=False, punctuation=False, whitespace=False, other=False, newline=False):
    """

    :param data:        str
    :param lower:       bool
    :param punctuation: bool
    :param whitespace:  bool
    :param other:       bool
    :param newline:     bool
    :return:            str
    """

    if lower:
        data = data.lower()

    if punctuation:
        for p in string.punctuation:
            data = data.replace(p, "")

    if newline:
        data = data.replace("\n", " ")

    if whitespace:
        # + non-breaking space
        for w in string.whitespace + " ":
            if w in (" ", "\n"):
                continue

            data = data.replace(w, " ")

        # Trim consecutive whitespace
        for _ in range(10):
            data = data.replace("  ", " ")

    if other:
        # Normalise punctuation
        for c in "“”":
            data = data.replace(c, "\"")
        for c in "‘’`":
            data = data.replace(c, "'")

        for c in "–—•":
            data = data.replace(c, "-")

        data = data.replace("…", "...")

        # Delete other characters
        for c in set(data):
            if c not in string.printable:
                data = data.replace(c, "")

    return data


def clean_with_vocab(data, w_i):
    # Delete characters not in vocab
    for c in set(data):
        if c not in w_i:
            data = data.replace(c, "")

    return data


def pretrained_embedding(filename, vocab, input_length, freeze=True):
    """
    Currently only works with json files

    :param filename:        str
    :param vocab:           dict[T: int]
    :param input_length:    int
    :param freeze:          bool
    :return:                keras.layers.Embedding
    """

    token_to_vec_map = read_json(filename)

    vocab_len = max(vocab.values()) + 1
    emb_dim = len(token_to_vec_map[list(vocab.keys())[0]])

    # Copy vectors to Embedding matrix
    emb_matrix = np.zeros((vocab_len, emb_dim))
    for token, index in vocab.items():
        try:
            emb_matrix[index, :] = token_to_vec_map[token]
        except KeyError:
            # Leave as zero-vector
            pass

    # Build embedding layer
    embedding_layer = keras.layers.Embedding(vocab_len, emb_dim, trainable=not freeze, input_length=input_length)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


class KerasBatchGenerator:
    def __init__(self, data, num_steps, batch_size, vocab_size, skip_step=1):
        """
        :param data:        list[int]
        :param num_steps:   int
        :param batch_size:  int
        :param vocab_size:  int
        :param skip_step:   int
        """
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocab_size))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = keras.utils.to_categorical(temp_y, num_classes=self.vocab_size)
                self.current_idx += self.skip_step
            yield x, y


class RandomSampleGenerator:
    def __init__(self, data, num_steps, batch_size, vocab_size):
        """
        :param data:        list[int]
        :param num_steps:   int
        :param batch_size:  int
        :param vocab_size:  int
        """
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocab_size = vocab_size

        self.sample_indices = np.arange(len(data) - num_steps)
        np.random.shuffle(self.sample_indices)
        self.current_idx = 0

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocab_size))
        while True:
            for i in range(self.batch_size):
                if self.current_idx >= len(self.sample_indices):
                    # reset the index back to the start of the data set
                    self.sample_indices = np.arange(len(self.data) - self.num_steps)
                    np.random.shuffle(self.sample_indices)
                    self.current_idx = 0

                idx = self.sample_indices[self.current_idx]
                x[i, :] = self.data[idx:idx + self.num_steps]
                temp_y = self.data[idx + 1:idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = keras.utils.to_categorical(temp_y, num_classes=self.vocab_size)
                self.current_idx += 1
            yield x, y
