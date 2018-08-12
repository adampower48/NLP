import codecs
import collections
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
        UNKNOWN_TOKEN = "<UNK>"
        UNKNOWN_INDEX = 0
        START_TOKEN = "<START>"
        START_INDEX = 1
        END_TOKEN = "<END>"
        END_INDEX = 2

        inds_to_tokens = {
            UNKNOWN_INDEX: UNKNOWN_TOKEN,
            START_INDEX: START_TOKEN,
            END_INDEX: END_TOKEN,
        }

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


def clean_data(data, lower=False, punctuation=False, whitespace=False, other=False):
    """

    :param data:        str
    :param lower:       bool
    :param punctuation: bool
    :param whitespace:  bool
    :param other:       bool
    :return:            str
    """

    if lower:
        data = data.lower()

    if punctuation:
        for p in string.punctuation:
            data = data.replace(p, "")

    if whitespace:
        # + non-breaking space
        for w in string.whitespace + " ":
            if w == " ":
                continue

            data.replace(w, " ")

        # Trim consecutive whitespace
        for _ in range(10):
            data = data.replace("  ", " ")

    if other:
        # Normalise punctuation
        for c in "‘’“”":
            data = data.replace(c, "\"")

        for c in "–—•":
            data = data.replace(c, "-")

        data = data.replace("…", "...")

        # Delete other characters
        for c in set(data):
            if c not in string.printable:
                data = data.replace(c, "")

    return data


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
