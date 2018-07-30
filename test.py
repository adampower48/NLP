from string import punctuation

import numpy as np


def normalise_file():
    FILENAME = "shakespeare.txt"

    with open(FILENAME) as f:
        data = f.read()

    data = data.lower()

    for p in punctuation:
        data = data.replace(p, "")

    data = data.replace("\n", " ")
    data = data.replace("\"", "")
    for _ in range(10):
        data = data.replace("  ", " ")

    with open(FILENAME, "w") as f:
        f.write(data)


# normalise_file()


# with open("shakespeare.txt") as f:
#     # Words
#     # word_list = f.read().split()[:100000]
#     # Characters
#     word_list = list(f.read(100000))
#
#
# chars = list(string.ascii_letters + " " + string.digits)
#
# vocab = keras.preprocessing.text.Tokenizer(num_words=10000, char_level=True, oov_token="?")
# vocab.fit_on_texts(word_list)
#
# print(vocab.texts_to_sequences(word_list))
#

x = np.array([[1, 2, 3, 4, 5]])
print(x)
x = np.append(x[:, 1:], [[6]], axis=1)
print(x)
