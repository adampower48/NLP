import string
import sys
from collections import Counter

import numpy as np
from tensorflow import keras

# Works with large single text bodies

FILENAMES_DATASETS = {
    "lorem_ipsum": "datasets/lorem.txt",
    "dracula": "datasets/dracula.txt",
    "shakespeare": "datasets/shakespeare.txt",
    "trump": "datasets/trump_tweets.json",
}

FILENAMES_MODELS = {
    "words": "checkpoints/model_words.hdf5",
    "chars": "checkpoints/model_chars.hdf5",
    "dracula_chars": "checkpoints/model_drac_c.hdf5",
    "dracula_words": "checkpoints/model_drac_w.hdf5",
    "trump_chars": "checkpoints/model_trump_c.hdf5",
}

# DATASET PARAMETERS
model_filename = FILENAMES_MODELS["trump_chars"]
data_filename = FILENAMES_DATASETS["trump"]
use_chars = True
clean_file = True
dictionary_size = 3000

CHARS = string.ascii_letters + string.whitespace + string.digits  # + string.punctuation

# NETWORK PARAMETERS
NUM_STEPS = 20
BATCH_SIZE = 20
HIDDEN_SIZE = 256
NUM_EPOCHS = 2000


def clean_data(data, remove_punctuation=True, lower=True, trim_whitespace=True):
    if lower:
        data = data.lower()

    if remove_punctuation:
        for p in string.punctuation:
            data = data.replace(p, "")

    if trim_whitespace:
        data = data.replace("\n", " ")
        for _ in range(10):
            data = data.replace("  ", " ")

    return data


def gen_indices(words, max_indices=1000):
    UNKNOWN_TOKEN = "<UNK>"
    UNKNOWN_INDEX = 0
    counter = Counter(words)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words_to_ind = {}
    ind_to_words = {UNKNOWN_INDEX: UNKNOWN_TOKEN}
    for i, (w, _) in enumerate(count_pairs):
        if i < max_indices:
            words_to_ind[w] = i + 1
            ind_to_words[i + 1] = w
        else:
            words_to_ind[w] = UNKNOWN_INDEX

    return words_to_ind, ind_to_words


def pretrained_embedding_layer(word_to_index):
    with open("pretrained_weights/glove.6B.50d.txt", encoding="utf8") as f:
        lines = [l.split() for l in f.readlines()]
        word_to_vec_map = {l[0]: np.array(list(map(float, l[1:]))) for l in lines}
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    vocab_len = len(word_to_index) + 1  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]  # define dimensionality of your GloVe word vectors (= 50)

    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))

    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        try:
            emb_matrix[index, :] = word_to_vec_map[word]
        except KeyError:
            pass

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False.
    embedding_layer = keras.layers.Embedding(vocab_len, emb_dim, trainable=False)
    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))

    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def parse_dataset(filename, chars=False, dataset_length=None, max_indices=1000, clean=False):
    with open(filename, encoding="utf8") as f:
        data = f.read()
        if chars:
            # Characters
            word_list = list(data)
        else:
            # Words
            if clean:
                word_list = clean_data(data).split()
            else:
                word_list = data.split()

    if dataset_length:
        word_list = word_list[:dataset_length]

    w_i, i_w = gen_indices(word_list, max_indices)
    index_list = [w_i[w] for w in word_list]

    return word_list, index_list, w_i, i_w


class KerasBatchGenerator:
    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = keras.utils.to_categorical(temp_y, num_classes=self.vocabulary)
                self.current_idx += self.skip_step
            yield x, y


word_list, index_list, w_i, i_w = parse_dataset(data_filename, use_chars, clean=clean_file,
                                                max_indices=dictionary_size)

split = int(len(index_list) * 0.8)
train_data, valid_data = index_list[:split], index_list[split:]

VOCAB_SIZE = len(i_w.keys())
print("Vocabulary size:", VOCAB_SIZE, "/", len(w_i.keys()))
print(*w_i.keys())
train_data_generator = KerasBatchGenerator(train_data, NUM_STEPS, BATCH_SIZE, VOCAB_SIZE, skip_step=1)
valid_data_generator = KerasBatchGenerator(valid_data, NUM_STEPS, BATCH_SIZE, VOCAB_SIZE, skip_step=1)


def train(resume=False):
    if not resume:
        model = keras.models.Sequential([
            keras.layers.Embedding(VOCAB_SIZE, HIDDEN_SIZE, input_length=NUM_STEPS),
            # pretrained_embedding_layer(w_i),  # Using GLOVE pre-trained embedding
            keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, dropout=0.2, implementation=2),
            keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, dropout=0.2, implementation=2),
            keras.layers.TimeDistributed(keras.layers.Dense(VOCAB_SIZE)),
            keras.layers.Lambda(lambda x: x / 5),  # Adding temperature to softmax layer
            keras.layers.Activation(keras.activations.softmax),

        ])
    else:
        model = keras.models.load_model(model_filename)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.categorical_accuracy])

    model.summary()

    checkpointer = keras.callbacks.ModelCheckpoint(model_filename, verbose=1, period=1, save_best_only=True)

    model.fit_generator(train_data_generator.generate(), (len(train_data) - NUM_STEPS) // BATCH_SIZE, NUM_EPOCHS,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=(len(valid_data) - NUM_STEPS) // BATCH_SIZE, callbacks=[checkpointer])


def predict_from_seed(model, seed_data, num_predict, verbose=False):
    determ_print_out = []
    prob_output = []
    for i in range(num_predict):
        prediction = model.predict(seed_data)
        determ_word = np.argmax(prediction[:, NUM_STEPS - 1, :])
        prob_word = np.random.choice(range(3001), p=prediction[0, NUM_STEPS - 1, :])
        seed_data = np.append(seed_data[:, 1:], [[prob_word]], axis=1)
        determ_print_out.append(i_w[determ_word])
        prob_output.append(i_w[prob_word])

        if verbose:
            # Show confidence
            top_n = np.argpartition(prediction[:, NUM_STEPS - 1, :], -5)[:, -5:]
            top_n_weights = prediction[:, NUM_STEPS - 1, top_n[0]]
            sorted_inds = np.argsort(top_n_weights)[0]
            print(top_n[:, sorted_inds], top_n_weights[:, sorted_inds])

    print("Probabilistic Output:\n", *prob_output, sep=" ")

    return determ_print_out


def predict_from_generator(model, generator, num_predict, dummy_iters, verbose=False):
    act_output = []
    pred_output = []
    prob_output = []
    for i in range(num_predict):
        data = next(generator.generate())
        prediction = model.predict(data[0])
        predict_word = np.argmax(prediction[:, NUM_STEPS - 1, :])
        act_output.append(word_list[NUM_STEPS + dummy_iters + i])
        pred_output.append(i_w[predict_word])
        prob_output.append(i_w[np.random.choice(range(3001), p=prediction[0, NUM_STEPS - 1, :])])

        if verbose:
            # Show confidence
            top_n = np.argpartition(prediction[:, NUM_STEPS - 1, :], -5)[:, -5:]
            top_n_weights = prediction[:, NUM_STEPS - 1, top_n[0]]
            sorted_inds = np.argsort(top_n_weights)[0]
            print(top_n[:, sorted_inds], top_n_weights[:, sorted_inds])

    print("Probabilistic Output:\n", *prob_output, sep=" ")

    return act_output, pred_output


def demo(SEED=True, words=True):
    if words:
        sep_char = " "
    else:
        sep_char = ""

    model = keras.models.load_model(model_filename)
    example_training_generator = KerasBatchGenerator(train_data, NUM_STEPS, 1, VOCAB_SIZE, skip_step=1)
    num_predict = 1000

    dummy_iters = np.random.randint(len(word_list) - num_predict)
    for i in range(dummy_iters):
        next(example_training_generator.generate())

    if SEED:
        # From seed
        seed_data = next(example_training_generator.generate())[0]
        print("Seed Input:\n", *[i_w[int(x)] for x in seed_data[0]], sep=sep_char)
        print("Deterministic Output:\n", *predict_from_seed(model, seed_data, num_predict), sep=sep_char)
    else:
        # From dataset
        act, pred = predict_from_generator(model, example_training_generator, num_predict, dummy_iters)
        print("Actual words:\n", *act, sep=sep_char)
        print("Deterministic Output:\n", *pred, sep=sep_char)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        train(resume=True)
        # demo(SEED=True)
        exit()

    if sys.argv[1] == "demo":
        demo()
    elif sys.argv[1] == "resume":
        train(resume=True)
    elif sys.argv[1] == "new":
        train(resume=False)
