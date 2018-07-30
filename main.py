import string
from collections import Counter

import numpy as np
from tensorflow import keras

FILENAMES_DATASETS = {
    "lorem_ipsum": "datasets/lorem.txt",
    "dracula": "datasets/dracula.txt",
    "shakespeare": "datasets/shakespeare.txt",
}

FILENAMES_MODELS = {
    "words": "checkpoints/model_words.hdf5",
    "chars": "checkpoints/model_chars.hdf5",
    "dracula_chars": "checkpoints/model_drac_c.hdf5",
}

# DATASET PARAMETERS
model_filename = FILENAMES_MODELS["dracula_chars"]
data_filename = FILENAMES_DATASETS["dracula"]
use_chars = True

CHARS = string.ascii_letters + string.whitespace + string.digits  # + string.punctuation

# NETWORK PARAMETERS
NUM_STEPS = 100
BATCH_SIZE = 20
HIDDEN_SIZE = 256
NUM_EPOCHS = 200


def gen_indices(words):
    counter = Counter(words)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words_to_ind = {}
    ind_to_words = {}
    for i, (w, _) in enumerate(count_pairs):
        words_to_ind[w] = i
        ind_to_words[i] = w

    return words_to_ind, ind_to_words


def parse_dataset(filename, chars=False, dataset_length=None):
    with open(filename) as f:
        if chars:
            # Characters
            word_list = list(f.read())
        else:
            # Words
            word_list = f.read().split()

    if dataset_length:
        word_list = word_list[:dataset_length]

    w_i, i_w = gen_indices(word_list)
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


word_list, index_list, w_i, i_w = parse_dataset(data_filename, use_chars)

split = int(len(index_list) * 0.8)
train_data, valid_data = index_list[:split], index_list[split:]

VOCAB_SIZE = len(w_i.keys())
print("Vocabulary size:", VOCAB_SIZE)
train_data_generator = KerasBatchGenerator(train_data, NUM_STEPS, BATCH_SIZE, VOCAB_SIZE, skip_step=1)
valid_data_generator = KerasBatchGenerator(valid_data, NUM_STEPS, BATCH_SIZE, VOCAB_SIZE, skip_step=1)


def train(resume=False):
    if not resume:
        model = keras.models.Sequential([
            keras.layers.Embedding(VOCAB_SIZE, HIDDEN_SIZE, input_length=NUM_STEPS),
            keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, dropout=0.2),
            # keras.layers.Dropout(0.2),
            keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, dropout=0.2),
            # keras.layers.Dropout(0.2),
            keras.layers.TimeDistributed(keras.layers.Dense(VOCAB_SIZE)),
            keras.layers.Activation(keras.activations.softmax),

        ])
    else:
        model = keras.models.load_model(model_filename)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.categorical_accuracy])

    model.summary()

    checkpointer = keras.callbacks.ModelCheckpoint(model_filename, verbose=1, period=5, save_best_only=True)

    model.fit_generator(train_data_generator.generate(), len(train_data) // (BATCH_SIZE * NUM_STEPS), NUM_EPOCHS,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=len(valid_data) // (BATCH_SIZE * NUM_STEPS), callbacks=[checkpointer])


def generate_words(model, seed_data, num_predict):
    pred_print_out = []
    for i in range(num_predict):
        prediction = model.predict(seed_data)
        predict_word = np.argmax(prediction[:, NUM_STEPS - 1, :])
        seed_data = np.append(seed_data[:, 1:], [[predict_word]], axis=1)
        pred_print_out.append(i_w[predict_word])

    return pred_print_out


def demo():
    model = keras.models.load_model(model_filename)
    dummy_iters = 10000
    example_training_generator = KerasBatchGenerator(train_data, NUM_STEPS, 1, VOCAB_SIZE, skip_step=1)
    for i in range(dummy_iters):
        next(example_training_generator.generate())
    num_predict = 100
    seed_data = next(example_training_generator.generate())[0]
    print("Seed:\n", *[i_w[int(x)] for x in seed_data[0]], sep="")
    print("Predicted words:\n", *generate_words(model, seed_data, num_predict), sep="")


if __name__ == '__main__':
    train(resume=True)
    # demo()
