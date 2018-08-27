import sys

import numpy as np
from tensorflow import keras

import models as myModels
from data_loader import clean_data, generate_indices, KerasBatchGenerator, read_file

# Works with large single text bodies

FILENAMES_DATASETS = {
    "lorem_ipsum": "datasets/lorem.txt",
    "dracula": "datasets/dracula.txt",
    "shakespeare": "datasets/shakespeare.txt",
    "trump_tweets": "datasets/trump_tweets.txt",
    "trump_speech": "datasets/trump_transcripts/speech_all.txt",
}

FILENAMES_MODELS = {
    "words": "checkpoints/model_words.hdf5",
    "chars": "checkpoints/model_chars.hdf5",
    "dracula_chars": "checkpoints/model_drac_c.hdf5",
    "dracula_words": "checkpoints/model_drac_w.hdf5",
    "trump_tweets_chars": "checkpoints/model_trump_tweets_c.hdf5",
    "trump_speech_chars_small": "checkpoints/model_trump_speech_c_small.hdf5",
    "trump_speech_chars_large": "checkpoints/model_trump_speech_c_large.hdf5",
}

FILENAMES_EMBEDDING = {
    "chars_300D": "pretrained_weights/chars_300D.json",
    "glove_6B_50D": "pretrained_weights/glove.6B.50d.txt",
}

# DATASET PARAMETERS
model_filename = FILENAMES_MODELS["trump_tweets_chars"]
data_filename = FILENAMES_DATASETS["trump_tweets"]
use_chars = True
clean = {
    "lower": False,
    "punctuation": False,
    "whitespace": True,
    "other": True,
}
dictionary_size = 3000

# NETWORK PARAMETERS
embedding_filename = FILENAMES_EMBEDDING["chars_300D"]
freeze_embedding = True
NUM_STEPS = 100
BATCH_SIZE = 256
HIDDEN_SIZE = 128
NUM_EPOCHS = 2000

# GENERATION PARAMETERS
PROB_CUTOFF = 0.9
DEMO_STEPS = 1000


def parse_dataset(filename, chars=False, dataset_length=None, max_indices=1000, clean=None):
    if clean is None:
        clean = {}

    data = read_file(filename, encoding="utf-8")
    if chars:
        # Characters
        word_list = list(clean_data(data, **clean))
    else:
        # Words
        word_list = clean_data(data, **clean).split()

    if dataset_length:
        word_list = word_list[:dataset_length]

    w_i, i_w = generate_indices(word_list, max_indices)
    index_list = [w_i[w] for w in word_list]

    return word_list, index_list, w_i, i_w


word_list, index_list, w_i, i_w = parse_dataset(data_filename, use_chars, clean=clean,
                                                max_indices=dictionary_size)
split = int(len(index_list) * 0.8)
train_data, valid_data = index_list[:split], index_list[split:]

VOCAB_SIZE = len(i_w.keys())
print("Vocabulary size:", VOCAB_SIZE, "/", len(w_i.keys()))
print(*sorted(list(w_i.keys())))
train_data_generator = KerasBatchGenerator(train_data, NUM_STEPS, BATCH_SIZE, VOCAB_SIZE, skip_step=1)
valid_data_generator = KerasBatchGenerator(valid_data, NUM_STEPS, BATCH_SIZE, VOCAB_SIZE, skip_step=1)


class DemoCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        demo(False, False, 280)


def filterPercentile(percentages, cutoff):
    sorted_idxs = np.argsort(percentages)[::-1]
    cum_per = np.cumsum(percentages[sorted_idxs])
    ind = next(i for i in range(len(cum_per)) if cum_per[i] > cutoff)
    top = sorted_idxs[:ind + 1]
    adjusted_percentages = percentages[top] / sum(percentages[top])

    return top, adjusted_percentages


def filterN(percentages, n):
    top_n = np.argpartition(percentages, -n)[-n:]
    top_n_weights = percentages[top_n]
    sorted_inds = np.argsort(top_n_weights)[::-1]
    top = top_n[sorted_inds]
    adjusted_percentages = top_n_weights[sorted_inds] / sum(top_n_weights[sorted_inds])

    return top, adjusted_percentages


def train(resume=False):
    if not resume:
        model = myModels.large_pt_embedding(VOCAB_SIZE, NUM_STEPS, w_i, embedding_filename, freeze_embedding)
    else:
        model = keras.models.load_model(model_filename)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.categorical_accuracy])

    model.summary()
    model.save(model_filename)

    checkpointer = keras.callbacks.ModelCheckpoint(model_filename, verbose=1, period=1, save_best_only=True)
    demoCallback = DemoCallback()

    model.fit_generator(train_data_generator.generate(), (len(train_data) - NUM_STEPS) // BATCH_SIZE, NUM_EPOCHS,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=(len(valid_data) - NUM_STEPS) // BATCH_SIZE,
                        callbacks=[checkpointer, demoCallback])


def predict_from_seed(model, seed_data, num_predict, verbose=False, deterministic=False):
    determ_print_out = []
    prob_output = []
    for i in range(num_predict):
        prediction = model.predict(seed_data)
        determ_word = np.argmax(prediction[:, NUM_STEPS - 1, :])
        # prob_word = np.random.choice(range(VOCAB_SIZE), p=prediction[0, NUM_STEPS - 1, :])  # All
        top, percentages = filterPercentile(prediction[0, NUM_STEPS - 1, :], PROB_CUTOFF)  # Percentile
        prob_word = np.random.choice(top, p=percentages)  # Percentile
        determ_print_out.append(i_w[determ_word])
        prob_output.append(i_w[prob_word])

        seed_data = np.append(seed_data[:, 1:], [[determ_word if deterministic else prob_word]], axis=1)

        if verbose:
            # Show confidence (Top N)
            # top, percentages = filterN(prediction[0, NUM_STEPS - 1, :], 5)

            # Top Percentile
            top, percentages = filterPercentile(prediction[0, NUM_STEPS - 1, :], PROB_CUTOFF)
            top_words = [i_w[i] for i in top]
            print(*["{} {:.4f}".format(repr(x[0]), x[1]) for x in zip(top_words, percentages)], sep=", \t")

    print("Probabilistic Output:\n", *prob_output, sep="")

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


def demo(deterministic=False, words=False, num_predict=280):
    sep_char = " " if words else ""

    # Build model & dataset
    model = keras.models.load_model(model_filename)
    example_training_generator = KerasBatchGenerator(train_data, NUM_STEPS, 1, VOCAB_SIZE, skip_step=1)

    # Random starting index
    dummy_iters = np.random.randint(10000 - num_predict)
    for i in range(dummy_iters):
        next(example_training_generator.generate())

    # Generate text
    seed_data = next(example_training_generator.generate())[0]
    print("Seed Input:\n", *[i_w[int(x)] for x in seed_data[0]], sep=sep_char)
    print("Deterministic Output:\n",
          *predict_from_seed(model, seed_data, num_predict, verbose=True, deterministic=deterministic), sep=sep_char)


if __name__ == '__main__':
    print("Model:", model_filename)
    print("Dataset:", data_filename)
    print("Embedding:", embedding_filename)

    if len(sys.argv) < 2:
        train(resume=True)
        # demo(False, False, DEMO_STEPS)
        exit()

    if sys.argv[1] == "demo":
        demo()
    elif sys.argv[1] == "resume":
        train(resume=True)
    elif sys.argv[1] == "new":
        train(resume=False)
