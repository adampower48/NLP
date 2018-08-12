import numpy as np
from tensorflow import keras

from data_loader import read_file, clean_data, generate_indices, KerasBatchGenerator

FILENAMES_DATASETS = {
    "trump": "datasets/trump_tweets.txt",
}

FILENAMES_MODELS = {
    "trump_chars": "checkpoints/model_trump_c.hdf5",
}

# Dataset Params
model_filename = FILENAMES_MODELS["trump_chars"]
data_filename = FILENAMES_DATASETS["trump"]

# Network Params
NUM_STEPS = 100
BATCH_SIZE = 256
HIDDEN_SIZE = 256
NUM_EPOCHS = 2000


def parse_dataset(filename, dataset_length=None):
    sentence_list = read_file(filename, encoding="utf-8").split("\n")

    if dataset_length:
        sentence_list = sentence_list[:dataset_length]

    for i in range(len(sentence_list)):
        sentence_list[i] = clean_data(sentence_list[i], whitespace=True, other=True)

    w_i, i_w = generate_indices(" ".join(sentence_list))
    index_list = []
    for sentence in sentence_list:
        index_list.append(1)  # <START>
        index_list += [w_i[w] for w in sentence]
        index_list.append(2)  # <END>

    return index_list, w_i, i_w


class DemoCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        demo(False, False, 280)


index_list, w_i, i_w = parse_dataset(data_filename)

split = int(len(index_list) * 0.9)
train_data, valid_data = index_list[:split], index_list[split:]

VOCAB_SIZE = len(i_w.keys())
print("Vocabulary size:", VOCAB_SIZE, "/", len(w_i.keys()))
print(*sorted(w_i.keys()))
train_data_generator = KerasBatchGenerator(train_data, NUM_STEPS, BATCH_SIZE, VOCAB_SIZE, skip_step=1)
valid_data_generator = KerasBatchGenerator(valid_data, NUM_STEPS, BATCH_SIZE, VOCAB_SIZE, skip_step=1)


def train(resume=False):
    if not resume:
        model = keras.models.Sequential([
            keras.layers.Embedding(VOCAB_SIZE, HIDDEN_SIZE, input_length=NUM_STEPS),
            # pretrained_embedding_layer(w_i),  # Using GLOVE pre-trained embedding
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, implementation=2, dropout=0.1),
            keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, implementation=2, dropout=0.1),
            keras.layers.TimeDistributed(keras.layers.Dense(VOCAB_SIZE)),
            keras.layers.Lambda(lambda x: x / 1.5),  # Adding temperature to softmax layer
            keras.layers.Activation(keras.activations.softmax),
        ])
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
        prob_word = np.random.choice(range(VOCAB_SIZE), p=prediction[0, NUM_STEPS - 1, :])
        determ_print_out.append(i_w[determ_word])
        prob_output.append(i_w[prob_word])

        seed_data = np.append(seed_data[:, 1:], [[determ_word if deterministic else prob_word]], axis=1)

        if verbose:
            # Show confidence
            n = 5
            top_n = np.argpartition(prediction[:, NUM_STEPS - 1, :], -n)[:, -n:]
            top_n_weights = prediction[:, NUM_STEPS - 1, top_n[0]]
            sorted_inds = np.argsort(top_n_weights)[0]
            top_n_words = [i_w[i] for i in top_n[0, sorted_inds]]
            print(*["{} {:.4f}".format(repr(x[0]), x[1]) for x in zip(top_n_words, top_n_weights[0, sorted_inds])],
                  sep=", \t")

    print("Probabilistic Output:\n", *prob_output, sep="")

    return determ_print_out


def demo(deterministic=False, words=True, num_predict=280):
    sep_char = " " if words else ""

    # Build model & dataset
    model = keras.models.load_model(model_filename)
    example_training_generator = KerasBatchGenerator(train_data, NUM_STEPS, 1, VOCAB_SIZE, skip_step=1)

    # Random starting index
    dummy_iters = np.random.randint(10000 - num_predict)
    for i in range(dummy_iters):
        next(example_training_generator.generate())

    # Go to start of next sentence
    while next(example_training_generator.generate())[0][0, 0] != 2:
        pass

    # Generate text
    seed_data = next(example_training_generator.generate())[0]
    print("Seed Input:\n", *[i_w[int(x)] for x in seed_data[0]], sep=sep_char)
    print("Deterministic Output:\n",
          *predict_from_seed(model, seed_data, num_predict, verbose=True, deterministic=deterministic), sep=sep_char)


if __name__ == '__main__':
    # train(resume=True)
    demo(False, False, 1000)
