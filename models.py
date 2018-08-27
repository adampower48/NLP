from tensorflow import keras

from data_loader import pretrained_embedding


def large_pt_embedding(VOCAB_SIZE, NUM_STEPS, w_i, embedding_filename, freeze_embedding, HIDDEN_SIZE=256, ):
    model = keras.models.Sequential([
        # keras.layers.Embedding(VOCAB_SIZE, HIDDEN_SIZE, input_length=NUM_STEPS),
        pretrained_embedding(embedding_filename, w_i, NUM_STEPS, freeze_embedding),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, implementation=2, dropout=0.1),
        keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, implementation=2, dropout=0.1),
        keras.layers.TimeDistributed(keras.layers.Dense(VOCAB_SIZE)),
        # Adding temperature to softmax layer (lower = more random, higher = more focussed)
        keras.layers.Lambda(lambda x: x * 0.6),
        keras.layers.Activation(keras.activations.softmax),
    ])

    return model


def small_pt_embedding(VOCAB_SIZE, NUM_STEPS, w_i, embedding_filename, freeze_embedding, HIDDEN_SIZE=128):
    model = keras.models.Sequential([
        # keras.layers.Embedding(VOCAB_SIZE, HIDDEN_SIZE, input_length=NUM_STEPS),
        pretrained_embedding(embedding_filename, w_i, NUM_STEPS, freeze_embedding),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, implementation=2, dropout=0.1),
        keras.layers.TimeDistributed(keras.layers.Dense(VOCAB_SIZE)),
        # Adding temperature to softmax layer (lower = more random, higher = more focussed)
        keras.layers.Lambda(lambda x: x * 0.6),
        keras.layers.Activation(keras.activations.softmax),
    ])

    return model
