import io
import os
from datetime import datetime
from typing import Text, Optional

import numpy
import numpy as np
import tensorflow as tf
from keras import layers
from keras.utils import np_utils
from tensorflow import keras

from gui2lm.gui2lm.data_abstracting.configuration.conf import Configuration
# load ascii text and covert to lowercase
from gui2lm.gui2lm.preprocessing.tokens import Tokens

MAX_LENGTH_GUI_REPRESENTATION = 87


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds.flatten(), 1)
    return np.argmax(probas)


def perplexity(y_true, y_pred):
    # cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    cross_entropy = tf.keras.backend.cast(tf.keras.backend.equal(tf.keras.backend.max(y_true, axis=-1),
                                                                 tf.keras.backend.cast(
                                                                     tf.keras.backend.argmax(y_pred, axis=-1),
                                                                     tf.keras.backend.floatx())),
                                          tf.keras.backend.floatx())
    perplexity = tf.exp(tf.reduce_mean(cross_entropy))
    return perplexity


class LanguageModel_V2:

    def __init__(self, conf: Configuration, folder_name: Optional[Text] = "language_model"):
        self.conf = conf
        self.char_indices = Tokens().char2int()
        self.indices_char = Tokens().int2char()
        self.maxlen = 40
        self.step = 1
        self.name = "language_model"
        self.directory = self.conf.path_trained_models + "/" + folder_name + "/"
        self.folder_name = folder_name
        self.model = None
        self.epochs = 3
        self.batch_size = 128
        self.embedding_dim = 100
        self.neuron_nr = 128
        self.test_run = True

    def prepare_training_data(self):
        if self.test_run:
            path_to_data = self.conf.path_preproc_text_small
        else:
            path_to_data = self.conf.path_preproc_text
        #     Read Text
        filename = "Y" + str(self.conf.number_splits_y) + "X" + str(self.conf.number_splits_x) + "/train"
        with io.open(path_to_data + filename, encoding="utf-8") as f:
            text = f.read()
            # print(text)
            gui_list = text.split("\n")
            print("GUI LIST", gui_list)
            print("Nr. GUIs for Training:", len(gui_list))

            # Read every GUI independently and pad each sample to a fixed length
            sentence_list = []
            next_chars_list = []

            dataX = []
            dataY = []

            for j in range(0, len(gui_list)):
                for i in range(1, len(gui_list[j])):
                    gui_subpart = gui_list[j][0: i].ljust(MAX_LENGTH_GUI_REPRESENTATION, "_")
                    sentence_list.append(gui_subpart)
                    next_chars_list.append(gui_list[j][i])

                    dataX.append([self.char_indices[char] for char in gui_subpart])
                    dataY.append(self.char_indices[gui_list[j][i]])

            print("Nr. Training Sampels:", len(dataX))

            # reshapes X to be [samples, time steps, features]
            X = np.reshape(dataX, (len(dataX), MAX_LENGTH_GUI_REPRESENTATION, 1))

            # one hot encodes the output variable
            y = np_utils.to_categorical(dataY)

            #
            # x = np.zeros((len(sentence_list), MAX_LENGTH_GUI_REPRESENTATION, len(self.char_indices)), dtype=np.bool)
            # y = np.zeros((len(sentence_list), len(self.char_indices)), dtype=np.bool)
            # for i, sentence in enumerate(sentence_list):
            #     for t, char in enumerate(sentence):
            #         x[i, t, self.char_indices[char]] = 1
            #     y[i, self.char_indices[next_chars_list[i]]] = 1
            # print("X", X)
            # print("Y", y)
            print("Training Data Prepared")

            # # Code for debugging purposes
            # with open(self.conf.PATH_ROOT + "/guckenWarumDerKackNichtGehr", 'w') as outfile:
            #     # I'm writing a header here just for the sake of readability
            #     # Any line starting with "#" will be ignored by numpy.loadtxt
            #     outfile.write('# Array shape: {0}\n'.format(x.shape))
            #
            #     # Iterating through a ndimensional array produces slices along
            #     # the last axis. This is equivalent to data[i,:,:] in this case
            #     for data_slice in x:
            #         # The formatting string indicates that I'm writing out
            #         # the values in left-justified columns 7 characters in width
            #         # with 2 decimal places.
            #         np.savetxt(outfile, data_slice, fmt='%-7.2f')
            #
            #         # Writing out a break to indicate different slices...
            #         outfile.write('# New slice\n')
            #
            # np.savetxt(self.conf.PATH_ROOT + "/guckenWarumDerKackNichtGehr2", y, fmt='%-7.2f')

            return text, X, y

    def prepare_validation_data(self):
        # Read Text
        if self.test_run:
            path_to_data = self.conf.path_preproc_text_small
        else:
            path_to_data = self.conf.path_preproc_text
        filename = "Y" + str(self.conf.number_splits_y) + "X" + str(self.conf.number_splits_x) + "/validate"
        with io.open(path_to_data + filename, encoding="utf-8") as f:
            text = f.read()
            # print(text)
            gui_list = text.split("\n")
            print("GUI LIST", gui_list)
            print("Nr. GUIs for Training:", len(gui_list))

            # Read every GUI independently and pad each sample to a fixed length
            sentence_list = []
            next_chars_list = []

            dataX = []
            dataY = []

            for j in range(0, len(gui_list)):
                for i in range(1, len(gui_list[j])):
                    gui_subpart = gui_list[j][0: i].ljust(MAX_LENGTH_GUI_REPRESENTATION, "_")
                    sentence_list.append(gui_subpart)
                    next_chars_list.append(gui_list[j][i])

                    dataX.append([self.char_indices[char] for char in gui_subpart])
                    dataY.append(self.char_indices[gui_list[j][i]])

            print("Nr. Validation Sampels:", len(dataX))

            # reshapes X to be [samples, time steps, features]
            X = np.reshape(dataX, (len(dataX), MAX_LENGTH_GUI_REPRESENTATION, 1))

            # one hot encodes the output variable
            y = np_utils.to_categorical(dataY)

            print("Validation Data Prepared")

            return X, y

    def build_model(self):
        model = keras.Sequential(
            [
                layers.Embedding(len(self.char_indices), self.embedding_dim, input_length=MAX_LENGTH_GUI_REPRESENTATION,
                                 mask_zero=True),
                layers.LSTM(self.neuron_nr, return_sequences=True),
                layers.LSTM(self.neuron_nr),
                layers.Dense(len(self.char_indices), activation="softmax"),
            ]
        )
        # TODO optimizer
        optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=[perplexity, "accuracy"])
        # model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        print(model.summary())
        return model

    def train_model(self):
        epochs = self.epochs
        batch_size = self.batch_size
        text, x, y = self.prepare_training_data()
        x_val, y_val = self.prepare_validation_data()
        model = self.build_model()

        # for epoch in range(epochs):
        # print("Generating text after epoch: %d" % epoch)
        # today = date.today()
        # d1 = today.strftime("%Y-%d-%m")
        # print("date and time =", d1)
        # directory = self.conf.path_trained_models + "/" + d1 + "/"
        # directory = self.conf.path_trained_models + "/" + d1 + "/"
        try:
            # Create target Directory
            os.mkdir(self.directory)
            print("Directory ", self.directory, " Created ")
        except FileExistsError:
            print("Directory ", self.directory, " already exists")
        path2file = self.directory + self.name
        model.save_weights(path2file)
        log_dir = "logs/fit/" + self.folder_name + "/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        history = model.fit(x, y, batch_size=batch_size, epochs=self.epochs, validation_data=(x_val, y_val),
                            callbacks=[tensorboard_callback])
        model.save_weights(path2file)
        self.model = model

    def generating_text(self, seed: Text):
        self.model = self.build_model()
        self.model.load_weights(self.directory + self.name)
        print()

        # start_index = random.randint(0, len(text) - self.maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print("...Diversity:", diversity)

            sentence = seed.ljust(MAX_LENGTH_GUI_REPRESENTATION, "_")
            generated = seed

            print('...Generating with seed: "' + seed + '"')

            next_char = ""
            # prevent and endless loop
            i = 0

            pattern = [[self.char_indices[char] for char in sentence]]

            while next_char != Tokens().token2char[Tokens.end_token] and i < MAX_LENGTH_GUI_REPRESENTATION:
                x = numpy.reshape(pattern, (1, MAX_LENGTH_GUI_REPRESENTATION, 1))
                prediction = self.model.predict(x, verbose=0)
                next_index = sample(prediction, diversity)
                next_char = self.indices_char[next_index]
                # next_char = self.indices_char[numpy.argmax(prediction)]
                sentence = sentence[0:len(seed) + i] + next_char
                generated += next_char
                # add padding
                sentence = sentence.ljust(MAX_LENGTH_GUI_REPRESENTATION, "_")
                pattern = [[self.char_indices[char] for char in sentence]]
                i += 1

            print("...Generated: ", generated)
            print()
