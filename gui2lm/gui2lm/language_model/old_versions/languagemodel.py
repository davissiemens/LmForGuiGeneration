import io
import os
from typing import Text, Optional

import numpy as np
from keras import layers
from tensorflow import keras

from gui2lm.gui2lm.configuration import Configuration
# load ascii text and covert to lowercase
from gui2lm.gui2lm.preprocessing.tokens import Tokens

MAX_LENGTH_GUI_REPRESENTATION = 87


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


class LanguageModel:

    def __init__(self, conf: Configuration, folder_name: Optional[Text] = "language_model"):
        self.conf = conf
        self.char_indices = Tokens().char2int()
        self.indices_char = Tokens().int2char()
        self.maxlen = 40
        self.step = 1
        self.name = "language_model"
        self.directory = self.conf.path_trained_models + "/" + folder_name + "/"
        self.model = None
        self.epochs = 1
        self.batch_size = 128

    def prepare_data(self):
        # Read Text
        path_to_data = self.conf.path_preproc_text
        filename = "Y" + str(self.conf.number_splits_y) + "X" + str(self.conf.number_splits_x)
        with io.open(path_to_data + filename, encoding="utf-8") as f:
            text = f.read()
            print(text)
            text = text.replace("\n", " ")  # We remove newlines chars for nicer display
            print("Corpus length:", len(text))

            sentences = []
            next_chars = []

            for i in range(0, len(text) - self.maxlen, self.step):
                sentences.append(text[i: i + self.maxlen])
                next_chars.append(text[i + self.maxlen])

            print("Number of sequences:", len(sentences))

            x = np.zeros((len(sentences), self.maxlen, len(self.char_indices)), dtype=np.bool)
            y = np.zeros((len(sentences), len(self.char_indices)), dtype=np.bool)
            for i, sentence in enumerate(sentences):
                for t, char in enumerate(sentence):
                    x[i, t, self.char_indices[char]] = 1
                y[i, self.char_indices[next_chars[i]]] = 1
            print("X", x)
            print("Y", y)
            print("Data Prepared")
            return text, x, y

    def prepare_data_alternatively(self):
        # Read Text
        path_to_data = self.conf.path_preproc_text_small
        filename = "Y" + str(self.conf.number_splits_y) + "X" + str(self.conf.number_splits_x)
        with io.open(path_to_data + filename, encoding="utf-8") as f:
            text = f.read()
            # print(text)
            gui_list = text.split("\n")
            print("GUI LIST", gui_list)
            print("Nr. GUIs for Training:", len(gui_list))

            # Read every GUI independently and pad each sample to a fixed length
            sentence_list = []
            next_chars_list = []
            for j in range(0, len(gui_list)):
                # sentence = []
                # next_chars = []
                for i in range(1, len(gui_list[j])):
                    gui_subpart = gui_list[j][0: i].rjust(MAX_LENGTH_GUI_REPRESENTATION, "_")
                    sentence_list.append(gui_subpart)
                    next_chars_list.append(gui_list[j][i])
                # sentence_list.append(sentence)
                # next_chars_list.append(next_chars)
            # # Code to debug and print samples
            # j = 0
            # for sentence in sentence_list:
            #     print(sentence_list[j])
            #     print(next_chars_list[j])
            #     j += 1

            x = np.zeros((len(sentence_list), MAX_LENGTH_GUI_REPRESENTATION, len(self.char_indices)), dtype=np.bool)
            y = np.zeros((len(sentence_list), len(self.char_indices)), dtype=np.bool)
            for i, sentence in enumerate(sentence_list):
                for t, char in enumerate(sentence):
                    x[i, t, self.char_indices[char]] = 1
                y[i, self.char_indices[next_chars_list[i]]] = 1
            print("X", x)
            print("Y", y)
            print("Data Prepared")

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

            return text, x, y

    def build_model(self):
        model = keras.Sequential(
            [
                keras.Input(shape=(self.maxlen, len(self.char_indices))),
                layers.LSTM(128),
                layers.Dense(len(self.char_indices), activation="softmax"),
            ]
        )
        optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer)
        return model

    def build_model_alternatively(self):
        model = keras.Sequential(
            [
                keras.Input(shape=(MAX_LENGTH_GUI_REPRESENTATION, len(self.char_indices))),
                layers.LSTM(128),
                layers.Dense(len(self.char_indices), activation="softmax"),
            ]
        )
        optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer)
        return model

    def train_model(self):
        epochs = self.epochs
        batch_size = self.batch_size
        text, x, y = self.prepare_data()
        model = self.build_model()

        for epoch in range(epochs):
            print("Generating text after epoch: %d" % epoch)
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
            model.fit(x, y, batch_size=batch_size, epochs=1)
            model.save_weights(path2file)
            self.model = model

    def train_model_alternatively(self):
        epochs = self.epochs
        batch_size = self.batch_size
        text, x, y = self.prepare_data_alternatively()
        model = self.build_model_alternatively()

        for epoch in range(epochs):
            print("Generating text after epoch: %d" % epoch)
            try:
                # Create target Directory
                os.mkdir(self.directory)
                print("Directory ", self.directory, " Created ")
            except FileExistsError:
                print("Directory ", self.directory, " already exists")
            path2file = self.directory + self.name
            model.save_weights(path2file)
            model.fit(x, y, batch_size=batch_size, epochs=1)
            model.save_weights(path2file)
            self.model = model

    def generating_text(self, sentence: Text):
        self.model = self.build_model_alternatively()
        self.model.load_weights(self.directory + self.name)
        print()

        # start_index = random.randint(0, len(text) - self.maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print("...Diversity:", diversity)

            generated = ""

            print('...Generating with seed: "' + sentence + '"')

            for i in range(40):
                x_pred = np.zeros((1, self.maxlen, len(self.char_indices)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, self.char_indices[char]] = 1.0
                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = self.indices_char[next_index]
                sentence = sentence[1:] + next_char
                generated += next_char

            print("...Generated: ", generated)
            print()

    def generating_text_alternatively(self, seed: Text):
        self.model = self.build_model_alternatively()
        self.model.load_weights(self.directory + self.name)
        print()

        # start_index = random.randint(0, len(text) - self.maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print("...Diversity:", diversity)

            sentence = seed
            generated = ""

            print('...Generating with seed: "' + seed + '"')

            next_char = ""
            # prevent and endless loop
            i = 0
            while next_char != Tokens.end_token and i < 40:
                x_pred = np.zeros((1, MAX_LENGTH_GUI_REPRESENTATION, len(self.char_indices)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, self.char_indices[char]] = 1.0
                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = self.indices_char[next_index]
                sentence = sentence[1:] + next_char
                generated += next_char
                i += 1

            print("...Generated: ", generated)
            print()

# #
# conf = Configuration()
# filename = conf.path_preproc_text_small + "Y" + str(conf.number_splits_y) + "X" + str(conf.number_splits_x)
# raw_text = open(filename, 'r', encoding='utf-8').read()
#
# # TODO create mapping of tokens to integers
# # chars = sorted(list(set(raw_text)))
# # char_to_int = dict((c, i) for i, c in enumerate(chars))
#
# # summarize the loaded data
# n_chars = len(raw_text)
# n_vocab = len(Tokens().tokens)
#
# print("Total Characters: ", n_chars)
# print("Total Vocab: ", n_vocab)
# # prepare the dataset of input to output pairs encoded as integers
# seq_length = 100
# dataX = []
# dataY = []
# for i in range(0, n_chars - seq_length, 1):
# 	seq_in = raw_text[i:i + seq_length]
# 	seq_out = raw_text[i + seq_length]
# 	dataX.append([char_to_int[char] for char in seq_in])
# 	dataY.append(char_to_int[seq_out])
# n_patterns = len(dataX)
# print "Total Patterns: ", n_patterns
# # reshape X to be [samples, time steps, features]
# X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# # normalize
# X = X / float(n_vocab)
# # one hot encode the output variable
# y = np_utils.to_categorical(dataY)
# # define the LSTM model
# model = Sequential()
# model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
# model.add(Dropout(0.2))
# model.add(Dense(y.shape[1], activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam')
# # define the checkpoint
# filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]
# # fit the model
# model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
