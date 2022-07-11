import io
import json
import os
from typing import Text, Optional

import keras_tuner
import numpy
import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Dropout
from keras.utils import np_utils
from keras_tuner import Hyperband
from tensorflow import keras

from gui2lm.gui2lm.configuration.conf import Configuration
from gui2lm.gui2lm.preprocessing.tokens import Tokens

TOKENS__INT_CHAR = Tokens().int2char()
TOKENS__CHAR_INT = Tokens().char2int()
MAX_LENGTH_GUI_REPRESENTATION = 87


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    # source: https://medium.com/mlearning-ai/text-generation-in-deep-learning-with-tensorflow-keras-f7cfd8d65d9e
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds.flatten(), 1)
    return np.argmax(probas)


# def perplexity(y_true, y_pred):
#     # https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CategoricalCrossentropy
#     cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
#
#     # https://en.wikipedia.org/wiki/Perplexity: "The perplexity is independent of the base, provided that the entropy and the exponentiation use the same base."
#     # Keras uses natural log to calculate cross-entropy like a lot of deep learning frameworks: https://github.com/keras-team/keras/blob/07e13740fd181fc3ddec7d9a594d8a08666645f6/keras/backend.py#L5075
#
#     # Another source for cross-entropy to perplexity: https://isl.anthropomatik.kit.edu/downloads/MA_Michael_Koch.pdf
#     return tf.exp(tf.reduce_mean(cross_entropy))


# build model function used for hyperparameter tuning
def build_model_for_param_search(hp):
    model = keras.Sequential()
    model.add(layers.Embedding(len(TOKENS__CHAR_INT), hp.Choice('embedding_dim', values=[8, 32, 64]),
                               input_length=MAX_LENGTH_GUI_REPRESENTATION,
                               mask_zero=True))
    neurons = hp.Choice('num_of_neurons', values=[64, 128])
    for i in range(hp.Int('num_of_layers', 1, 2)):
        model.add(Dropout(0.2))
        model.add(layers.LSTM(neurons, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(layers.LSTM(neurons))
    model.add(Dropout(0.2))
    # Softmax converts a vector of values to a probability distribution. - https://keras.io/api/layers/activations/
    model.add(layers.Dense(len(TOKENS__CHAR_INT), activation="softmax"))

    optimizer = keras.optimizers.RMSprop(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]))
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    print(model.summary())
    return model

# build model function
def build_model_from_dict(dict):
    model = keras.Sequential()
    model.add(layers.Embedding(len(TOKENS__CHAR_INT), dict['embedding_dim'],
                               input_length=MAX_LENGTH_GUI_REPRESENTATION,
                               mask_zero=True))
    neurons = dict['num_of_neurons']
    for i in range(dict['num_of_layers'] - 1):
        model.add(Dropout(0.2))
        model.add(layers.LSTM(neurons, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(layers.LSTM(neurons))
    model.add(Dropout(0.2))
    # Softmax converts a vector of values to a probability distribution. - https://keras.io/api/layers/activations/
    model.add(layers.Dense(len(TOKENS__CHAR_INT), activation="softmax"))

    optimizer = keras.optimizers.RMSprop(learning_rate=dict['learning_rate'])
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    print(model.summary())
    return model


class LanguageModel_WithParamTuning:

    def __init__(self, conf: Configuration, folder_name: Optional[Text] = "language_model"):
        self.tuner = None
        self.conf = conf
        self.name = "language_model"
        self.directory = self.conf.path_trained_models + "/" + folder_name + "/"
        self.folder_name = folder_name
        self.model = None

        self.batch_size = 128

        self.best_values = None

        # Deactivate for final runs
        self.test_run = False

    def prepare_training_data(self, print_data=False):
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
            if (print_data):
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

                    dataX.append([TOKENS__CHAR_INT[char] for char in gui_subpart])
                    dataY.append(TOKENS__CHAR_INT[gui_list[j][i]])

            print("Nr. Training Sampels:", len(dataX))

            # reshapes X to be [samples, time steps, features]
            X = np.reshape(dataX, (len(dataX), MAX_LENGTH_GUI_REPRESENTATION, 1))

            # one hot encodes the output variable
            y = np_utils.to_categorical(dataY)

            print("Training Data Prepared")

            return text, X, y

    def prepare_validation_data(self, print_data=False):
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
            if (print_data):
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

                    dataX.append([TOKENS__CHAR_INT[char] for char in gui_subpart])
                    dataY.append(TOKENS__CHAR_INT[gui_list[j][i]])

            print("Nr. Validation Sampels:", len(dataX))

            # reshapes X to be [samples, time steps, features]
            X = np.reshape(dataX, (len(dataX), MAX_LENGTH_GUI_REPRESENTATION, 1))

            # one hot encodes the output variable
            y = np_utils.to_categorical(dataY)

            print("Validation Data Prepared")

            return X, y

    def prepare_test_data(self, print_data=False):
        # Read Text
        if self.test_run:
            path_to_data = self.conf.path_preproc_text_small
        else:
            path_to_data = self.conf.path_preproc_text
        filename = "Y" + str(self.conf.number_splits_y) + "X" + str(self.conf.number_splits_x) + "/test"
        with io.open(path_to_data + filename, encoding="utf-8") as f:
            text = f.read()
            # print(text)
            gui_list = text.split("\n")
            if (print_data):
                print("GUI LIST", gui_list)
                print("Nr. GUIs for Testing:", len(gui_list))

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

                    dataX.append([TOKENS__CHAR_INT[char] for char in gui_subpart])
                    dataY.append(TOKENS__CHAR_INT[gui_list[j][i]])

            print("Nr. Test Sampels:", len(dataX))

            # reshapes X to be [samples, time steps, features]
            X = np.reshape(dataX, (len(dataX), MAX_LENGTH_GUI_REPRESENTATION, 1))

            # one hot encodes the output variable
            y = np_utils.to_categorical(dataY)

            print("Test Data Prepared")

            return X, y

    # function to find best hyperparameter, number of epochs and afterwards train model
    def hypertune_and_train_model(self):
        # Hypertune model using keras-tuner and find best:
        #   - embedding dimension for embedding layer
        #   - number of neurons for each lstm layer
        #   - number of lstm layers
        #   - best learning rate for optimizer

        # see: https://www.tensorflow.org/tutorials/keras/keras_tuner
        # Uses Hyperband tuner : https://medium.com/criteo-engineering/hyper-parameter-optimization-algorithms-2fe447525903
        # modify "build_model_for_param_search" function to change hyperparam range
        if (self.test_run):
            print("This is a TEST run")
        self.tuner = Hyperband(build_model_for_param_search,
                               objective=keras_tuner.Objective("val_loss", direction="min"),
                               max_epochs=10,
                               # executions_per_trial=3,
                               directory=self.conf.path_trained_models + "/tuning/",
                               project_name=self.folder_name)

        batch_size = self.batch_size
        text, x, y = self.prepare_training_data()
        x_val, y_val = self.prepare_validation_data()

        # callback function for hyperparameter tuning
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, mode="min")
        self.tuner.search(x, y, batch_size=batch_size, validation_data=(x_val, y_val),
                          callbacks=[stop_early])

        # safe results in file
        best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        with open(self.conf.PATH_ROOT + "language_model/hyper_param/" + self.folder_name + ".json", 'w') as f:
            dict = {
                'embedding_dim': best_hps.get('embedding_dim'),
                'num_of_neurons': best_hps.get('num_of_neurons'),
                'num_of_layers': best_hps.get('num_of_layers') + 1,
                'learning_rate': best_hps.get('learning_rate')
            }
            f.write(json.dumps(dict))
            f.close()

        # print result on console
        self.best_values = best_hps
        string = f"""
        The hyperparameter search is complete. 
        Embedding dimension: {best_hps.get('embedding_dim')} .
        Number of neurons: {best_hps.get('num_of_neurons')}
        Number of layers:{best_hps.get('num_of_layers') + 1} 
        Learning rate:{best_hps.get('learning_rate')} 
        """
        print(string)

        # Build model using prior found hyper-parameters. Then find best number of epochs to train for
        # Maximum nr. of epochs is set to 30
        self.model = self.tuner.hypermodel.build(best_hps)
        log_dir = "logs/fit/epochs/" + self.folder_name
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # callback function for finding epochs
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, mode="min")
        history = self.model.fit(x, y, batch_size=batch_size, epochs=30, validation_data=(x_val, y_val),
                                 callbacks=[tensorboard_callback, stop_early])

        val_loss_per_epoch = history.history['val_loss']
        best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))

        # Train model once again using best hyperparam and epoch number
        self.model = self.tuner.hypermodel.build(best_hps)
        log_dir = "logs/fit/" + self.folder_name
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        history = self.model.fit(x, y, batch_size=batch_size, epochs=best_epoch, validation_data=(x_val, y_val),
                                 callbacks=[tensorboard_callback])

        path2file = self.directory + self.name
        self.model.save_weights(path2file)
        print('Model Completed')

    # function to train model with hyperparameter set in gui2lm/gui2lm/language_model/hyper_param/best_hp.json
    # Number of epochs is set to 10
    def train_model(self):
        epochs = 10
        batch_size = self.batch_size
        text, x, y = self.prepare_training_data()
        x_val, y_val = self.prepare_validation_data()
        with open(self.conf.PATH_ROOT + "language_model/hyper_param/best_hp.json") as f:
            parameters = json.load(f)
            self.model = build_model_from_dict(parameters)
        try:
            # Create target Directory
            os.mkdir(self.directory)
            print("Directory ", self.directory, " Created ")
        except FileExistsError:
            print("Directory ", self.directory, " already exists")
        path2file = self.directory + self.name
        self.model.save_weights(path2file)
        log_dir = "logs/fit/" + self.folder_name
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        history = self.model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val),
                                 callbacks=[tensorboard_callback])
        self.model.save_weights(path2file)
        print('Model Trained')

    # Function to calculate test cross-entropy and accuracy
    def test_model(self):
        batch_size = self.batch_size
        x, y = self.prepare_test_data()
        with open(self.conf.PATH_ROOT + "language_model/hyper_param/best_hp.json") as f:
            parameters = json.load(f)
            self.model = build_model_from_dict(parameters)
        try:
            # Create target Directory
            os.mkdir(self.directory)
            print("Directory ", self.directory, " Created ")
        except FileExistsError:
            print("Directory ", self.directory, " already exists")
        self.model.load_weights(self.directory + self.name)
        history = self.model.evaluate(x, y, batch_size=batch_size)
        print('Model Evaluated')
        print("test loss, test acc:", history)

    # different training function to save model after every epoch, but that does not log learning curve

    # def train_model_and_save_after_every_epoch(self):
    #     epochs = 30
    #     batch_size = self.batch_size
    #     text, x, y = self.prepare_training_data()
    #     x_val, y_val = self.prepare_validation_data()
    #     with open(self.conf.PATH_ROOT + "language_model/hyper_param/best_hp.json") as f:
    #         parameters = json.load(f)
    #         self.model = build_model_from_dict(parameters)
    #     for epoch in range(epochs):
    #         try:
    #             # Create target Directory
    #             os.mkdir(self.directory)
    #             print("Directory ", self.directory, " Created ")
    #         except FileExistsError:
    #             print("Directory ", self.directory, " already exists")
    #         path2file = self.directory + self.name + "_epoch_" + str(epoch)
    #         self.model.save_weights(path2file)
    #         log_dir = "logs/fit/" + self.folder_name
    #         tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    #         history = self.model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val),
    #                                  callbacks=[tensorboard_callback])
    #         self.model.save_weights(path2file)
    #         print('Epoch ' + str(epoch) + "")

    # generate text via NLG approach
    def generating_text(self, seed: Text, actual_sentence: Text):
        with open(self.conf.PATH_ROOT + "language_model/hyper_param/best_hp.json") as f:
            parameters = json.load(f)
            self.model = build_model_from_dict(parameters)
        self.model.load_weights(self.directory + self.name)
        print()
        returns = {}
        # start_index = random.randint(0, len(text) - self.maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print("...Diversity:", diversity)

            sentence = seed.ljust(MAX_LENGTH_GUI_REPRESENTATION, "_")
            generated = seed

            print('...Generating with seed: "' + seed + '"')

            next_char = ""
            # prevent and endless loop
            i = 0
            pattern = [[TOKENS__CHAR_INT[char] for char in sentence]]
            while next_char != Tokens().token2char[Tokens.end_token] and len(
                    seed) + i + 1 < MAX_LENGTH_GUI_REPRESENTATION:
                x = numpy.reshape(pattern, (1, MAX_LENGTH_GUI_REPRESENTATION, 1))
                prediction = self.model.predict(x, verbose=0)
                next_index = sample(prediction, diversity)
                next_char = TOKENS__INT_CHAR[next_index]
                # next_char =  TOKENS__INT_CHAR[numpy.argmax(prediction)]
                sentence = sentence[0:len(seed) + i] + next_char
                generated += next_char
                # add padding
                sentence = sentence.ljust(MAX_LENGTH_GUI_REPRESENTATION, "_")
                pattern = [[TOKENS__CHAR_INT[char] for char in sentence]]
                i += 1

            print("...Sentence : ", actual_sentence)
            print("...Generated: ", generated)
            returns[diversity] = generated
        return returns

    # generate text via forced teaching
    def generating_text_forced_teaching_v3(self, sentence: Text):
        with open(self.conf.PATH_ROOT + "language_model/hyper_param/best_hp.json") as f:
            parameters = json.load(f)
            self.model = build_model_from_dict(parameters)
        self.model.load_weights(self.directory + self.name)
        print()
        returns = {}

        for diversity in [0.2, 0.5, 1.0]:
            print("...Diversity:", diversity)

            cubes = sentence.split(Tokens().token2char[Tokens().split])
            seed = cubes[0] + Tokens().token2char[Tokens().split]
            generated = Tokens().token2char[Tokens().start_token] + Tokens().token2char[Tokens().split]
            next_char = ""

            print('...Forced teaching with Sentence: "' + sentence + '"')
            for cube in cubes[1:len(cubes)]:
                i = 0
                generated_cube = ""
                seed_plus_generated_cube = (seed + generated_cube).ljust(MAX_LENGTH_GUI_REPRESENTATION, "_")
                while next_char != Tokens().token2char[Tokens.split]:
                    pattern = [[TOKENS__CHAR_INT[char] for char in seed_plus_generated_cube]]
                    x = numpy.reshape(pattern, (1, MAX_LENGTH_GUI_REPRESENTATION, 1))
                    prediction = self.model.predict(x, verbose=0)
                    next_index = sample(prediction, diversity)
                    next_char = TOKENS__INT_CHAR[next_index]
                    generated_cube += next_char

                    seed_plus_generated_cube = (seed + generated_cube).ljust(MAX_LENGTH_GUI_REPRESENTATION, "_")
                seed += cube + Tokens().token2char[Tokens().split]
                generated += generated_cube
                next_char = ""

            print()
            print("...Sentence : ", sentence)
            print("...Generated: ", generated)
            print()
            print()
            returns[diversity] = generated

        return returns
