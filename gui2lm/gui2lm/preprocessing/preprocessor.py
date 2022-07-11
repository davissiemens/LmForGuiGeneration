import io
import json
import random

import numpy as np
import shutil

from gui2lm.gui2lm import utils
from gui2lm.gui2lm.data_abstracting.configuration.conf import Configuration
from gui2lm.gui2lm.language_model.printer import format_to_pretty_print_without_compare
from gui2lm.gui2lm.preprocessing.tokens import Tokens


class Preprocessor:

    def __init__(self, conf: Configuration, tokens: Tokens):
        self.conf = conf
        self.token_config = tokens
        self.count_filter = 0

    def preprocess_abstracted_gui(self, gui_number: int):
        dataset = self.conf.path_abstraction + "Y" + str(self.conf.number_splits_y) \
                  + "X" + str(self.conf.number_splits_x) + "/"
        file_name = dataset + str(gui_number) + ".json"
        with open(file_name, 'r', encoding='utf8') as file:
            ui_json = json.load(file)
            char_string = self.abstraction2char_string(self.conf, ui_json)
            readable_string = self.abstraction2readable_string(self.conf, ui_json)
        with open(self.conf.path_preproc_text + "Y" + str(self.conf.number_splits_y) + "X" + str(
                self.conf.number_splits_x) + "/single_guis/" + str(gui_number), 'w') as write_f:
            write_f.write(readable_string + "\n")
            write_f.write(char_string + "\n")
            write_f.write(format_to_pretty_print_without_compare(char_string))

    # method to identify GUIs afterwards if needed
    def find_gui_by_preprocessed_string(self, char_string):
        conf = self.conf

        dataset = conf.path_abstraction + "Y" + str(conf.number_splits_y) + "X" + str(conf.number_splits_x) + "/"
        file_names = []
        for file_name in utils.iter_files_in_dir(dataset, ending='.json'):
            with open(dataset + file_name, 'r', encoding='utf8') as file_1:
                ui_json = json.load(file_1)
                if conf.filter_guis & ui_json["metadata"]["is_advertisement"]:
                    self.count_filter += 1
                    continue
                if char_string == self.abstraction2char_string(conf, ui_json):
                    file_names.append(file_name)
        return file_names

    def count_test_data(self, print_data=False):
        filename = "Y" + str(self.conf.number_splits_y) + "X" + str(self.conf.number_splits_x) + "/test"
        with io.open(self.conf.path_preproc_text + filename, encoding="utf-8") as f:
            text = f.read()
            print("length total", len(text))
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
            count = 0;
            for gui in gui_list:
                split = gui.split()
                for word in split:
                    if (len(word) == 1):
                        if (word.isnumeric()):
                            count += 1
                        if (word == 'A') | (word == 'B') | (word == 'C'):
                            count += 1
            print(count)
            return count

            # for j in range(0, len(gui_list)):
            #     for i in range(1, len(gui_list[j])):
            #         gui_subpart = gui_list[j][0: i].ljust(MAX_LENGTH_GUI_REPRESENTATION, "_")
            #         sentence_list.append(gui_subpart)
            #         next_chars_list.append(gui_list[j][i])
            #
            #         dataX.append([TOKENS__CHAR_INT[char] for char in gui_subpart])
            #         dataY.append(TOKENS__CHAR_INT[gui_list[j][i]])
            #
            # print("Nr. Test Sampels:", len(dataX))
            #
            # # reshapes X to be [samples, time steps, features]
            # X = np.reshape(dataX, (len(dataX), MAX_LENGTH_GUI_REPRESENTATION, 1))
            #
            # # one hot encodes the output variable
            # y = np_utils.to_categorical(dataY)
            #
            # print("Test Data Prepared")

    def list_all_test_guis(self, ):
        conf = self.conf
        file1 = open(conf.path_preproc_text + "Y" + str(conf.number_splits_y) + "X" + str(
            conf.number_splits_x) + "/test", 'r')
        Lines = file1.readlines()
        path2target = "/Users/davis/PycharmProjects/LmForGuiGeneration/gui2lm/gui2lm/resources/test_guis"
        path2combined = "/Users/davis/PycharmProjects/LmForGuiGeneration/gui2lm/gui2lm/resources/combined"
        path2semantic = "/Users/davis/PycharmProjects/LmForGuiGeneration/gui2lm/gui2lm/resources/semantic_annotations"
        count = 0
        names = []
        for line in Lines:
            count += 1
            gui_nr = self.find_gui_by_preprocessed_string(line.strip()).split(".")[0]
            print(gui_nr)
            shutil.copy(path2combined + "/" + str(gui_nr) + ".jpg", path2target + "/" + str(gui_nr) + ".jpg")
            shutil.copy(path2semantic + "/" + str(gui_nr) + ".png", path2target + "/" + str(gui_nr) + ".png")

    def find_gui_by_readable_string(self, readable_string):
        conf = self.conf

        dataset = conf.path_abstraction + "Y" + str(conf.number_splits_y) + "X" + str(conf.number_splits_x) + "/"

        for file_name in utils.iter_files_in_dir(dataset, ending='.json'):
            with open(dataset + file_name, 'r', encoding='utf8') as file_1:
                ui_json = json.load(file_1)
                if conf.filter_guis & ui_json["metadata"]["is_advertisement"]:
                    self.count_filter += 1
                    continue
                # Readable dataset is for debugging purposes
                if readable_string == self.abstraction2readable_string(conf, ui_json):
                    return file_name
        return "No GUI Found"

    def preprocess_abstracted_gui_dataset(self):
        conf = self.conf

        dataset = conf.path_abstraction + "Y" + str(conf.number_splits_y) + "X" + str(conf.number_splits_x) + "/"

        readable_strings = []
        char_strings = []

        for file_name in utils.iter_files_in_dir(dataset, ending='.json'):
            with open(dataset + file_name, 'r', encoding='utf8') as file_1:
                ui_json = json.load(file_1)
                # filter advertisements
                # TODO more filtering
                if conf.filter_guis & ui_json["metadata"]["is_advertisement"]:
                    self.count_filter += 1
                    continue
                # Readable dataset is for debugging purposes
                readable_strings.append(self.abstraction2readable_string(conf, ui_json))
                # char string is used by the langauge model
                char_strings.append(self.abstraction2char_string(conf, ui_json))

                # # Readable dataset is for debugging purposes
                # readable_string = self.abstraction2readable_string(conf, ui_json)
                # # char string is used by the langauge model
                # char_string = self.abstraction2char_string(conf, ui_json)
                #

        # shuffle gui data to guarantee an homogenous data set
        shuffle = list(zip(readable_strings, char_strings))
        random.shuffle(shuffle)
        readable_strings, char_strings = zip(*shuffle)

        number_guis = len(readable_strings)

        # Split dataset in train/val/test data

        r_train, r_validate, r_test = np.split(readable_strings,
                                               [int(number_guis * 0.7), int(number_guis * 0.85)])
        c_train, c_validate, c_test = np.split(char_strings,
                                               [int(number_guis * 0.7), int(number_guis * 0.85)])

        # Write dataset to files
        with open(conf.path_preproc_text + "Y" + str(conf.number_splits_y) + "X" + str(
                conf.number_splits_x) + "_readable/train", 'w') as write_f:
            for element in r_train:
                write_f.write(element + "\n")
        with open(conf.path_preproc_text + "Y" + str(conf.number_splits_y) + "X" + str(
                conf.number_splits_x) + "_readable/validate", 'w') as write_f:
            for element in r_validate:
                write_f.write(element + "\n")
        with open(conf.path_preproc_text + "Y" + str(conf.number_splits_y) + "X" + str(
                conf.number_splits_x) + "_readable/test", 'w') as write_f:
            for element in r_test:
                write_f.write(element + "\n")

        with open(conf.path_preproc_text + "Y" + str(conf.number_splits_y) + "X" + str(
                conf.number_splits_x) + "/train", 'w') as write_f:
            for element in c_train:
                write_f.write(element + "\n")
        with open(conf.path_preproc_text + "Y" + str(conf.number_splits_y) + "X" + str(
                conf.number_splits_x) + "/validate", 'w') as write_f:
            for element in c_validate:
                write_f.write(element + "\n")
        with open(conf.path_preproc_text + "Y" + str(conf.number_splits_y) + "X" + str(
                conf.number_splits_x) + "/test", 'w') as write_f:
            for element in c_test:
                write_f.write(element + "\n")

        # write_f.write(readable_string)
        # write_f.write('\n')
        #
        # write_f_chars.write(char_string)
        # write_f_chars.write('\n')
        print("Nr. Filtered: ", self.count_filter)

    def abstraction2readable_string(self, conf, ui_json):
        data_string = self.token_config.start_token + self.token_config.split
        for y in range(0, conf.number_splits_y):
            for x in range(0, conf.number_splits_x):
                square = ui_json[str(y)][str(x)]
                if len(square) == 0:
                    data_string += self.token_config.no_element
                data_string += self.extract_square_elements_to_tokens(square)
                data_string += self.token_config.split
            data_string += self.token_config.line_break + self.token_config.split
        # remove last redundant line break
        data_string = data_string[: -2]
        data_string += self.token_config.end_token
        return data_string

    def extract_square_elements_to_tokens(self, square):
        square_string = ""
        square_string_list = []
        for element in square:
            square_string_list.append(str(element))
        # Sort tokens alphabetical
        square_string_list.sort()
        for element in square_string_list:
            square_string += element + self.token_config.concatenation
        # remove last redundant concatenation
        square_string = square_string[: -1]
        return square_string

    def abstraction2char_string(self, conf, ui_json):
        token_config = Tokens()
        token2char = token_config.token2char
        data_string = token2char[self.token_config.start_token] + token2char[self.token_config.split]
        for y in range(0, conf.number_splits_y):
            for x in range(0, conf.number_splits_x):
                square = ui_json[str(y)][str(x)]
                if len(square) == 0:
                    data_string += token2char[self.token_config.no_element]
                data_string += self.extract_square_elements_to_chars(square)
                data_string += token2char[self.token_config.split]
            data_string += token2char[self.token_config.line_break] + token2char[self.token_config.split]
        # remove last redundant line break
        data_string = data_string[: -2]
        data_string += token2char[self.token_config.end_token]
        return data_string

    def extract_square_elements_to_chars(self, square):
        token2char = self.token_config.token2char
        square_string = ""
        square_string_list = []
        for element in square:
            square_string_list.append(token2char[str(element)])
        # Sort tokens alphabetical
        square_string_list.sort()
        for element in square_string_list:
            square_string += element + token2char[self.token_config.concatenation]
        # remove last redundant concatenation
        square_string = square_string[: -1]
        return square_string

    def get_max_length(self):
        path_to_data = self.conf.path_preproc_text
        filename = "Y" + str(self.conf.number_splits_y) + "X" + str(self.conf.number_splits_x)
        with io.open(path_to_data + filename, encoding="utf-8") as f:
            text = f.read()
            gui_list = text.split("\n")
            max_length = len(max(gui_list, key=len))
            print("Max length is:", max_length)
