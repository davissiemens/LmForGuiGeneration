import os

from gui2lm.gui2lm.data_abstracting.configuration.conf import Configuration
# from gui2lm.gui2lm.language_model.lstm import LSTM
from gui2lm.gui2lm.language_model.languagemodel_with_hypertuning_V2 import LanguageModel_WithParamTuning
from gui2lm.gui2lm.language_model.printer import format_generation_output, format_generation_output_forced_learning

NUM_BATCHES = 13500


def write_console_to_file(hypertune=False):
    # TODO readme create file and conf run configurations
    with open(conf.PATH_ROOT + "console_output/lm_output") as file:
        L = []
        lastOne = False
        for line in file:
            lstrip = line.lstrip()
            first_word = lstrip.split(" ")[0]
            split_of_first_word = first_word.split("/")
            # Very complicated "if statement" to delete the messed up lines in keras console output
            if len(split_of_first_word) > 1:
                if split_of_first_word[0].isdigit() & split_of_first_word[1].isdigit():
                    # Test if first token in line is [<148]/148
                    if split_of_first_word[0]:
                        if split_of_first_word[1]:
                            if int(split_of_first_word[0]) < NUM_BATCHES & int(split_of_first_word[1]) == NUM_BATCHES:
                                continue
                            if int(split_of_first_word[0]) == NUM_BATCHES & int(split_of_first_word[1]) == NUM_BATCHES:
                                if not lastOne:
                                    lastOne = True
                                    L.append(line.rstrip())
                                    continue
                                else:
                                    L[len(L) - 1] = line.rstrip()
                                    continue
            lastOne = False
            L.append(line.rstrip())
    dir = ""
    if (hypertune):
        dir = "hypertune/"
    with open(conf.PATH_ROOT + "language_model/console_output/" + dir + name, 'w') as f:
        for item in L:
            f.write("%s\n" % item)


def write_generation_to_file(sentence, seed, forced_teaching, lstm, name, compare=True):
    if forced_teaching:
        ui_name = sentence
        try:
            # Create target Directory
            os.mkdir(conf.PATH_ROOT + "language_model/text_generations/" + name)
            print("Directory ", conf.PATH_ROOT + "language_model/text_generations/" + name, " Created ")
        except FileExistsError:
            print("Directory ", conf.PATH_ROOT + "language_model/text_generations/" + name,
                  " already exists")
        try:
            # Create target Directory
            os.mkdir(conf.PATH_ROOT + "language_model/text_generations/" + name + "/forced/")
            print("Directory ", conf.PATH_ROOT + "language_model/text_generations/" + name + "/forced/", " Created ")
        except FileExistsError:
            print("Directory ", conf.PATH_ROOT + "language_model/text_generations/" + name + "/forced/",
                  " already exists")
        with open(conf.PATH_ROOT + "language_model/text_generations/" + name + "/forced/" + ui_name, 'w') as f:
            f.write(
                format_generation_output_forced_learning(lstm.generating_text_forced_teaching_v3(sentence), sentence,
                                                         compare))
    else:
        ui_name = seed
        try:
            # Create target Directory
            os.mkdir(conf.PATH_ROOT + "language_model/text_generations/" + name)
            print("Directory ", conf.PATH_ROOT + "language_model/text_generations/" + name, " Created ")
        except FileExistsError:
            print("Directory ", conf.PATH_ROOT + "language_model/text_generations/" + name,
                  " already exists")
        try:
            # Create target Directory
            os.mkdir(conf.PATH_ROOT + "language_model/text_generations/" + name + "/seed/")
            print("Directory ", conf.PATH_ROOT + "language_model/text_generations/" + name + "/seed/", " Created ")
        except FileExistsError:
            print("Directory ", conf.PATH_ROOT + "language_model/text_generations/" + name + "/seed/",
                  " already exists")
        with open(conf.PATH_ROOT + "language_model/text_generations/" + name + "/seed/" + ui_name, 'w') as f:
            f.write(format_generation_output(lstm.generating_text(seed, sentence), seed, sentence, compare))


if __name__ == '__main__':
    # name = input("Enter Folder/Model Name: ")
    name = "Overreach1"
    # name2 = "V3_Run1_Train"
    conf = Configuration()
    lstm = LanguageModel_WithParamTuning(conf, name)
    # lstm.test_model()
    # lstm2 = LanguageModel_WithParamTuning(conf, name2)
    # lstm.train_model()
    # write_console_to_file(hypertune=True)

    # forced_teaching = True
    # write_generation_to_file("< 1&2&4 2&4 4 | 0 1&2 0 | 0 2 0 | 0 1&2 0 >", "", forced_teaching, lstm, name)
    forced_teaching = False
    # write_generation_to_file("", "< 3", forced_teaching, lstm, name, False)
    # write_generation_to_file("", "< 2", forced_teaching, lstm, name, False)
    # write_generation_to_file("< 2&3 1 0 | 2&3 1 0 | 2&3 1 0 | 2&3 1 0 >", "< 2&3 1 0 | 2&3 1 0 |", forced_teaching, lstm, name)
    # write_generation_to_file("< 3 1 0 | 3 1 0 | 3 1 0 | 3 1 0 >", "< 3 1 0 ", forced_teaching, lstm, name)
    # write_generation_to_file("< 3 1 0 | 3 1 0 | 3 1 0 | 3 1 0 >", "< 3 1 0 | 3 1 0 |", forced_teaching, lstm, name)
    # write_generation_to_file("< 2&3 1&5 2 | 3 5 2 | 4 4 4 | 4 3&4 4 >", "< 2&3 1&5 2 |", forced_teaching, lstm, name)
    write_generation_to_file("< 1&3 1&2 1&2&3 | 1 1&2&7 2 | 1&3 1&2&3&4&6 2&3 | 1&3 2&3&B 1&2&3 >", "< 1&3 1&2 1&2&3 ",
                             forced_teaching, lstm, name)
    write_generation_to_file("< 1&3 1&2 1&2&3 | 1 1&2&7 2 | 1&3 1&2&3&4&6 2&3 | 1&3 2&3&B 1&2&3 >",
                             "< 1&3 1&2 1&2&3 | 1 1&2&7 2 ", forced_teaching, lstm, name)
    # write_generation_to_file("< 1&2&3 1 3 | 2&3 1 0 | 2&3 1 0 | 2 1&6 0 >", "< 1&2&3 1 3 | 2&3 1 0 |", forced_teaching, lstm2, name2)
    # write_generation_to_file("< 1&2&4 2&4 4 | 0 1&2 0 | 0 2 0 | 0 1&2 0 >", "< 1&2&4 2&4 4 | 0 1&2 0 |", forced_teaching, lstm, name)
    # write_generation_to_file("< 1&2&4 2&4 4 | 0 1&2 0 | 0 2 0 | 0 1&2 0 >", "< 1&2&4 2&4 4 | 0 1&2 0 | 0 2 0 |", forced_teaching, lstm, name)

    # forced_teaching = True
    # write_generation_to_file("< 1&3 1 4 | 0 4 0 | 0 4 0 | 0 1 0 >", "", forced_teaching, lstm, name)
    # forced_teaching = False
    # write_generation_to_file("<1&3 1 4 | 0 4 0 | 0 4 0 | 0 1 0>", "<", forced_teaching, lstm, name)
    # write_generation_to_file("< 0 1&4 0 | 0 4 0 | 0 4 0 | 0 4 4 >", "< 0 1&4 0 |", forced_teaching, lstm, name, True)
    # write_generation_to_file("< 0 1&4 0 | 0 4 0 | 0 4 0 | 0 4 4 >", "< 0 1&4 0 | 0 4 0 |", forced_teaching, lstm, name, True)

    # write_generation_to_file("< 1&2&3 2&3 2&3 | 2 2 2 | 2 2 2 | 2 1&2&3 2 >", "< 1&2&3 2&3 2&3 | 2 2 2 |", forced_teaching, lstm, name, True)
    # write_generation_to_file("< 1&3 1 3 | 1 1&4 0 | 1 0 0 | 1 6 0 >", "< 1&3 1 3 | 1 1&4 0 |", forced_teaching, lstm, name, True)
    # write_generation_to_file("< 2&3 0 0 | 0 0 0 | 0 6 0 | 0 0 0 >", "< 2&3 0 0 ", forced_teaching, lstm, name, True)
    # write_generation_to_file("< 2&3 0 0 | 0 0 0 | 0 6 0 | 0 0 0 >", "< 2&3 0 0 | 0 0 0 ", forced_teaching, lstm, name, True)
    # write_generation_to_file("< 2&3 0 0 | 0 0 0 | 0 6 0 | 0 0 0 >", "< 2&3 0 0 ", forced_teaching, lstm, name, True)
    # write_generation_to_file("< 2&3 0 0 | 0 0 0 | 0 6 0 | 0 0 0 >", "", True, lstm, name, True)
    # write_generation_to_file("< 2 1 3 | 0 0 0 | 0 6 0 | 0 0 0 >", "< 2 1 3 |", False, lstm, name, True)
    # write_generation_to_file("< 2 1 3 | 0 0 0 | 0 6 0 | 0 0 0 >", "< 2 1 3 | 0 0 0 |", False, lstm, name, True)
    # write_generation_to_file("< 1&2&3 1 3 | 1&2&3 1 3 | 1&2&3 1&7 3 | 1&2&3 1 2&3 >", "< 1&2&3 1 3 |", False, lstm, name, True)
    # write_generation_to_file("< 1&2&3 1 3 | 1&2&3 1 3 | 1&2&3 1&7 3 | 1&2&3 1 2&3 >", "< 1&2&3 1 3 | 1&2&3 1 3", False, lstm, name, True)
    # write_generation_to_file("< 1&2&3 1 3 | 1&2&3 1 3 | 1&2&3 1&7 3 | 1&2&3 1 2&3 >", "< 1&2&3 1 3 | 1&2&3 1 3", True, lstm, name, True)
    # write_generation_to_file("< 2 1 3 | 0 0 0 | 0 6 0 | 0 0 0 >", "<", False, lstm, name, False)
    # write_generation_to_file("< 2 1 3 | 0 0 0 | 0 6 0 | 0 0 0 >", "< 2", False, lstm, name, False)
    # write_generation_to_file("< 2 1 3 | 0 0 0 | 0 6 0 | 0 0 0 >", "< 0", False, lstm, name, False)
    # write_generation_to_file("< 1&3 1 4 | 0 4 0 | 0 4 0 | 0 1 0 >", "< 1&3 1 4 | 0 4 0 |", forced_teaching, lstm, name)
    # write_generation_to_file("< 1&3 1 4 | 0 4 0 | 0 4 0 | 0 1 0 >", "< 1&3 1 4 | 0 4 0 | 0 4 0 |", forced_teaching, lstm, name)
    # write_generation_to_file("< 3 1&2 3 | 0 2&7 0 | 2&3 2 0 | 2&3&4&8 2&4&6&8 1&3&8 >", "", forced_teaching, lstm, name)
    # write_generation_to_file("< 3&5 1 3 | 0 0 0 | 2 0 2&3 | 2 6 0 >", "", forced_teaching, lstm, name)
    # write_generation_to_file("< 1&2&3 1 3 | 2&3 1 0 | 2&3 1 0 | 2 1&6 0 >", "", forced_teaching, lstm, name)
    # #
    # # "< 1&2&4 2&4 4 | 0 1&2 0 | 0 2 0 | 0 1&2 0 >"
    # sentence = "< 1&2&4 2&4 4 | 0 1&2 0 | 0 2 0 | 0 1&2 0 >"
    # seed = "< 1"
    # forced_teaching = False
    # write_generation_to_file(sentence, seed, forced_teaching, lstm, name)
    # seed = "< 1&2&4 2&4 4 "
    # write_generation_to_file(sentence, seed, forced_teaching, lstm, name)
    # seed = "< 1&2&4 2&4 4 | 0 1&2"
    # write_generation_to_file(sentence, seed, forced_teaching, lstm, name)
    # seed = "< 1&2&4 2&4 4 | 0 1&2 0 | 0 2"
    # write_generation_to_file(sentence, seed, forced_teaching, lstm, name)
    # seed = "< 1&2&4 2&4 4 | 0 1&2 0 | 0 2 0 | 0 1&2"
    # write_generation_to_file(sentence, seed, forced_teaching, lstm, name)
    #
    # # "< 3&5 1 3 | 0 0 0 | 2 0 2&3 | 2 6 0 >"
    # sentence = "< 3&5 1 3 | 0 0 0 | 2 0 2&3 | 2 6 0 >"
    # seed = "< 3&5 1"
    # forced_teaching = False
    # write_generation_to_file(sentence, seed, forced_teaching, lstm, name)
    # seed = "< 3&5 1 3 |"
    # write_generation_to_file(sentence, seed, forced_teaching, lstm, name)
    # seed = "< 3&5 1 3 | 0 0 0"
    # write_generation_to_file(sentence, seed, forced_teaching, lstm, name)
    # seed = "< 3&5 1 3 | 0 0 0 | 2 0 2&3"
    # write_generation_to_file(sentence, seed, forced_teaching, lstm, name)
    # seed = "< 3&5 1 3 | 0 0 0 | 2 0 2&3 | 2 6"
    # write_generation_to_file(sentence, seed, forced_teaching, lstm, name)
    #
    # write_console_to_file()
