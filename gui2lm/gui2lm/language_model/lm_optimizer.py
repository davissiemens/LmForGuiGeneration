import os


# from gui2lm.gui2lm.language_model.lstm import LSTM
from gui2lm.gui2lm.configuration.conf import Configuration
from gui2lm.gui2lm.language_model.languagemodel_with_hypertuning_V2 import LanguageModel_WithParamTuning
from gui2lm.gui2lm.language_model.printer import format_generation_output, format_generation_output_forced_learning

NUM_BATCHES = 13500


# def write_console_to_file(hypertune=False):
#     with open(conf.PATH_ROOT + "console_output/lm_output") as file:
#         L = []
#         lastOne = False
#         for line in file:
#             lstrip = line.lstrip()
#             first_word = lstrip.split(" ")[0]
#             split_of_first_word = first_word.split("/")
#             # Very complicated "if statement" to delete the messed up lines in keras console output
#             if len(split_of_first_word) > 1:
#                 if split_of_first_word[0].isdigit() & split_of_first_word[1].isdigit():
#                     # Test if first token in line is [<148]/148
#                     if split_of_first_word[0]:
#                         if split_of_first_word[1]:
#                             if int(split_of_first_word[0]) < NUM_BATCHES & int(split_of_first_word[1]) == NUM_BATCHES:
#                                 continue
#                             if int(split_of_first_word[0]) == NUM_BATCHES & int(split_of_first_word[1]) == NUM_BATCHES:
#                                 if not lastOne:
#                                     lastOne = True
#                                     L.append(line.rstrip())
#                                     continue
#                                 else:
#                                     L[len(L) - 1] = line.rstrip()
#                                     continue
#             lastOne = False
#             L.append(line.rstrip())
#     dir = ""
#     if (hypertune):
#         dir = "hypertune/"
#     with open(conf.PATH_ROOT + "language_model/console_output/" + dir + name, 'w') as f:
#         for item in L:
#             f.write("%s\n" % item)


def write_generation_to_file(sentence, seed, forced_teaching, lstm, name, compare=True):
    conf = Configuration()
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


#  Run script to use hyperparameter optimization
if __name__ == '__main__':
    name = input("Enter Folder/Model Name: ")
    conf = Configuration()
    lstm = LanguageModel_WithParamTuning(conf, name)
    lstm.hypertune_and_train_model()
