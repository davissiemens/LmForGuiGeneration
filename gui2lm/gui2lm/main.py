from gui2lm.gui2lm.configuration.conf import Configuration
from gui2lm.gui2lm.data_abstracting.abstractor import Abstractor
from gui2lm.gui2lm.data_abstracting.filter import Filter
from gui2lm.gui2lm.language_model.languagemodel_with_hypertuning_V2 import LanguageModel_WithParamTuning
from gui2lm.gui2lm.language_model.lm_optimizer import write_generation_to_file
from gui2lm.gui2lm.preprocessing.preprocessor import Preprocessor
from gui2lm.gui2lm.preprocessing.tokens import Tokens

if __name__ == '__main__':

    conf = Configuration()

    name = input("Enter Folder/Model Name - Type nothing to use standard model: ")
    if name==None or name == "":
      name = "Overreach1"

    # Run code below to abstract and later preprocess GUI dataset
    # This is only needed if dataset was not pulled from GutHub or the grid finesse (x & y value) was modified

    # filt = Filter(app_meta_data_path=conf.path_app_details, app_ui_details_path=conf.path_ui_details)
    # abstractor = Abstractor()
    # abstractor.abstract_semantic_gui_dataset(conf, filt)
    # preprocessor = Preprocessor(conf, Tokens())
    # preprocessor.preprocess_abstracted_gui_dataset()


    # Run code below to train the language model. When doing so it is recommended to type in a name.
    # Otherwise the standard model gets overwritten
    #
    # name = input("Enter Folder/Model Name: ")
    # lstm = LanguageModel_WithParamTuning(conf, name)
    # lstm.train_model()

    # example code to generate GUIs

    lstm = LanguageModel_WithParamTuning(conf, name)
    # change to true if forced teaching method shall be used
    forced_teaching = False
    # change to false if generation without context are done -> seed = ""
    compare_to_original_gui = True

    original_gui = "< 1&2&3 1 3 | 2&3 1 0 | 2&3 1 0 | 2 1&6 0 >"
    seed = "< 1&2&3 1 3"

    # writes text generations to gui2lm/gui2lm/language_model/text_generations/[Name of Model], with filename as name
    write_generation_to_file(original_gui, seed, forced_teaching, lstm, name, compare_to_original_gui)
