from gui2lm.gui2lm.data_abstracting.configuration.conf import Configuration
# from gui2lm.gui2lm.language_model.lstm import LSTM
from gui2lm.gui2lm.language_model.languagemodel_with_hypertuning_V2 import LanguageModel_WithParamTuning
from gui2lm.gui2lm.preprocessing.tokens import Tokens

if __name__ == '__main__':
    # tokens=Tokens()
    # for i in tokens.tokens:
    #     print(str(i)+" "+str(tokens.tokens[i]))
    name = input("Enter Folder/Model Name: ")
    conf = Configuration()
    lstm = LanguageModel_WithParamTuning(conf, name)
    # lstm.hypertune_and_train_model()
    lstm.generating_text_forced_teaching("< 1&2&4 2&4 4 | 0 1&2 0 | 0 2 0 | 0 1&2 0 >")
    # lstm.prepare_training_data()
    # lstm.find_param()
    # lstm.find_epochs()
    # < 1&2&4 2&4 4 | 0 1&2 0 | 0 2 0 | 0 1&2 0 >

    sentence = "< 1&2&4 2&4 4 "
    lstm.generating_text(sentence)
    sentence = "< 1&2&4 2&4 4 | 0 1&2"
    lstm.generating_text(sentence)
    sentence = "< 1&2&4 2&4 4 | 0 1&2 0 | 0 2"
    lstm.generating_text(sentence)
    sentence = "< 1&2&4 2&4 4 | 0 1&2 0 | 0 2 0 | 0 1&2"
    lstm.generating_text(sentence)
    sentence = "< 1"
    lstm.generating_text(sentence)



