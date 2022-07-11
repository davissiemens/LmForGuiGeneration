from gui2lm.gui2lm.data_abstracting.configuration.conf import Configuration
from gui2lm.gui2lm.preprocessing.preprocessor import Preprocessor
from gui2lm.gui2lm.preprocessing.tokens import Tokens

if __name__ == '__main__':
    preprocessor = Preprocessor(Configuration(), Tokens())
    # preprocessor.list_all_test_guis()
    preprocessor.preprocess_abstracted_gui(68322)
    # preprocessor.preprocess_abstracted_gui(26141)
    # print(*preprocessor.find_gui_by_preprocessed_string("< 1&3 1&2 1&2&3 | 1 1&2&7 2 | 1&3 1&2&3&4&6 2&3 | 1&3 2&3&B 1&2&3 >"))
    # print(*preprocessor.find_gui_by_preprocessed_string("< 0 1&4 0 | 0 4 0 | 0 4 0 | 0 4 4 >"))
    # print(preprocessor.find_gui_by_readable_string("START Image Text Icon | Empty Empty Empty | Empty WebView Empty | Empty Empty Empty END"))
    # print(preprocessor.find_gui_by_readable_string("START Empty Empty Empty | Empty WebView Empty | Empty Empty Empty | Empty TextButton Empty END"))
    # print(preprocessor.find_gui_by_readable_string("START Empty Empty Image | Empty Empty Empty | Empty WebView Empty | Empty Empty Empty END"))
    # print(preprocessor.find_gui_by_readable_string("START Icon&Text Empty Empty | Empty Empty Empty | Empty WebView Empty | Empty WebView Empty END"))
    # print(preprocessor.find_gui_by_preprocessed_string("< 1&2&3 1 3 | 1&2&3 1 3 | 1&2&3 1&7 3 | 1&2&3 1 2&3 >"))
    # preprocessor.preprocess_abstracted_gui_dataset()
    # print(Tokens().char2int())
    # print(Tokens().int2char())
    # preprocessor.get_max_length()