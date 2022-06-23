from gui2lm.gui2lm.data_abstracting.configuration.conf import Configuration
from gui2lm.gui2lm.preprocessing.preprocessor import Preprocessor
from gui2lm.gui2lm.preprocessing.tokens import Tokens

if __name__ == '__main__':
    preprocessor = Preprocessor(Configuration(), Tokens())
    # preprocessor.preprocess_abstracted_gui(25675)
    # print(preprocessor.find_gui_by_preprocessed_string("< 1&2&4 2&4 4 | 0 1&2 0 | 0 2 0 | 0 1&2 0 >"))
    # print(preprocessor.find_gui_by_readable_string("START Image Text Icon | Empty Empty Empty | Empty WebView Empty | Empty Empty Empty END"))
    # print(preprocessor.find_gui_by_readable_string("START Empty Empty Empty | Empty WebView Empty | Empty Empty Empty | Empty TextButton Empty END"))
    # print(preprocessor.find_gui_by_readable_string("START Empty Empty Image | Empty Empty Empty | Empty WebView Empty | Empty Empty Empty END"))
    # print(preprocessor.find_gui_by_readable_string("START Icon&Text Empty Empty | Empty Empty Empty | Empty WebView Empty | Empty WebView Empty END"))
    print(preprocessor.find_gui_by_preprocessed_string("< 1&2&3 1 3 | 1&2&3 1 3 | 1&2&3 1&7 3 | 1&2&3 1 2&3 >"))
    # preprocessor.preprocess_abstracted_gui_dataset()
    # print(Tokens().char2int())
    # print(Tokens().int2char())
    # preprocessor.get_max_length()