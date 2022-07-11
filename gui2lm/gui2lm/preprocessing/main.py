from gui2lm.gui2lm.data_abstracting.configuration.conf import Configuration
from gui2lm.gui2lm.preprocessing.preprocessor import Preprocessor
from gui2lm.gui2lm.preprocessing.tokens import Tokens

if __name__ == '__main__':
    preprocessor = Preprocessor(Configuration(), Tokens())
    preprocessor.preprocess_abstracted_gui(68322)
