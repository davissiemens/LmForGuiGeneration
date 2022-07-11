from gui2lm.gui2lm.configuration.conf import Configuration
from gui2lm.gui2lm.data_abstracting.abstractor import Abstractor
from gui2lm.gui2lm.data_abstracting.filter import Filter

# run this script to abstract semantic dataset
# Grid finesse can be configured in configuration class
if __name__ == '__main__':
    conf = Configuration()
    filt = Filter(app_meta_data_path=conf.path_app_details, app_ui_details_path=conf.path_ui_details)
    abstractor = Abstractor()
    abstractor.abstract_semantic_gui_dataset(conf, filt)