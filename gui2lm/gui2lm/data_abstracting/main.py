# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from gui2lm.gui2lm.data_abstracting.abstractor import Abstractor
from gui2lm.gui2lm.data_abstracting.configuration.conf import Configuration
from gui2lm.gui2lm.data_abstracting.filter import Filter


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    conf = Configuration()
    filt = Filter(app_meta_data_path=conf.path_app_details, app_ui_details_path=conf.path_ui_details)
    abstractor = Abstractor()
    abstractor.abstract_semantic_gui_dataset(conf, filt)
    #
    # generator = LabelCounter()
    # generator.write_leaf_count_to_file(conf)

    # print(generator.count_labels_in_dataset(conf))
    # for label in labels:
    #     print(label)

    #
    # for file_name in utils.iter_files_in_dir(conf.path_dsls, ending='.json'):
    #     # print(file_name)
    #     # generator.text_extraction_from_file_v2(conf.path_semantic, file_name, conf, filt)
    #     generator.label_extraction_from_file(conf.path_semantic, file_name, conf, filt)
    #     # print(dat
    # print('Done')

