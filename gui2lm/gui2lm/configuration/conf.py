import os
from typing import Text, Optional, List


class Configuration():
    # TODO Change project directory before runnning
    PATH_ROOT = "/Users/davis/PycharmProjects/LmForGuiGeneration/gui2lm/gui2lm/"

    def __init__(self, path_guis: Optional[Text] = os.path.join(PATH_ROOT, 'resources/combined_small/'),
                 path_trained_models: Optional[Text] = os.path.join(PATH_ROOT, 'models/test'),
                 path_semantic: Optional[Text] = os.path.join(PATH_ROOT, "resources/semantic_annotations_small/"),
                 path_semantic_all: Optional[Text] = os.path.join(PATH_ROOT, "resources/semantic_annotations/"),
                 path_abstractions: Optional[Text] = os.path.join(PATH_ROOT, "resources/abstractions/"),
                 path_count_csv: Optional[Text] = os.path.join(PATH_ROOT, "resources/"),
                 path_preproc_text: Optional[Text] = os.path.join(PATH_ROOT, "resources/preprocessed/"),
                 path_preproc_text_small: Optional[Text] = os.path.join(PATH_ROOT, "resources/preprocessed_small/"),
                 path_app_details: Optional[Text] = os.path.join(PATH_ROOT, "resources/app_details.csv"),
                 path_ui_details: Optional[Text] = os.path.join(PATH_ROOT, "resources/ui_details.csv"),
                 path_models: Optional[Text] = 'models/test',
                 filter_guis: Optional[bool] = True,
                 number_splits_x: Optional[int] = 3,
                 number_splits_y: Optional[int] = 4):
        self.number_splits_x = number_splits_x
        self.number_splits_y = number_splits_y
        self.path_guis = path_guis
        self.path_trained_models = path_trained_models
        self.path_semantic = path_semantic
        self.path_abstraction = path_abstractions
        self.path_count_csv = path_count_csv
        self.path_semantic_all = path_semantic_all
        self.path_preproc_text = path_preproc_text
        self.path_preproc_text_small = path_preproc_text_small
        self.path_app_details = path_app_details
        self.path_ui_details = path_ui_details
        self.path_models = path_models
        self.filter_guis = filter_guis
