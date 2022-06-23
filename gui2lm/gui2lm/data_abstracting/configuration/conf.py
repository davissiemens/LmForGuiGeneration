import os
from typing import Text, Optional, List
from gui2lm.gui2lm.data_abstracting.preprocess import Stemming

class Configuration():
    # TODO Change project directory before runnning
    PATH_ROOT = "/Users/davis/PycharmProjects/LmForGuiGeneration/gui2lm/gui2lm/"
    def __init__(self, path_guis: Optional[Text] = os.path.join(PATH_ROOT, 'resources/combined_small/'),
                 path_dsls: Optional[Text] = os.path.join(PATH_ROOT, 'resources/combined_small/'),
                 path_trained_models: Optional[Text] = os.path.join(PATH_ROOT, 'models/test/'),
                 path_semantic: Optional[Text] = os.path.join(PATH_ROOT, "resources/semantic_annotations_small/"),
                 path_semantic_all: Optional[Text] = os.path.join(PATH_ROOT, "resources/semantic_annotations/"),
                 path_abstractions: Optional[Text] = os.path.join(PATH_ROOT, "resources/abstractions/"),
                 path_count_csv: Optional[Text] = os.path.join(PATH_ROOT, "resources/"),
                 path_preproc_text: Optional[Text] = os.path.join(PATH_ROOT, "resources/preprocessed/"),
                 path_preproc_text_small: Optional[Text] = os.path.join(PATH_ROOT, "resources/preprocessed_small/"),
                 # path_app_details: Optional[Text] = '../../resources/app_details.csv',
                 path_app_details: Optional[Text] = os.path.join(PATH_ROOT, "resources/app_details.csv"),
                 path_ui_details : Optional[Text] = os.path.join(PATH_ROOT, "resources/ui_details.csv"),
                 path_models: Optional[Text] = '/models',
                 path_ui_comp_models: Optional[Text] = '/models/ui_comps/',
                 threshold: Optional[float] = 0.0,
                 tokenized: Optional[bool] = True,
                 preprocessing_stemmed: Optional[bool] = False,
                 preprocessing_stemmer: Optional[Text] = Stemming.KROVETZ,
                 preprocessing_rm_stopwords: Optional[bool] = True,
                 filter_guis: Optional[bool] = True,
                 dir_name_prefix: Optional[Text] = '',
                 text_segments_used: Optional[List[Text]] = None,
                 number_splits_x: Optional[int] = 3,
                 number_splits_y: Optional[int] = 4):
        self.number_splits_x = number_splits_x
        self.number_splits_y = number_splits_y
        self.path_guis = path_guis
        self.path_trained_models = path_trained_models
        self.path_dsls = path_dsls
        self.path_semantic = path_semantic
        self.path_abstraction = path_abstractions
        self.path_count_csv = path_count_csv
        self.path_semantic_all = path_semantic_all
        self.path_preproc_text = path_preproc_text
        self.path_preproc_text_small = path_preproc_text_small
        self.path_app_details = path_app_details
        self.path_ui_details = path_ui_details
        self.path_models = path_models
        self.path_ui_comp_models = path_ui_comp_models
        self.threshold = threshold
        self.tokenized = tokenized
        self.preprocesing_stemmed = preprocessing_stemmed
        self.preprocessing_stemmer = preprocessing_stemmer
        self.preprocesing_rm_stopwords = preprocessing_rm_stopwords
        self.filter_guis = filter_guis
        self.dir_name_prefix = dir_name_prefix
        if not text_segments_used:
            text_segments_used = ['text-all']
        self.text_segments_used = text_segments_used

    def get_desc(self) -> Text:
        desc = self.dir_name_prefix
        desc = desc + ('_stem_T'+self.preprocessing_stemmer) if self.preprocesing_stemmed else desc + '_stem_F'
        desc = desc + '_stpw_T' if self.preprocesing_rm_stopwords else desc + '_stpw_F'
        desc = desc + '_filter_T' if self.filter_guis else desc + '_filter_F'
        desc = desc + '_uts_' + '_'.join(self.text_segments_used)
        return desc

    def get_desc_dataset(self) -> Text:
        desc = self.dir_name_prefix
        desc = desc + ('_stem_T'+self.preprocessing_stemmer) if self.preprocesing_stemmed else desc + '_stem_F'
        desc = desc + '_stpw_T' if self.preprocesing_rm_stopwords else desc + '_stpw_F'
        desc = desc + '_filter_T' if self.filter_guis else desc + '_filter_F'
        return desc