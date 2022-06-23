import json
import gui2lm.gui2lm.utils as utils
from json2xml import json2xml
import logging
from typing import Text, List, Dict
from parsel import Selector
from collections import Counter
import csv

from gui2lm.gui2lm.data_abstracting.configuration.conf import Configuration
from gui2lm.gui2lm.data_abstracting.filter import Filter

logging.getLogger().setLevel(logging.INFO)


class LabelCounter():
    def __init__(self):
        self.json_read_error = 0

    def write_label_count_to_file_test(self, conf: Configuration):
        label_count = self.count_labels_in_dataset_test(conf)
        with open(conf.path_count_csv+'count_labels_test.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in label_count.items():
                writer.writerow([key, value])

    def count_labels_in_dataset_test(self, conf: Configuration) -> Dict[Text, int]:
        # unique_labels=Set[Text]
        label_count = {}
        dataset = conf.path_semantic
        for file_name in utils.iter_files_in_dir(dataset, ending='.json'):
            data = self.label_extraction_from_file(dataset, file_name, filter)
            if data is not None:
                file_labels = dict(Counter(data))
                # print(file_labels)
                label_count = utils.mergeDict(label_count, file_labels)
                # print(label_count)
        return label_count

    def write_label_count_to_file(self, conf: Configuration):
        label_count = self.count_labels_in_dataset(conf)
        with open(conf.path_count_csv+'count_labels.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in label_count.items():
                writer.writerow([key, value])

    def count_labels_in_dataset(self, conf: Configuration) -> Dict[Text,int]:
        # unique_labels=Set[Text]
        label_count = {}
        dataset= conf.path_semantic_all
        for file_name in utils.iter_files_in_dir(dataset, ending='.json'):
            data = self.label_extraction_from_file(dataset, file_name, filter)
            if data is not None:
                file_labels = dict(Counter(data))
                # print(file_labels)
                label_count = utils.mergeDict(label_count, file_labels)
                # print(label_count)
        return label_count



    def label_extraction_from_file(self, file_path_semantic: Text,
                                  file_name: Text,
                                  filter: Filter) -> List[Text]:
        with open(file_path_semantic + file_name, 'r', encoding='utf8') as file_1:
            ui_json_semantic = json.load(file_1)
            try:
                ui_xml_semantic = json2xml.Json2xml(ui_json_semantic).to_xml()
                selector_semantic = Selector(text=ui_xml_semantic)
                results = selector_semantic.xpath('//componentlabel/text()').getall()
                return results
            except:
                self.json_read_error += 1
                print("AN ERROR OCCURED")
                return None


    def write_leaf_count_to_file(self, conf: Configuration):
        label_count = self.count_leafs_in_dataset(conf)
        with open(conf.path_count_csv+'count_leafs.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in label_count.items():
                writer.writerow([key, value])

    def count_leafs_in_dataset(self, conf: Configuration) -> Dict[Text,int]:
        # unique_labels=Set[Text]
        label_count = {}
        dataset= conf.path_semantic_all
        for file_name in utils.iter_files_in_dir(dataset, ending='.json'):
            print(file_name)
            data = self.leaf_extraction_from_file(dataset, file_name, filter)
            if data is not None:
                file_labels = dict(Counter(data))
                # print(file_labels)
                label_count = utils.mergeDict(label_count, file_labels)
                # print(label_count)
        return label_count



    def leaf_extraction_from_file(self, file_path_semantic: Text,
                                  file_name: Text,
                                  filter: Filter) -> List[Text]:
        with open(file_path_semantic + file_name, 'r', encoding='utf8') as file_1:
            # ui_json_semantic = json.load(file_1)
            try:
                ui_json_semantic = json.load(file_1)
                all_leafs = []
                for children in ui_json_semantic['children']:
                    self.getLeafCount(children, all_leafs)
                return all_leafs
            except:
                self.json_read_error += 1
                print("AN ERROR OCCURED")
                return None

    def abstract_semantic_gui_json_rek(self, file_path_semantic: Text,
                                       file_name: Text,
                                       conf: Configuration,
                                       filter: Filter) -> None:
        if conf.filter_guis:  # -> Default = False
            filter_cat = filter.filter_categories(file_name)
            if filter_cat:
                self.filter_count_cat += 1
                return None
        with open(file_path_semantic + file_name, 'r', encoding='utf8') as file_1:
            # try:
            ui_json_semantic = json.load(file_1)
            all_leafs = []
            for children in ui_json_semantic['children']:
                self.getLeaf(children, all_leafs)
            gui = Gui(file_name, ui_json_semantic["bounds"], all_leafs)
            abstracted_gui = AbstractedGui(gui,conf)
            gui.formatted()
            abstracted_gui.formatted()

    def getLeafCount(self, json: Text, all_leafs: List[Text]) -> None:
        childs_not_none = "children" in json
        childs_not_empty = False
        if childs_not_none:
            childs_not_empty = len(json["children"]) != 0
        if childs_not_empty:
            for children in json["children"]:
                self.getLeafCount(children, all_leafs)
        else:
            leaf = json["componentLabel"]
            all_leafs.append(leaf)