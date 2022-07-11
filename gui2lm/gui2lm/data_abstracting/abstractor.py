import json
import logging
import os
from typing import Text, List

from gui2lm.gui2lm import utils
from gui2lm.gui2lm.configuration.conf import Configuration
from gui2lm.gui2lm.data_abstracting.guiclasses.abstracted_gui_classes import AbstractedGui

from gui2lm.gui2lm.data_abstracting.exceptions.bounds_beyond_gui import BoundsBeyondGui
from gui2lm.gui2lm.data_abstracting.exceptions.bounds_in_minus import BoundsInMinus
from gui2lm.gui2lm.data_abstracting.exceptions.gui_size_errors import GuiSizeError
from gui2lm.gui2lm.data_abstracting.guiclasses.gui_classes import GuiElement, Gui
from gui2lm.gui2lm.data_abstracting.filter import Filter

logging.getLogger().setLevel(logging.INFO)


# Abstractor class to abstract semantic dataset into grid representations
class Abstractor:
    def __init__(self):
        self.filter_count_cat = 0
        self.filter_ads_count = 0
        self.xml_parse_errors = 0
        self.json_parse_errors = 0

    def abstract_semantic_gui_dataset(self, conf: Configuration, filter):
        dataset = conf.path_semantic_all
        # self.abstract_semantic_gui_json_rek(dataset, "1.json", conf, filter)
        for file_name in utils.iter_files_in_dir(dataset, ending='.json'):
            self.abstract_semantic_gui_json_rek(dataset, file_name, conf, filter)

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
            print(file_name)
            ui_json_semantic = json.load(file_1)
            all_leafs = []
            for children in ui_json_semantic['children']:
                self.getLeaf(children, all_leafs)
            try:
                gui = Gui(file_name, ui_json_semantic["bounds"], all_leafs)
                abstracted_gui = AbstractedGui(gui, conf, int(file_name.split('.')[0]), filter)
                try:
                    # Create target Directory
                    os.mkdir(conf.path_abstraction + "Y" + str(conf.number_splits_y) + "X" + str(
                        conf.number_splits_x))
                    print("Directory ", conf.path_abstraction + "Y" + str(conf.number_splits_y) + "X" + str(
                        conf.number_splits_x), " Created ")
                except FileExistsError:
                    print("Directory ", conf.path_abstraction + "Y" + str(conf.number_splits_y) + "X" + str(
                        conf.number_splits_x),
                          " already exists")
                with open(conf.path_abstraction + "Y" + str(conf.number_splits_y) + "X" + str(
                        conf.number_splits_x) + "/" + file_name, 'w') \
                        as write_f:
                    json.dump(abstracted_gui.to_dict(), write_f)

            except BoundsInMinus as e:
                # The bounds of the GUIs elements are in minus
                self.json_parse_errors += 1
                print("AN ERROR OCCURED: BOUNDS-IN-MINUS")
            except BoundsBeyondGui as e:
                # The bounds of the GUIs elements are beyond the GUI
                self.json_parse_errors += 1
                print("AN ERROR OCCURED: BOUNDS-BEYOND-GUI")
            except GuiSizeError as e:
                # The GUI does not have a defined width or height
                self.json_parse_errors += 1
                print("AN ERROR OCCURED: GUI-SIZE-ERROR")

    def getLeaf(self, json: Text, all_leafs: List[GuiElement]) -> None:
        childs_not_none = "children" in json
        childs_not_empty = False
        if childs_not_none:
            childs_not_empty = len(json["children"]) != 0
        if childs_not_empty:
            for children in json["children"]:
                self.getLeaf(children, all_leafs)
        else:
            leaf = GuiElement(json["bounds"], json["componentLabel"])
            all_leafs.append(leaf)
