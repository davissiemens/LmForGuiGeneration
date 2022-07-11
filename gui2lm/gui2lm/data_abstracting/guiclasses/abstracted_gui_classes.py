import math
from collections import Counter
from typing import List, Dict

from gui2lm.gui2lm.configuration.conf import Configuration
from gui2lm.gui2lm.data_abstracting.exceptions.bounds_beyond_gui import BoundsBeyondGui
from gui2lm.gui2lm.data_abstracting.exceptions.bounds_in_minus import BoundsInMinus
from gui2lm.gui2lm.data_abstracting.guiclasses.gui_classes import Gui, GuiElement, LeafElementType


class Coordinate():
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class AbstractedGuiElement():
    def __init__(self, type: str):
        self.type = type

    def print(self):
        print(self.type)

    def to_enum(self) -> LeafElementType:
        type_without_whitespace = self.type.replace(" ", "")
        if (type_without_whitespace in [e.name for e in LeafElementType]):
            LeafElementType[type_without_whitespace]
            return LeafElementType[type_without_whitespace]
        return None


class MatrixElement():
    def __init__(self):
        self.elements = []

    def append(self, item: AbstractedGuiElement):
        self.elements.append(item)

    def print(self):
        if len(self.elements) == 0:
            print("NONE")
        else:
            for element in self.elements:
                element_label = element.to_enum()
                if (element_label is not None):
                    print(element_label.name)

    def to_json_string(self) -> Dict[int, int]:
        elements = []
        if len(self.elements) == 0:
            return ""

        for element in self.elements:
            element_label = element.to_enum()
            if (element_label is not None):
                elements.append(element_label.name)
        return dict(Counter(elements))
        # print(json_representation)


def calculate_center(element: GuiElement) -> Coordinate:
    center_x = element.bounds[0] + (element.bounds[2] - element.bounds[0]) / 2
    center_y = element.bounds[1] + (element.bounds[3] - element.bounds[1]) / 2
    coordinate = Coordinate(center_x, center_y)
    return coordinate


class AbstractedGui():

    def __init__(self, gui: Gui, conf: Configuration, ui_number, filter):
        self.filename = gui.filename
        self.numberOfLeafs = 0
        self.ui_number = ui_number
        self.number_splits_x = conf.number_splits_x
        self.number_splits_y = conf.number_splits_y
        self.elements = []
        self.matrix = self.abstract_elements(gui)
        self.filter = filter

    def print(self):
        print(self.filename)
        for i in range(0, self.number_splits_y):
            for j in range(0, self.number_splits_x):
                print("X:" + str(j) + " Y:" + str(i))
                self.matrix[j][i].print()

    def to_dict(self) -> Dict:
        dict_of_ui = {}
        for i in range(0, self.number_splits_y):
            dict_of_x_axis = {}
            for j in range(0, self.number_splits_x):
                dict_of_x_axis[j] = self.matrix[j][i].to_json_string()
                # print(str(i)+str(j)+str(self.matrix[j][i].to_json_string()))
            dict_of_ui[i] = dict_of_x_axis

        is_advertisement = self.check_if_advertisement()

        metadata = {"category": self.filter.get_category(self.ui_number), "number_of_elements": self.numberOfLeafs,
                    "is_advertisement": is_advertisement}
        dict_of_ui["metadata"] = metadata

        return dict_of_ui

    def check_if_advertisement(self) -> bool:
        if self.numberOfLeafs == 1 and LeafElementType.WebView in self.elements:
            return True
        elif self.numberOfLeafs == 2 and (LeafElementType.WebView in self.elements) and (
                LeafElementType.Icon in self.elements):
            return True
        else:
            return False

    def abstract_elements(self, gui: Gui) -> List[List[MatrixElement]]:
        # initiate matrix
        matrix = [[MatrixElement() for x in range(self.number_splits_y)] for y in range(self.number_splits_x)]

        for element in gui.elements:
            center = calculate_center(element)
            x_in_matrix = math.floor(center.x * self.number_splits_x)
            if x_in_matrix == self.number_splits_x:
                x_in_matrix -= 1
            if x_in_matrix > self.number_splits_x:
                raise BoundsBeyondGui()
            y_in_matrix = math.floor(center.y * self.number_splits_y)
            if y_in_matrix == self.number_splits_y:
                y_in_matrix -= 1
            if y_in_matrix > self.number_splits_y:
                raise BoundsBeyondGui()
            if x_in_matrix < 0 or y_in_matrix < 0:
                raise BoundsInMinus()
            abstracted_gui_element = AbstractedGuiElement(element.type)
            matrix[x_in_matrix][y_in_matrix].append(abstracted_gui_element)
            # element_type = element.type.replace(" ", "")
            element2enum = abstracted_gui_element.to_enum()
            if element2enum is not None:
                self.elements.append(element2enum)
            self.numberOfLeafs += 1
        return matrix

    def calculate_x_split(self, center: Coordinate):
        self.number_splits_x
        pass
