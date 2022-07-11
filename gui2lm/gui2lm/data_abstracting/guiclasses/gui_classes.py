from enum import Enum
from typing import List, Text

from gui2lm.gui2lm.data_abstracting.exceptions.gui_size_errors import GuiSizeError


class GuiElement():

    def __init__(self, bounds: List[float], type: str):
        self.bounds = bounds
        self.type = type

    def print(self):
        print(self.type)
        print(self.bounds)


class Gui():
    def __init__(self, file_name: Text, bounds: List[float], elements: List[GuiElement]):
        self.filename = file_name
        self.bounds = bounds
        self.elements = self.normalizeBounds(elements)
        # self.elements = elements

    def print(self):
        print(self.filename)
        print(self.bounds)
        for element in self.elements:
            element.print()

    def normalizeBounds(self, elements) -> List[GuiElement]:

        width = self.bounds[2] - self.bounds[0]
        height = self.bounds[3] - self.bounds[1]

        if (width <= 0 or height <= 0):
            raise GuiSizeError()

        for element in elements:
            # Set top left corner of screen to x=0 and y=0
            element.bounds[0] -= self.bounds[0]
            element.bounds[1] -= self.bounds[1]
            element.bounds[2] -= self.bounds[0]
            element.bounds[3] -= self.bounds[1]

            element.bounds[0] /= width
            element.bounds[1] /= height
            element.bounds[2] /= width
            element.bounds[3] /= height

        # normalize screen bounds
        self.bounds[0] = 0
        self.bounds[1] = 0
        self.bounds[2] = 1
        self.bounds[3] = 1
        return elements


class LeafElementType(Enum):
    # Text=1
    # Image=2
    # Icon=3
    # TextButton=4
    # Input=5
    # WebView=6
    # BackgroundImage=7
    # RadioButton=8
    # PagerIndicator=9
    # Checkbox=10
    # Slider=11
    # Video=12

    Text = "Text"
    Image = "Image"
    Icon = "Icon"
    TextButton = "Text Button"
    Input = "Input"
    WebView = "Web View"
    BackgroundImage = "Background Image"
    RadioButton = "Radio Button"
    PagerIndicator = "Pager Indicator"
    Checkbox = "Checkbox"
    Slider = "Slider"
    Video = "Video"
