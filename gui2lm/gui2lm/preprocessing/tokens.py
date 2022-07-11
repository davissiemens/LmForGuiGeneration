# Class defining tokens and mapping them to one unique char


class Tokens():
    padding = "_"
    no_element = "Empty"
    line_break = "|"
    start_token = "START"
    end_token = "END"
    concatenation = "&"
    split = " "
    token2char = {
        padding: "_",
        no_element: "0",
        line_break: "|",
        start_token: "<",
        end_token: ">",
        concatenation: "&",
        split: " ",

        "Text": "1",
        "Image": "2",
        "Icon": "3",
        "TextButton": "4",
        "Input": "5",
        "WebView": "6",
        "BackgroundImage": "7",
        "RadioButton": "8",
        "PagerIndicator": "9",
        "Checkbox": "A",
        "Slider": "B",
        "Video": "C"
    }

    def int2char(self):
        char2int = Tokens().char2int()
        int2char = {v: k for k, v in char2int.items()}
        return int2char

    def char2token(self, char):
        inv_map = {v: k for k, v in Tokens().token2char.items()}
        return inv_map[char]

    def char2int(self):
        char2int = {}
        i = 0
        for c in Tokens().token2char.values():
            char2int[c] = i
            i += 1
        return char2int
