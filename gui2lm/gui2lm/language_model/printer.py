from gui2lm.gui2lm.preprocessing.tokens import Tokens

intend = 36


def format_to_pretty_print_without_compare(sentence):
    sentence = sentence.replace("<", "")
    sentence = sentence.replace(">", "")
    sentence = sentence.strip()
    lines = sentence.split("|")
    formatted = "".ljust(intend * 3 + 3, "-")
    formatted += "\n"
    global_max_token_nr = 0
    for line in lines:
        line = line.strip()
        cubes = line.split(" ")
        translated_line = []
        max_token_nr = 3
        i = 0
        for cube in cubes:
            chars = cube.split("&")
            tokens = []
            try:
                for char in chars:
                    token = Tokens().char2token(char)
                    if i == 0:
                        start = "| "
                    else:
                        start = " "
                    tokens.append((start + token).ljust(intend) + "|")
            except KeyError:
                return "DID NOT WORK\n"
            if len(tokens) > max_token_nr:
                max_token_nr = len(tokens)
            i += 1
            translated_line.append(tokens)
        i = 0
        for translated_cube in translated_line:
            while len(translated_cube) < max_token_nr:
                if i == 0:
                    token = "|"
                else:
                    token = ""
                translated_cube.append(token.ljust(intend) + "|")
            i += 1
        for i in range(max_token_nr):
            for j in range(len(translated_line)):
                formatted += translated_line[j][i]
            formatted += "\n"
        formatted += "".ljust(intend * 3 + 3, "-")
        formatted += "\n"
    return formatted


# I am sorry this code is a mess
# Started of
def format_to_pretty_print_and_compare(predicted_sentence, actual_sentence):
    predicted_sentence = predicted_sentence.replace("<", "")
    predicted_sentence = predicted_sentence.replace(">", "")
    predicted_sentence = predicted_sentence.strip()
    actual_sentence = actual_sentence.replace("<", "")
    actual_sentence = actual_sentence.replace(">", "")
    actual_sentence = actual_sentence.strip()
    lines = predicted_sentence.split("|")
    actual_lines = actual_sentence.split("|")
    formatted = "".ljust(intend * 3 + 3, "-")
    formatted += "\n"
    global_max_token_nr = 0
    count_lines = 0
    for line in lines:
        actual_line = actual_lines[count_lines]
        line = line.strip()
        actual_line = actual_line.strip()
        cubes = line.split(" ")
        actual_cubes = actual_line.split(" ")
        translated_line = []
        max_token_nr = 3
        i = 0
        count_cubes = 0
        for cube in cubes:
            actual_cube = actual_cubes[count_cubes]
            chars = cube.split("&")
            actual_chars = actual_cube.split("&")
            max_chars_length = len(chars)
            if (len(chars) < len(actual_chars)):
                max_chars_length = len(actual_chars)
                for i in range(len(chars), max_chars_length):
                    chars.append(" ")
            else:
                max_chars_length = len(chars)
                for i in range(len(actual_chars), max_chars_length):
                    actual_chars.append(" ")
            tokens = []
            try:
                for i in range(0, max_chars_length):
                    # for char in chars:
                    if (actual_chars[i] == " "):
                        token = Tokens().char2token(chars[i])
                    else:
                        token = Tokens().char2token(chars[i]).ljust(int(intend / 2)) + "(" + Tokens().char2token(
                            actual_chars[i]) + ")"
                    if i == 0:
                        start = " "
                    else:
                        start = " "
                    tokens.append((start + token).ljust(intend) + "")
            except KeyError:
                return "DID NOT WORK\n"
            if max_chars_length > max_token_nr:
                max_token_nr = max_chars_length
            i += 1
            count_cubes += 1
            translated_line.append(tokens)
        count_lines += 1
        i = 0
        for translated_cube in translated_line:
            while len(translated_cube) < max_token_nr:
                if i == 0:
                    token = ""
                else:
                    token = ""
                translated_cube.append(token.ljust(intend) + "")
            i += 1
        for i in range(max_token_nr):
            for j in range(len(translated_line)):
                formatted += "| " + translated_line[j][i]
            formatted += "\n"
        formatted += "".ljust(intend * 3 + 3, "-")
        formatted += "|\n"
    return formatted


# def compare_to_generated_and_format(seed, generated, forced_teaching=False):
#     if (forced_teaching):
#         print("Original-Sentence:", seed)
#     seed_output = "Seed:\n"
#     seed_output += format_to_pretty_print(seed)
#     generated_output = "Generated:\n"
#     generated_output += format_to_pretty_print(generated)
#     return seed_output, generated_output


def format_generation_output(diversities_and_generations, seed, sentence, compare=True):
    string = f"""
Text generation with a seed and temperature sampling.  
Seed:     {seed}
Sentence: {sentence}
"""
    string += format_to_pretty_print_without_compare(sentence)
    for diversity, generation in diversities_and_generations.items():
        string += "\nGenerated with diversity " + str(diversity) + ": \"" + generation + "\"\n"
        if (compare):
            string += format_to_pretty_print_and_compare(generation, sentence)
        else:
            string += format_to_pretty_print_without_compare(generation)
    return string


def format_generation_output_forced_learning(diversities_and_generations, seed, compare=True):
    string = f"""
Text generation with a forced learning and temperature sampling. 
Sentence: {seed}
"""
    string += format_to_pretty_print_without_compare(seed)
    for diversity, generation in diversities_and_generations.items():
        string += "\nGenerated with diversity " + str(diversity) + ": \"" + generation + "\"\n"
        if (compare):
            string += format_to_pretty_print_and_compare(generation, seed)
        else:
            string += format_to_pretty_print_without_compare(generation)
    return string

# if __name__ == '__main__':
# print(format_to_pretty_print("< 1&2&4 2&4 4 | 0 1&2 0 | 0 2 0 | 0 1&2 0 >"))
# x, y = compare_to_generated_and_format("< 1&2&4 2&4 4 | 0 1&2 0 | 0 2 0 | 0 1&2 0 >",
#                                        "< 1&2&4 2&4 4 | 0 0 0 | 0 2 0 | 0 1&2 0 >")
# print(x)
# print(y)
# abstraction = []
# first_split = sentence.split("|")
# for line in first_split:
#     line.strip()
#     cubes = line.split(" ")
#     for cube in cubes:
#         chars = cube.split("&")
#         tokens = ""
#         for char in chars:
#             token = Tokens().char2token(char)
#             tokens += token + " "

# sentence.replace(" ", "\t")
# sentence.replace("|", "\n")
# translation = ""
# for char in sentence:
#     translation +=
# "< 1&2&4 2&4 4 4 0 4 5 1 4 2 0 0 2&4 7 | 2&| 7&1 3 4&&0 1 0 3 2&5&6 4&7&>"
