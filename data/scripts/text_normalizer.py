from nemo_text_processing.text_normalization.normalize import Normalizer
from whisper_normalizer.english import EnglishNumberNormalizer, EnglishSpellingNormalizer
import re

import re
import unicodedata

import regex

# non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


# def remove_symbols_and_diacritics(s: str, keep=""):
#     """
#     Replace any other markers, symbols, and punctuations with a space,
#     and drop any diacritics (category 'Mn' and some manual mappings)
#     """
#     return "".join(
#         c
#         if c in keep
#         else ADDITIONAL_DIACRITICS[c]
#         if c in ADDITIONAL_DIACRITICS
#         else ""
#         if unicodedata.category(c) == "Mn"
#         else " "
#         if unicodedata.category(c)[0] in "MSP"
#         else c
#         for c in unicodedata.normalize("NFKD", s)
#     )

# def remove_symbols(s: str):
#     """
#     Replace any other markers, symbols, punctuations with a space, keeping diacritics
#     """
#     return "".join(
#         " " if unicodedata.category(c)[0] in "MSP" else c
#         for c in unicodedata.normalize("NFKC", s)
#     )


def remove_symbols_and_diacritics(s: str, keep=".,\"'?!"):
    """
    Replace any other markers, symbols, and punctuations with a space,
    and drop any diacritics (category 'Mn' and some manual mappings).
    Keep specified punctuation symbols.
    """
    return "".join(
        c
        if c in keep  # Keep punctuation if in the 'keep' set
        else ADDITIONAL_DIACRITICS[c]
        if c in ADDITIONAL_DIACRITICS
        else ""
        if unicodedata.category(c) == "Mn"
        else " "
        if unicodedata.category(c)[0] in "MSP" and c not in keep  # Replace symbols except those in 'keep'
        else c
        for c in unicodedata.normalize("NFKD", s)
    )
def remove_symbols(s: str, keep=".,\"'?!"):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics.
    Keep specified punctuation symbols.
    """
    return "".join(
        c if c in keep  # Keep punctuation if in the 'keep' set
        else " " if unicodedata.category(c)[0] in "MSP" and c not in keep  # Replace symbols except those in 'keep'
        else c
        for c in unicodedata.normalize("NFKC", s)
    )




class BasicTextNormalizer:
    def __init__(self, remove_diacritics: bool = False, split_letters: bool = False):
        self.clean = (
            remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        )
        self.split_letters = split_letters

    def __call__(self, s: str):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s).lower()

        if self.split_letters:
            s = " ".join(regex.findall(r"\X", s, regex.U))

        s = re.sub(
            r"\s+", " ", s
        )  # replace any successive whitespace characters with a space

        return s

#################
#    whisper    #
#################

class EnglishTextNormalizer:
    def __init__(self):
        self.replacers = {
            r"\bma'am\b": "madam",
            r"\bmr\b": "mister ",
            r"\bmrs\b": "missus ",
            r"\bst\b": "saint ",
            r"\bdr\b": "doctor ",
            r"\bprof\b": "professor ",
            r"\bcapt\b": "captain ",
            r"\bgov\b": "governor ",
            r"\bald\b": "alderman ",
            r"\bgen\b": "general ",
            r"\bsen\b": "senator ",
            r"\brep\b": "representative ",
            r"\bpres\b": "president ",
            r"\brev\b": "reverend ",
            r"\bhon\b": "honorable ",
            r"\basst\b": "assistant ",
            r"\bassoc\b": "associate ",
            r"\blt\b": "lieutenant ",
            r"\bcol\b": "colonel ",
            r"\bjr\b": "junior ",
            r"\bsr\b": "senior ",
            r"\besq\b": "esquire ",
        }
        self.standardize_numbers = EnglishNumberNormalizer()
        self.standardize_spellings = EnglishSpellingNormalizer()

    def __call__(self, s: str):
        s = s.lower()
        # s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        # s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = re.sub(r"\s+'", "'", s)  # when there's a space before an apostrophe

        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        s = re.sub(r"(\d),(\d)", r"\1\2", s)  # remove commas between digits
        s = remove_symbols_and_diacritics(s, keep=",.!?\"'")  # keep numeric symbols

        # s = self.standardize_numbers(s)
        # s = self.standardize_spellings(s)
        #
        # # now remove prefix/suffix symbols that are not preceded/followed by numbers
        # s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        # s = re.sub(r"([^0-9])%", r"\1 ", s)

        s = re.sub(r"\s+", " ", s)  # replace any successive whitespaces with a space

        # Ensure periods at the end of sentences are preserved
        s = re.sub(r"\s\.", ".", s)

        return s



class TextNormalizer:
    def __init__(self):
        self.nemo = Normalizer(input_case='cased', lang='en')
        self.whisper = EnglishTextNormalizer()

    def normalize(self, text):
        text = self.whisper(text)
        text = self.nemo.normalize(text, punct_post_process=True)
        return text