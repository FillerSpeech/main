import re
import unicodedata

from g2p_en import G2p
from g2p_en.expand import normalize_numbers
from nltk import pos_tag
from nltk.tokenize import TweetTokenizer

from speech_synthesis.data_gen.tts.txt_processors.base_text_processor import (
    BaseTxtProcessor,
    register_txt_processors,
)
from utils.text.text_encoder import PUNCS


class EnG2p(G2p):
    word_tokenize = TweetTokenizer().tokenize

    def __call__(self, text):
        # preprocessing
        words = EnG2p.word_tokenize(text)
        tokens = pos_tag(words)  # tuples of (word, tag)

        # steps
        prons = []
        for word, pos in tokens:
            if re.search("[a-z]", word) is None:
                pron = [word]

            elif word in self.homograph2features:  # Check homograph
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                else:
                    pron = pron2
            elif word in self.cmu:  # lookup CMU dict
                pron = self.cmu[word][0]
            else:  # predict for oov
                pron = self.predict(word)

            prons.extend(pron)
            prons.extend([" "])

        return prons[:-1]


@register_txt_processors("en_filler")
class TxtProcessor(BaseTxtProcessor):
    g2p = EnG2p()

    @staticmethod
    def preprocess_text(text):
        text = normalize_numbers(text)
        text = "".join(
            char
            for char in unicodedata.normalize("NFD", text)
            if unicodedata.category(char) != "Mn"
        )  # Strip accents
        text = text.lower()
        text = re.sub("[\"()]+", "", text)
        text = re.sub(r"(?<!\w)'|'(?!\w)", "", text)
        text = re.sub("[-]+", " ", text)
        text = re.sub(f"[^ a-z{PUNCS}<>']", "", text)
        text = re.sub(f" ?([{PUNCS}]) ?", r"\1", text)  # !! -> !
        text = re.sub(f"([{PUNCS}])+", r"\1", text)  # !! -> !
        text = text.replace("i.e.", "that is")
        text = text.replace("etc.", "etc")
        text = re.sub(f"([{PUNCS}])", r" \1 ", text)
        text = re.sub(rf"\s+", r" ", text)
        return text

    @classmethod
    def process(cls, txt, preprocess_args):
        txt = cls.preprocess_text(txt).strip()
        phs = cls.g2p(txt)
        txt_struct = [[w, []] for w in txt.split(" ")]
        i_word = 0
        for p in phs:
            if p == " ":
                i_word += 1
            else:
                txt_struct[i_word][1].append(p)
        txt_struct = cls.postprocess(txt_struct, preprocess_args)
        return txt_struct, txt

    @classmethod
    def add_bdr(cls, txt_struct):
        txt_struct_ = []
        for i, ts in enumerate(txt_struct):
            txt_struct_.append(ts)
            if (
                i != len(txt_struct) - 1
                and not cls.is_sil_phoneme(txt_struct[i][0])
                and not cls.is_sil_phoneme(txt_struct[i + 1][0])
            ):
                txt_struct_.append(["|", ["|"]])
        return txt_struct_

    @classmethod
    def is_sil_phoneme(cls, p):
        if p[0] == '<':
            return False
        else:
            return p == '' or not p[0].isalpha()

    @classmethod
    def postprocess(cls, txt_struct, preprocess_args):
        # remove sil phoneme in head and tail
        while len(txt_struct) > 0 and cls.is_sil_phoneme(txt_struct[0][0]):
            txt_struct = txt_struct[1:]
        while len(txt_struct) > 0 and cls.is_sil_phoneme(txt_struct[-1][0]):
            txt_struct = txt_struct[:-1]
        if preprocess_args["with_phsep"]:
            txt_struct = cls.add_bdr(txt_struct)
        if preprocess_args["add_eos_bos"]:
            txt_struct = [["<BOS>", ["<BOS>"]]] + txt_struct + [["<EOS>", ["<EOS>"]]]
        return txt_struct
