from .base import Tokenizer
from ...tags import TAG_Tibetan


class TibetanSyllableTokenizer(Tokenizer):
    """
    Tibetan tokenizer based on syllable level

    :Language: tibetan
    """

    TAGS = {TAG_Tibetan}

    def do_tokenize(self, x, pos_tagging):
        ret = []
        for syllable in x.split('་'):
            if pos_tagging:
                ret.append((syllable, "other"))
            else:
                ret.append(syllable)
        return ret

    def do_detokenize(self, x):
        return '་'.join(x)
