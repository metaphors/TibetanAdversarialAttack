from .base import Tokenizer
from .jieba_tokenizer import JiebaTokenizer
from .punct_tokenizer import PunctTokenizer
from .tibetan_syllable_tokenizer import TibetanSyllableTokenizer
from .TibetSegEYE.Tibet_tokenlize.maintest import TibetanWordTokenizer
from .transformers_tokenizer import TransformersTokenizer

def get_default_tokenizer(lang):
    from ...tags import TAG_English, TAG_Chinese, TAG_Tibetan
    if lang == TAG_English:
        return PunctTokenizer()
    if lang == TAG_Chinese:
        return JiebaTokenizer()
    if lang == TAG_Tibetan:
        return TibetanSyllableTokenizer()
    return PunctTokenizer()