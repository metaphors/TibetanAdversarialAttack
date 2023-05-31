from .chinese import CHINESE_FILTER_WORDS
from .english import ENGLISH_FILTER_WORDS
from .tibetan import TIBETAN_FILTER_WORDS


def get_default_filter_words(lang):
    from ...tags import TAG_Chinese, TAG_English, TAG_Tibetan
    if lang == TAG_Chinese:
        return CHINESE_FILTER_WORDS
    if lang == TAG_English:
        return ENGLISH_FILTER_WORDS
    if lang == TAG_Tibetan:
        return TIBETAN_FILTER_WORDS
    return ENGLISH_FILTER_WORDS