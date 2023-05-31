"""
:type: OpenAttack.utils.XLMRobertaClassifier
:Size: 763 MB
:Package Requirements:
    * transformers
    * pytorch

Fine-tune CINO-base-v2 on TUSA.
CINO: Pre-trained Language Models for Chinese Minority Languages.
TUSA: Tibet University Sentiment Analysis.
"""

from OpenAttack.utils import make_zip_downloader

NAME = "Victim.XLMROBERTA.CINO-BASE-V2_TUSA"

DOWNLOAD = make_zip_downloader("")


def LOAD(path):
    from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
    model = XLMRobertaForSequenceClassification.from_pretrained(path, num_labels=2)
    tokenizer = XLMRobertaTokenizer.from_pretrained(path)

    from OpenAttack.victim.classifiers import TransformersClassifier
    return TransformersClassifier(model, tokenizer, model.roberta.embeddings.word_embeddings, max_length=512, lang="tibetan")
