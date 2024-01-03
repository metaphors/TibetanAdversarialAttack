"""
:type: OpenAttack.utils.BertClassifier
:Size: 446 MB
:Package Requirements:
    * transformers
    * pytorch

Fine-tune Tibetan-BERT on TUSA.
Tibetan-BERT: Tibetan BERT.
TUSA: Tibet University Sentiment Analysis.
"""

from OpenAttack.utils import make_zip_downloader

NAME = "Victim.BERT.TIBETAN-BERT_TUSA"

DOWNLOAD = make_zip_downloader("")


def LOAD(path):
    from transformers import BertForSequenceClassification, BertTokenizer
    model = BertForSequenceClassification.from_pretrained(path, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained(path)

    from OpenAttack.victim.classifiers import TransformersClassifier
    return TransformersClassifier(model, tokenizer, model.bert.embeddings.word_embeddings, max_length=512, lang="tibetan")
