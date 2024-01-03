"""
:type: OpenAttack.utils.BertClassifier
:Size: 446 MB
:Package Requirements:
    * transformers
    * pytorch

Fine-tune Tibetan-BERT on TNCC-document.
Tibetan-BERT: Tibetan BERT.
TNCC: Tibetan News Classification Corpus.
"""

from OpenAttack.utils import make_zip_downloader

NAME = "Victim.BERT.TIBETAN-BERT_TNCC-DOCUMENT"

DOWNLOAD = make_zip_downloader("")


def LOAD(path):
    from transformers import BertForSequenceClassification, BertTokenizer
    model = BertForSequenceClassification.from_pretrained(path, num_labels=12)
    tokenizer = BertTokenizer.from_pretrained(path)

    from OpenAttack.victim.classifiers import TransformersClassifier
    return TransformersClassifier(model, tokenizer, model.bert.embeddings.word_embeddings, max_length=512, lang="tibetan")
