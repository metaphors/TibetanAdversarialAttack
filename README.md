# Tibetan Adversarial Attack

## Introduction

This repo is the attacker part in the paper below.

***[Multi-Granularity Tibetan Textual Adversarial Attack Method Based on Masked Language Model (Cao et al., WWW 2024 Workshop - SocialNLP)](https://dl.acm.org/doi/10.1145/3589335.3652503)***

⬆️ commit id: HEAD

***[Pay Attention to the Robustness of Chinese Minority Language Models! Syllable-level Textual Adversarial Attack on Tibetan Script (Cao et al., ACL 2023 Workshop - TrustNLP)](https://aclanthology.org/2023.trustnlp-1.4)***

⬆️ commit id: a17c605d44d53b222b0127f77643519ae33aefd9

We developed a simple Tibetan syllable-level adversarial attack method based on [OpenAttack](https://github.com/thunlp/OpenAttack) ([OpenAttack: An Open-source Textual Adversarial Attack Toolkit (Zeng et al., ACL 2021)](https://aclanthology.org/2021.acl-demo.43.pdf)).

⬆️ commit id: 4df712e0a5aebc03daa9b1ef353da4b7ea0a1b23

## Usage Method

1. You need to put [the fine-tuned LMs](https://github.com/metaphors/TibetanPLMsFineTuning) into the dirs (data/Victim.XLMROBERTA.CINO-SMALL-V2_TNCC-TITLE, data/Victim.XLMROBERTA.CINO-SMALL-V2_TUSA, data/Victim.XLMROBERTA.CINO-BASE-V2_TNCC-DOCUMENT, data/Victim.XLMROBERTA.CINO-BASE-V2_TNCC-TITLE, data/Victim.XLMROBERTA.CINO-BASE-V2_TUSA, data/Victim.XLMROBERTA.CINO-LARGE-V2_TNCC-DOCUMENT, data/Victim.XLMROBERTA.CINO-LARGE-V2_TNCC-TITLE, data/Victim.XLMROBERTA.CINO-LARGE-V2_TUSA, data/Victim.XLMROBERTA.TIBETAN-BERT_TNCC-TITLE, data/Victim.XLMROBERTA.TIBETAN-BERT_TUSA, etc.).
2. You need to download and unzip [the Tibetan word vectors](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bo.300.vec.gz) ([Learning Word Vectors for 157 Languages (Grave et al., LREC 2018)](https://aclanthology.org/L18-1550.pdf)) into the dir (data/AttackAssist.TibetanWord2Vec).
3. You need to put the pre-trained LM: [Tibetan-BERT](https://huggingface.co/UTibetNLP/tibetan_bert) ([Research and Application of Tibetan Pre-training Language Model Based on BERT (Zhang et al., ICCIR 2022)](https://dl.acm.org/doi/10.1145/3548608.3559255)), [TiBERT](http://tibert.cmli-nlp.com) ([TiBERT: Tibetan Pre-trained Language Model (Liu et al., SMC 2022)](https://ieeexplore.ieee.org/document/9945074)), etc. into the dirs (data/AttackAssist.Tibetan_BERT, data/AttackAssist.TiBERT, etc.).
4. You need to put the trained model: segbase.cpkt (link: [https://pan.baidu.com/s/1j_60cDWVlfryikaP-1Nvbw](https://pan.baidu.com/s/1j_60cDWVlfryikaP-1Nvbw) password: 19pe) of TibetSegEYE ([https://github.com/yjspho/TibetSegEYE](https://github.com/yjspho/TibetSegEYE)) into the dir (data/AttackAssist.TibetSegEYE).
5. You need to follow [the OpenAttack README](https://github.com/thunlp/OpenAttack) ([OpenAttack: An Open-source Textual Adversarial Attack Toolkit (Zeng et al., ACL 2021)](https://aclanthology.org/2021.acl-demo.43.pdf)) to install the development environment. 
6. You can run the attack scripts in the dir (demo_tibetan).

## Citation

If you think our work useful, please kindly cite our paper.

```
@inproceedings{10.1145/3589335.3652503,
    author = {Cao, Xi and Qun, Nuo and Gesang, Quzong and Zhu, Yulei and Nyima, Trashi},
    title = {Multi-Granularity Tibetan Textual Adversarial Attack Method Based on Masked Language Model},
    year = {2024},
    isbn = {9798400701726},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3589335.3652503},
    doi = {10.1145/3589335.3652503},
    booktitle = {Companion Proceedings of the ACM on Web Conference 2024},
    pages = {1672–1680},
    numpages = {9},
    keywords = {language model, robustness, textual adversarial attack, tibetan},
    location = {Singapore, Singapore},
    series = {WWW '24}
}
```

```
@inproceedings{cao-etal-2023-pay-attention,
    title = "Pay Attention to the Robustness of {C}hinese Minority Language Models! Syllable-level Textual Adversarial Attack on {T}ibetan Script",
    author = "Cao, Xi  and
      Dawa, Dolma  and
      Qun, Nuo  and
      Nyima, Trashi",
    booktitle = "Proceedings of the 3rd Workshop on Trustworthy Natural Language Processing (TrustNLP 2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.trustnlp-1.4",
    pages = "35--46"
}
```