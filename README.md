# Tibetan Adversarial Attack

## Introduction

This repo is the attacker part in the paper below.

***[Pay Attention to the Robustness of Chinese Minority Language Models! Syllable-level Textual Adversarial Attack on Tibetan Script](https://trustnlpworkshop.github.io/papers/6.pdf) (Cao et al., ACL 2023 Workshop - TrustNLP)***

We developed a simple Tibetan syllable-level adversarial attack method based on [OpenAttack](https://github.com/thunlp/OpenAttack) ([OpenAttack: An Open-source Textual Adversarial Attack Toolkit (Zeng et al., ACL 2021)](https://aclanthology.org/2021.acl-demo.43.pdf)).

## Usage Method

1. You need to put [the fine-tuned LMs](https://github.com/metaphors/TibetanPLMsFineTuning) into the dirs (data/Victim.XLMROBERTA.CINO-BASE-V2_TNCC-DOCUMENT, data/Victim.XLMROBERTA.CINO-BASE-V2_TNCC-TITLE, data/Victim.XLMROBERTA.CINO-BASE-V2_TUSA, data/Victim.XLMROBERTA.CINO-LARGE-V2_TNCC-DOCUMENT, data/Victim.XLMROBERTA.CINO-LARGE-V2_TNCC-TITLE, data/Victim.XLMROBERTA.CINO-LARGE-V2_TUSA).
2. You need to download and unzip [the Tibetan word vectors](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bo.300.vec.gz) ([Learning Word Vectors for 157 Languages (Grave et al., LREC 2018)](https://aclanthology.org/L18-1550.pdf)) into the dir (data/AttackAssist.TibetanWord2Vec).
3. You need to follow [the OpenAttack README](https://github.com/thunlp/OpenAttack) ([OpenAttack: An Open-source Textual Adversarial Attack Toolkit (Zeng et al., ACL 2021)](https://aclanthology.org/2021.acl-demo.43.pdf)) to install the development environment. 
4. You can run the attack scripts in the dir (demo_tibetan).