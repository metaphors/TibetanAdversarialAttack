"""
:type: OpenAttack.utils.WordVector
:Size: 175 MB
"""

from OpenAttack.utils import make_zip_downloader

NAME = "AttackAssist.TibetanWord2Vec"

DOWNLOAD = make_zip_downloader("")


def LOAD(path):
    import os
    import numpy as np
    from OpenAttack.attack_assist import WordEmbedding

    tibetan_list = []
    with open(os.path.join(path, "tibetan.txt"), "r", encoding="utf-8") as f:
        for line in f.readlines():
            tibetan_list.append(line.strip())

    word2id = {}
    id2vec = []
    with open(os.path.join(path, "cc.bo.300.vec"), "r", encoding="utf-8") as f:
        id = 0
        for index, line in enumerate(f.readlines()):
            if index == 0:
                continue
            tmp = line.strip().split(" ")
            word = tmp[0]
            for tibetan in tibetan_list:
                if word == tibetan:
                    embedding = np.array([float(x) for x in tmp[1:]])
                    if len(embedding) != 300:
                        break
                    word2id[word] = id
                    id += 1
                    id2vec.append(embedding)
                    break
        id2vec = np.stack(id2vec)
    return WordEmbedding(word2id, id2vec)
