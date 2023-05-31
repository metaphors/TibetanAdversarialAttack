from .embed_based import EmbedBasedSubstitute
from ....data_manager import DataManager
from ....tags import TAG_Tibetan
import torch


class TibetanWord2VecSubstitute(EmbedBasedSubstitute):
    TAGS = {TAG_Tibetan}

    # def __init__(self, cosine=True, k=50, threshold=0.5, device=None):
        # 60 degree
    def __init__(self, cosine=True, k=50, threshold=0.2929, device=None):
        # 45 degree
    # def __init__(self, cosine=True, k=50, threshold=0.1340, device=None):
        # 30 degree
        """
        Tibetan word substitute based on word2vec.

        Args:
            cosine: If `true` then the cosine distance is used, otherwise the Euclidian distance is used.
            k: Top-k results to return. If k is `None`, all results will be returned. Default: 50
            threshold: Distance threshold. Default: 0.5
            device: A pytocrh device for computing distances. Default: "cpu"

        :Data Requirements: :py:data:`.AttackAssist.TibetanWord2Vec`
        :Language: tibetan

        """
        word_vec = DataManager.load("AttackAssist.TibetanWord2Vec")

        super().__init__(
            word2id=word_vec.word2id,
            embedding=torch.from_numpy(word_vec.embedding),
            cosine=cosine,
            k=k,
            threshold=threshold,
            device=device
        )
