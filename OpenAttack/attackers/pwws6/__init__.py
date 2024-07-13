from typing import List, Optional
import numpy as np

from ..classification import ClassificationAttacker, Classifier, ClassifierGoal
from ...text_process.tokenizer import Tokenizer, get_default_tokenizer
# from ...attack_assist.substitute.word import WordSubstitute, get_default_substitute
from ...utils import get_language, check_language, language_by_name
# from ...exceptions import WordNotInDictionaryException
from ...tags import Tag
from ...attack_assist.filter_words import get_default_filter_words

# import torch
# from transformers import BertTokenizer, BertForMaskedLM, XLMRobertaTokenizer, XLMRobertaForMaskedLM

import json

class PWWSAttacker6(ClassificationAttacker):
    @property
    def TAGS(self):
        return { self.__lang_tag,  Tag("get_pred", "victim"), Tag("get_prob", "victim") }

    def __init__(self,
            tokenizer : Optional[Tokenizer] = None,
            # substitute : Optional[WordSubstitute] = None,
            token_unk : str = "<UNK>",
            filter_words : List[str] = None,
            lang = None
        ):
        """
        Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency. Shuhuai Ren, Yihe Deng, Kun He, Wanxiang Che. ACL 2019.
        `[pdf] <https://www.aclweb.org/anthology/P19-1103.pdf>`__
        `[code] <https://github.com/JHL-HUST/PWWS/>`__

        Args:
            tokenizer: A tokenizer that will be used during the attack procedure. Must be an instance of :py:class:`.Tokenizer`
            # substitute: A substitute that will be used during the attack procedure. Must be an instance of :py:class:`.WordSubstitute`
            token_unk: The token id or the token name for out-of-vocabulary words in victim model. **Default:** ``"<UNK>"``
            lang: The language used in attacker. If is `None` then `attacker` will intelligently select the language based on other parameters.
            filter_words: A list of words that will be preserved in the attack procesudre.

        :Classifier Capacity:
            * get_pred
            * get_prob

        
        """
       
        lst = []
        if tokenizer is not None:
            lst.append(tokenizer)
        # if substitute is not None:
        #     lst.append(substitute)
        if len(lst) > 0:
            self.__lang_tag = get_language(lst)
        else:
            self.__lang_tag = language_by_name(lang)
            if self.__lang_tag is None:
                raise ValueError("Unknown language `%s`" % lang)
        
        # if substitute is None:
        #     substitute = get_default_substitute(self.__lang_tag)
        # self.substitute = substitute

        if tokenizer is None:
            tokenizer = get_default_tokenizer(self.__lang_tag)
        self.tokenizer = tokenizer

        # check_language([self.tokenizer, self.substitute], self.__lang_tag)
        check_language([self.tokenizer], self.__lang_tag)

        self.token_unk = token_unk
        if filter_words is None:
            filter_words = get_default_filter_words(self.__lang_tag)
        self.filter_words = set(filter_words)

        self.basic_extension = []
        with open('data/AttackAssist.TibetanExtensionBasicTable/extension_basic.txt', 'r', encoding='utf-8') as file:
            for line in file:
                (extension, basic) = line.strip().split('\t')
                basic_chr_list = [basic[i:i + 4] for i in range(0, len(basic), 4)]
                basic_str = ''
                for basic_chr in basic_chr_list:
                    basic_str += chr(int(basic_chr, 16))
                extension_str = chr(int(extension, 16))
                self.basic_extension.append((basic_str, extension_str))
        
    def attack(self, victim: Classifier, sentence : str, goal : ClassifierGoal):
        x_orig = sentence.lower()


        x_orig = self.tokenizer.tokenize(x_orig)
        poss =  list(map(lambda x: x[1], x_orig)) 
        x_orig =  list(map(lambda x: x[0], x_orig)) 

        S = self.get_saliency(victim, x_orig, goal) # (len(sent), )
        S_softmax = np.exp(S - S.max())
        S_softmax = S_softmax / S_softmax.sum()

        w_star = [ self.get_wstar(victim, x_orig, i, poss[i], goal) for i in range(len(x_orig)) ]  # (len(sent), )
        H = [ (idx, w_star[idx][0], S_softmax[idx] * w_star[idx][1]) for idx in range(len(x_orig)) ]

        H = sorted(H, key=lambda x:-x[2])
        ret_sent = x_orig.copy()
        for i in range(len(H)):
            idx, wd, _ = H[i]
            if ret_sent[idx] in self.filter_words:
                continue
            ret_sent[idx] = wd
            
            curr_sent = self.tokenizer.detokenize(ret_sent)
            pred = victim.get_pred([curr_sent])[0]
            if goal.check(curr_sent, pred):
                return curr_sent
        return None


    
    def get_saliency(self, clsf, sent, goal : ClassifierGoal):
        x_hat_raw = []
        for i in range(len(sent)):
            left = sent[:i]
            right = sent[i + 1:]
            x_i_hat = left + [self.token_unk] + right
            x_hat_raw.append(self.tokenizer.detokenize(x_i_hat))
        x_hat_raw.append(self.tokenizer.detokenize(sent))
        res = clsf.get_prob(x_hat_raw)[:, goal.target]
        if not goal.targeted:
            res = res[-1] - res[:-1]
        else:
            res = res[:-1] - res[-1]
        return res

    def get_wstar(self, clsf, sent, idx, pos, goal : ClassifierGoal):
        word = sent[idx]
        # try:
        #     rep_words = list(map(lambda x:x[0], self.substitute(word, pos)))
        # except WordNotInDictionaryException:
        #     rep_words = []
        # rep_words = list(filter(lambda x: x != word, rep_words))
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # # Tibetan_BERT is better and simpler.
        # # mlm_path = "data/AttackAssist.TiBERT"
        # mlm_path = "data/AttackAssist.Tibetan_BERT"
        # mlm_tokenizer = BertTokenizer.from_pretrained(mlm_path)
        # mlm_model = BertForMaskedLM.from_pretrained(mlm_path).to(device)
        # # mlm_path = "data/AttackAssist.CINO-small-v2"
        # # mlm_path = "data/AttackAssist.CINO-base-v2"
        # # mlm_path = "data/AttackAssist.CINO-large-v2"
        # # mlm_tokenizer = XLMRobertaTokenizer.from_pretrained(mlm_path)
        # # mlm_model = XLMRobertaForMaskedLM.from_pretrained(mlm_path).to(device)
        # max_length = 512
        # masked_tokens = sent[: idx] + ['[MASK]'] + sent[idx + 1: max_length]
        # inputs = mlm_tokenizer(self.tokenizer.detokenize(masked_tokens), return_tensors="pt").to(device)
        # with torch.no_grad():
        #     logits = mlm_model(**inputs).logits
        # top_k = 50
        # predicted_indices = torch.topk(logits[0, idx], top_k)[1]
        # predicted_tokens = mlm_tokenizer.convert_ids_to_tokens(predicted_indices)
        # rep_words = []
        # for token in predicted_tokens:
        #     if token == word:
        #         continue
        #     if token == "[UNK]":
        #         continue
        #     if token == "་":
        #         continue
        #     if token == "༼":
        #         continue
        #     if token == "༽":
        #         continue
        #     if "##" in token:
        #         continue
        #     rep_words.append(token)
        rep_words = []
        for item in self.basic_extension:
            if item[0] in word:
                rep_words.append(word.replace(item[0], item[1]))
        if len(rep_words) == 0:
            return ( word, 0 )
        sents = []
        for rw in rep_words:
            new_sent = sent[:idx] + [rw] + sent[idx + 1:]
            sents.append(self.tokenizer.detokenize(new_sent))
        sents.append(self.tokenizer.detokenize(sent))
        res = clsf.get_prob(sents)[:, goal.target]
        prob_orig = res[-1]
        res = res[:-1]
        if goal.targeted:
            return (rep_words[ res.argmax() ],  res.max() - prob_orig )
        else:
            return (rep_words[ res.argmin() ],  prob_orig - res.min() )
    








