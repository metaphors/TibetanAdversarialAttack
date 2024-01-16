
from .utils import model_use
from .models import config, No_encoder_model
from importlib import import_module
import torch
from .appedix_restore import dict_confirm
import numpy as np
#TibetSegEye
#单次输入


config=config()

model=No_encoder_model(config)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(3)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark =False

#use
model.load_state_dict(torch.load(config.save_result))
id2label = config.i2b
model.to(config.device)
# while 1:
#     seq=input()
#     dictory=dict_confirm(config)
#     seq_out=model_use(model,config,seq,dictory,id2label)
#     print(seq_out)
# pass
from ...base import Tokenizer
from .....tags import TAG_Tibetan


class TibetanWordTokenizer(Tokenizer):
    """
    Tibetan tokenizer based on word level

    :Language: tibetan
    """

    TAGS = {TAG_Tibetan}

    def do_tokenize(self, x, pos_tagging):
        ret = []
        dictory = dict_confirm(config)
        x_out=model_use(model,config,x,dictory,id2label)[1:]
        for word in x_out.split('\\'):
            if pos_tagging:
                ret.append((word, "other"))
            else:
                ret.append(word)
        return ret

    def do_detokenize(self, x):
        return ''.join(x)
