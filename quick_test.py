import numpy as np
import itertools as it
from collections import Counter
from vocab import Vocab
from model import LM
from utils import batch_iter, smi_tokenizer
import torch
import time
import math
import sys

# constants
model_save_path = './model.bin'
input_str = sys.argv[1]
print('Input is ' + input_str)
chars = []
for c in input_str:
    chars.append(c)

# load model
print("load model from {}".format(model_save_path), file=sys.stderr)
model = LM.load(model_save_path)

# params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
# model.load_state_dict(params['state_dict'])

model.eval()
hypotheses = model.beam_search(chars)

for hypothesis in hypotheses:
    print('extension {} has value {}'.format(''.join(hypothesis[0]).replace('</s>',''), hypothesis[1]))