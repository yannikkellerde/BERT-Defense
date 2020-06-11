# Code based on https://stackoverflow.com/questions/54978443/predicting-missing-words-in-a-sentence-natural-language-processing-model

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
from tqdm import trange,tqdm
from util import load_dictionary,each_char_in
from letter_stuff import singular_punctuations,all_chars

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "Opinionated"
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)