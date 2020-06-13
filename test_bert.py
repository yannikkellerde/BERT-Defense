# Code based on https://stackoverflow.com/questions/54978443/predicting-missing-words-in-a-sentence-natural-language-processing-model

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
from tqdm import trange,tqdm
from util import load_dictionary,each_char_in
from letter_stuff import singular_punctuations,all_chars
import numpy as np
import math

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
"""model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()"""


text = "1001"
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
print(tokenized_text,indexed_tokens)
"""
segments_ids = [0] * len(indexed_tokens)

tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

with torch.no_grad():
    predictions = model(tokens_tensor, segments_tensors)

masked_indices = [i for i, x in enumerate(indexed_tokens) if x == tokenizer.vocab["[MASK]"]]

print(masked_indices)

for masked_index in masked_indices:
    preds = predictions[0, masked_index].numpy()
    sort_preds = np.argsort(-preds)
    print([(tokenizer.ids_to_tokens[x],preds[x]) for x in sort_preds[:10]])"""