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
text = "[CLS] What is [MASK] ? [SEP]"
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
print(tokenized_text,indexed_tokens)
exit()

# Create the segments tensors.
segments_ids = [0] * len(tokenized_text)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

# Predict all tokens
with torch.no_grad():
    predictions = model(tokens_tensor, segments_tensors)

masked_index = tokenized_text.index('[MASK]')
preds = predictions[0, masked_index].numpy()
print(preds[:5])
best_indexied = list(reversed(np.argsort(preds)))
tokens = []
for i in trange(len(best_indexied)):
    predicted_token = tokenizer.convert_ids_to_tokens([best_indexied[i]])[0]
    tokens.append(predicted_token)
big_dict = [x.lower() for x in load_dictionary("DATA/wiki-100k.txt")+list(singular_punctuations)]
with open("DATA/bert_wiki_full_words.txt","w") as f:
    f.write("\n".join(list(filter(lambda x:x.lower() in big_dict,tokens))))
with open("DATA/bert_wiki_word_pieces.txt","w") as f:
    for t in tqdm(tokens):
        if t.startswith("##"):
            if each_char_in(all_chars,t[2:]):
                f.write(t[2:]+"\n")