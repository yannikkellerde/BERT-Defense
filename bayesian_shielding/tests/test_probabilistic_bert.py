import sys
sys.path.append("..")
from context_bert.probabilistic_bert import my_BertForMaskedLM
from pytorch_pretrained_bert import BertTokenizer,BertForMaskedLM
import torch


tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
#print(tokenizer.vocab["man"])
#exit()
model = BertForMaskedLM.from_pretrained('bert-large-uncased')
model.eval()

probmodel = my_BertForMaskedLM.from_pretrained('bert-large-uncased')
probmodel.eval()
print("done loading probmodel")



sent_1 = "[CLS] a man is [MASK] a keyboard . [SEP]"
sent_2 = "[CLS] a squirrel runs [MASK] in circles . [SEP]"
mask_index = 4


tokens = torch.tensor([[tokenizer.vocab[x] for x in sent_1.split(" ")]])
with torch.no_grad():
    predictions = model(tokens)
    preds = predictions[0, mask_index].numpy()
dump = []
for word,tid in tokenizer.vocab.items():
    dump.append([preds[tid],word])
dump.sort(reverse=True)
print("\n Man normal")
print(dump[:40])

tokens = torch.tensor([[tokenizer.vocab[x] for x in sent_2.split(" ")]])
with torch.no_grad():
    predictions = model(tokens)
    preds = predictions[0, mask_index].numpy()
dump = []
for word,tid in tokenizer.vocab.items():
    dump.append([preds[tid],word])
dump.sort(reverse=True)
print("\n squirrel normal")
print(dump[:40])

sent_1_parts = sent_1.split(" ")
sent_2_parts = sent_2.split(" ")

weights_tensor = torch.zeros((1,len(sent_1_parts),len(tokenizer.vocab)))
for i in range(len(sent_1_parts)):
    weights_tensor[0][i][tokenizer.vocab[sent_1_parts[i]]] += 0.5
    weights_tensor[0][i][tokenizer.vocab[sent_2_parts[i]]] += 0.5

with torch.no_grad():
    predictions = probmodel(weights_tensor)
    preds = predictions[0, mask_index].numpy()

dump = []
for word,tid in tokenizer.vocab.items():
    dump.append([preds[tid],word])
dump.sort(reverse=True)
print("\n combo")
print(dump[:40])