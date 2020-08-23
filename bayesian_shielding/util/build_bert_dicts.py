import sys
sys.path.append("..")
from pytorch_pretrained_bert import BertTokenizer
from util.letter_stuff import *
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

morphemes = []
letter_beginnings = []
number_beginnings = []
begin_satzzeichens = []
trash_dump = []

for word in tokenizer.vocab:
    trash = False
    anum = True
    if word.startswith("[") and word.endswith("]"):
        trash = True
    for let in word:
        if let not in all_chars:
            trash = True
            break
        if let not in numbers:
            anum = False
    if trash:
        trash_dump.append(word)
    elif anum:
        number_beginnings.append(word)
    elif word.startswith("##"):
        morphemes.append(word[2:])
    else:
        if word in punctuations:
            begin_satzzeichens.append(word)
        else:
            letter_beginnings.append(word)

for alist in [morphemes,letter_beginnings,number_beginnings,begin_satzzeichens,trash_dump]:
    alist.sort(key=len)

with open("../../DATA/dictionaries/bert_morphemes.txt","w") as f:
    f.write("\n".join(morphemes))
with open("../../DATA/dictionaries/bert_letter_begin.txt","w") as f:
    f.write("\n".join(letter_beginnings))
with open("../../DATA/dictionaries/bert_number_begin.txt","w") as f:
    f.write("\n".join(number_beginnings))
with open("../../DATA/dictionaries/bert_punctuations.txt","w") as f:
    f.write("\n".join(begin_satzzeichens))
with open("../../DATA/dictionaries/bert_trash.txt","w") as f:
    f.write("\n".join(trash_dump))