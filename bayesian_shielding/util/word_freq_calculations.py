import sys
import pickle
from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm
sys.path.append("..")
from util.utility import get_full_word_dict,load_dictionary

all_dict = {}
with open("../../DATA/word_frequency.txt","r") as f:
    for line in f.read().splitlines():
        all_dict[line.split(" ")[0]] = int(line.split(" ")[1])

all_freq_word = list(all_dict.items())
all_freq_word.sort(key=lambda x:-x[1])

word_dict = get_full_word_dict()+load_dictionary("../../DATA/dictionaries/bert_morphemes.txt")

"""tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
pseudo_morphemes = [x[0] for x in all_freq_word[:150] if x[0] not in word_dict]
pseudo_morphemes = [(x,tokenizer.tokenize(x)) for x in pseudo_morphemes]
with open("../../DATA/dictionaries/pseudo_morphemes.txt", "w") as f:
    f.write("\n".join(["\t".join((x[0]," ".join(x[1]))) for x in pseudo_morphemes]))"""

freq_with_word = [(value,key) for i,(key,value) in tqdm(enumerate(all_freq_word)) if key in word_dict or i<1000]

freqranking = {word:i for i,(num,word) in enumerate(freq_with_word)}

with open("../binaries/freq_ranking.pkl", "wb") as f:
    pickle.dump(freqranking,f)