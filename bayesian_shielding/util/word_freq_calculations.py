import sys
import pickle as pkl
sys.path.append("..")
from util.util import get_full_word_dict,load_dictionary

all_dict = {}
with open("../../word_frequency.txt","r") as f:
    for line in f.read().splitlines():
        all_dict[line.split(" ")[0]] = int(line.split(" ")[1])

all_freq_word = list(all_dict.items())
all_freq_word.sort(lambda x:-x[1])

word_dict = get_full_word_dict()+load_dictionary("../../DATA/dictionaries/bert_morphemes")
freq_with_word = [(value,key) for i,(key,value) in enumerate(all_freq_word) if key in word_dict or i<1000]

freqranking = {word:i for i,(num,word) in enumerate(freq_with_word)}

with open("../binaries/freq_ranking.pkl", "wb") as f:
    pickle.dump(freqranking,f)