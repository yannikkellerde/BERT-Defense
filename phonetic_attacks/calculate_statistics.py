import pickle
import os
from tqdm import tqdm

def insert_to_ngrams(ngram,phonemes,letters):
    if phonemes in ngram:
        if letters in ngram[phonemes]:
            ngram[phonemes][letters]+=1
        else:
            ngram[phonemes][letters]=1
    else:
        ngram[phonemes] = {letters:1}

def calc_statistics(infile="cmudict.aligned",out_folder="statistics"):
    gram_1 = {}
    gram_2 = {}
    gram_3 = {}
    cmu_mapping = {}
    with open(infile, "r") as f:
        lines = f.read().splitlines()
    for line in tqdm(lines):
        letters,phonemes = [["[START]"]+x.split("|")[:-1]+["[END]"] for x in line.split("\t")]
        cmu_mapping["".join(letters[1:-1]).replace(":","")] = (tuple(letters),tuple(phonemes))
        for i in range(1,len(letters)-1):
            insert_to_ngrams(gram_1,(phonemes[i],),(letters[i],))
            insert_to_ngrams(gram_2,(phonemes[i-1],phonemes[i]),(letters[i-i],letters[i]))
            insert_to_ngrams(gram_3,(phonemes[i-1],phonemes[i],phonemes[i+1]),(letters[i-i],letters[i],letters[i+1]))
    with open(os.path.join(out_folder,"gram_1.pkl"),"wb") as f:
        pickle.dump(gram_1,f)
    with open(os.path.join(out_folder,"gram_2.pkl"),"wb") as f:
        pickle.dump(gram_2,f)
    with open(os.path.join(out_folder,"gram_3.pkl"),"wb") as f:
        pickle.dump(gram_3,f)
    with open(os.path.join(out_folder,"cmu_map.pkl"),"wb") as f:
        pickle.dump(cmu_mapping,f)

if __name__=="__main__":
    calc_statistics()