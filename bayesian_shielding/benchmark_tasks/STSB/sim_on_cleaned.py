import sys
sys.path.append("../..")
from STSB.RoBERTa_handler import init_model_roberta,encode_one_sentence
from util.utility import cosine_similarity
import numpy as np
from tqdm import tqdm
import sys

def sim_on_cleaned(filename,output,which_model="second"):
    with open(filename,"r") as f:
        cleaned_dataset = [x.split("\t") for x in f.read().splitlines()]
    
    if which_model == "second":
        scores = get_similarity(cleaned_dataset)
    else:
        model = init_model_roberta()
        scores = []
        for line in tqdm(cleaned_dataset):
            s1 = encode_one_sentence(model,line[0])
            s2 = encode_one_sentence(model,line[1])
            scores.append(cosine_similarity(s1,s2))
    scores = np.clip(scores, 0.0, 1.0)
    with open(output,"w") as f:
        f.write("\n".join(list(map(str,scores))))

if __name__ == "__main__":
    sim_on_cleaned(*sys.argv[1:])