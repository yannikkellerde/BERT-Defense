from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
import torch
import numpy as np
sys.path.append("../..")
from util.utility import cosine_similarity

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_bert_model():
    tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-STS-B")
    model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-STS-B")
    model.to(device)
    return tokenizer,model

def get_similarity(tokenizer,model,s1,s2):
    with torch.no_grad():
        encoded = tokenizer(s1,s2,padding=True,return_tensors='pt')
        for key in encoded:
            encoded[key]=encoded[key].to(device)
        res = model(**encoded)
        return res[0].cpu().numpy().squeeze()/5 # Default scale goes from 0 to 5

if __name__ == "__main__":
    tokenizer,model = get_bert_model()
    print(get_similarity(tokenizer,model,["Person with two ski poles skiing down a snowy hill.","A man is riding a horse."],["A person skiing down a snowy hill.","A man is driving a car."]))
    print(get_similarity(tokenizer,model,"A man is riding a horse.","A man is driving a car."))
    print(get_similarity(tokenizer,model,"A gray cat laying on a brown table.","A grey cat lying on a wooden table."))
    print(get_similarity(tokenizer,model,"A man is running on a street.","A man plays the guitar."))