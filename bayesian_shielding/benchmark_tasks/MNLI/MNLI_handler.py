from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
import torch
import numpy as np
sys.path.append("../..")
from util.utility import read_labeled_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_mnli_model():
    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
    model.to(device)
    return tokenizer,model

def eval_mnli(tokenizer,model,first_sentences,second_sentences):
    with torch.no_grad():
        encoded = tokenizer(first_sentences,second_sentences,padding=True,return_tensors='pt')
        for key in encoded:
            encoded[key]=encoded[key].to(device)
        res = model(**encoded)
        return res[0].cpu().numpy().squeeze()

def get_mnli_accuracy(tokenizer,model,first_sentences,second_sentences,labels):
    label_map = {"contradiction":0,"neutral":1,"entailment":2}
    predictions = eval_mnli(tokenizer,model,first_sentences,second_sentences)
    corrects = 0
    for label,predict in zip(labels,predictions):
        if label_map[label] == np.argmax(predict):
            corrects+=1
    return corrects/len(labels)

if __name__=="__main__":
    labels,first_sentences,second_sentences=read_labeled_data("../../../evaluation/attacked_mnli/rand2.txt",do_float=False)
    tokenizer,model = get_mnli_model()
    print(get_mnli_accuracy(tokenizer,model,first_sentences,second_sentences,labels))