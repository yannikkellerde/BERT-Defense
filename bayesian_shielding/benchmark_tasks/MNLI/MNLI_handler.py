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

def eval_mnli(tokenizer,model,first_sentences,second_sentences,batch_size=128):
    with torch.no_grad():
        full_res = []
        for i in range(0,len(first_sentences),batch_size):
            encoded = tokenizer(first_sentences[i:i+batch_size],second_sentences[i:i+batch_size],padding=True,return_tensors='pt')
            for key in encoded:
                encoded[key]=encoded[key].to(device)
            full_res.append(model(**encoded)[0].cpu().numpy().squeeze())
        return np.concatenate(full_res)

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