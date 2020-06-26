# Code based on https://github.com/huggingface/transformers/blob/master/model_cards/SparkBeyond/roberta-large-sts-b/README.md

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import toolz
import torch
import time
batch_size = 6

tokenizer = AutoTokenizer.from_pretrained("SparkBeyond/roberta-large-sts-b")

model = AutoModelForSequenceClassification.from_pretrained("SparkBeyond/roberta-large-sts-b").cuda()

def roberta_similarity_batches(to_predict):
    batches = toolz.partition(batch_size, to_predict)
    similarity_scores = []    
    for batch in batches: 
        sentences = [(sentence_similarity["sent1"], sentence_similarity["sent2"])  for sentence_similarity in batch]   
        batch_scores = similarity_roberta(sentences)
        similarity_scores = similarity_scores + batch_scores[0].cpu().squeeze(axis=1).tolist()
    return similarity_scores

def similarity_roberta(sent_pairs):
    with torch.no_grad():
        batch_token = tokenizer.batch_encode_plus(sent_pairs, pad_to_max_length=True, max_length=500)
        res = model(torch.tensor(batch_token['input_ids']).cuda(), attention_mask=torch.tensor(batch_token["attention_mask"]).cuda())    
        return res

def get_similarity(sent_pairs):
    return similarity_roberta(sent_pairs)[0].cpu().numpy().reshape((len(sent_pairs),))/5

if __name__ == '__main__':
    start = time.perf_counter()
    b = get_similarity([['Two men standing on beach.',
                                        'Two women standing in front of tour bus.'],
                                        ['Macau Gambling Revenue Hits Record $38 bn in 2012',
                                        'Macau gambling revenue hits record US$38b in 2012']])
    print(time.perf_counter()-start)
    print(b)