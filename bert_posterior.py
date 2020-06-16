import torch
import numpy as np
from util import softmax
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import logging
from copy import deepcopy
logger = logging.getLogger()

logger.info("loading bert model ...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()
logger.info("done loading bert model")

def format_dict(dictionary):
    return [tokenizer.vocab[x] for x in dictionary]

def format_prior_and_dict(prior,dictionary):
    """prior is a WxD 2d numpy float array with W the amount of words in sentence and D the amount of words in the dictionary.
    dictionary is a list of words
    """
    new_prior=[]
    for p in prior:
        d = [(dictionary.index(x[0]), x[1]) for x in p]
        new_prior.append([x[1] for x in sorted(d)])
    new_prior = np.array(new_prior)
    return new_prior, format_dict(dictionary)

def bert_posterior(prior,dictionary,maxdepth):
    return bert_posterior_recur(prior.copy(),prior,np.zeros(len(prior)),dictionary,maxdepth)

def bert_posterior_recur(orig_prior,prior,alreadys,dictionary,maxdepth):
    """dictionary should be formated according to format_dict
    """
    if maxdepth<=0:
        return prior
    my_min = np.inf
    for i,(p,_c) in enumerate(prior):
        s = np.sort(p)
        diff = s[-1]-s[-2]
        if diff==1:
            diff = 100
        if diff+alreadys[i]<my_min:
            lowest = i
            my_min = diff+alreadys[i]
    alreadys[lowest] += 1
    sent_ray = []
    for i,(p,c) in enumerate(prior):
        pmaxin = np.argmax(p)
        if i==lowest:
            old_word = dictionary[pmaxin]
            sent_ray.append(tokenizer.vocab["[MASK]"])
        elif len(c) == 0:
            sent_ray.append(dictionary[pmaxin])
        else:
            csum = sum(x[1] for x in c)
            if p[pmaxin]/(1+csum)>c[0][1]:
            #if p[pmaxin]>c[0][1]:
                sent_ray.append(dictionary[pmaxin])
            else:
                for w in c[0][0]:
                    sent_ray.append(tokenizer.vocab[w])
    logger.debug(tokenizer.convert_ids_to_tokens(sent_ray))
    logger.debug(f"Masked token: {tokenizer.convert_ids_to_tokens([old_word])[0]}")
    indexed_tokens = [tokenizer.vocab["[CLS]"]]+sent_ray+[tokenizer.vocab["[SEP]"]]
    
    # Create the segments tensors.
    segments_ids = [0] * len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)
    
    masked_index = indexed_tokens.index(tokenizer.vocab["[MASK]"])
    preds = predictions[0, masked_index].numpy()
    likelihood = softmax(np.array([preds[dictionary[i]] for i in range(len(dictionary))]),theta=0.5)

    best_indexied = list(reversed(np.argsort(likelihood)))
    best_scores = [likelihood[index] for index in best_indexied[:10]]
    predicted_tokens = list(zip(tokenizer.convert_ids_to_tokens([dictionary[x] for x in best_indexied[:5]]),best_scores[:5]))
    logger.debug(predicted_tokens)
    if len(predicted_tokens)==0:
        logger.warn(f"{preds.shape}, {best_indexied[:5]}, {likelihood[:5]}, {best_scores[:5]}, {predicted_tokens}")
    logger.debug(f"{tokenizer.convert_ids_to_tokens([old_word])[0]}, {likelihood[dictionary.index(old_word)]}")

    numerator = orig_prior[lowest][0] * likelihood
    posterior = deepcopy(prior)
    posterior[lowest][0] = numerator/np.sum(numerator)
    return bert_posterior_recur(orig_prior,posterior,alreadys, dictionary, maxdepth-1)