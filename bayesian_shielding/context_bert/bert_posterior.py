import sys
sys.path.append("..")
import torch
import numpy as np
from util.util import softmax,get_most_likely_sentence,get_full_word_dict
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
from context_bert.probabilistic_bert import my_BertForMaskedLM
import logging
from copy import deepcopy
full_word_dic = get_full_word_dict()
logger = logging.getLogger()

logger.info("loading bert model ...")
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-large-uncased')
model.eval()

probmodel = my_BertForMaskedLM.from_pretrained('bert-large-uncased')
probmodel.eval()

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

def convert_prior_to_weights_tensor(prior,dictionary):
    weights_tensor = torch.zeros((len(prior),len(tokenizer.vocab)))
    for i,p in enumerate(prior):
        for j,weight in enumerate(p):
            weights_tensor[i][dictionary[j]] = weight
    return weights_tensor

mask_tensor = torch.zeros(len(tokenizer.vocab))
mask_tensor[tokenizer.vocab["[MASK]"]]=1
def bert_posterior_probabilistic(prior,dictionary,iterations_left):
    if iterations_left <= 0:
        return prior
    with torch.no_grad():
        weights_tensor = convert_prior_to_weights_tensor(prior,dictionary)
        likelihood = np.empty_like(prior)
        for mask_id in range(len(prior)):
            inner_tensor = weights_tensor.clone().reshape((1,len(prior),len(tokenizer.vocab)))
            inner_tensor[0][mask_id] = mask_tensor
            predictions = probmodel(inner_tensor)
            preds = predictions[0, mask_id].numpy()
            likelihood[mask_id] = softmax(np.array([preds[dictionary[i]] for i in range(len(dictionary))]),theta=0.5)
    posterior_numerator = prior*likelihood
    posterior = (posterior_numerator.T/np.sum(posterior_numerator,axis=1)).T
    print("likelihood",get_most_likely_sentence(likelihood,full_word_dic))
    print("posterior",get_most_likely_sentence(posterior,full_word_dic))
    return bert_posterior_probabilistic(posterior, dictionary, iterations_left-1)
        


def bert_posterior_recur(orig_prior,prior,alreadys,dictionary,maxdepth):
    """dictionary should be formated according to format_dict
    """
    if maxdepth <= 0:
        return prior
    my_min = np.inf
    for i,p in enumerate(prior):
        s = np.partition(-p,1)
        diff = s[1]-s[0]
        if diff == 1 or s[0]==0:
            diff = 100
        if diff+alreadys[i]<my_min:
            lowest = i
            my_min = diff+alreadys[i]
    alreadys[lowest] += 1
    sent_ray = []
    for i,p in enumerate(prior):
        pmaxin = np.argmax(p)
        if i==lowest:
            old_word = dictionary[pmaxin]
            sent_ray.append(tokenizer.vocab["[MASK]"])
        sent_ray.append(dictionary[pmaxin])

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

    numerator = orig_prior[lowest] * likelihood
    posterior = deepcopy(prior)
    posterior[lowest] = numerator/np.sum(numerator)
    return bert_posterior_recur(orig_prior,posterior,alreadys, dictionary, maxdepth-1)