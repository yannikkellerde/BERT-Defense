import sys
sys.path.append("..")
import torch
import numpy as np
from util.utility import softmax,get_most_likely_sentence,get_full_word_dict
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
from context_bert.probabilistic_bert import my_BertForMaskedLM
import logging
from copy import deepcopy

class BertPosterior():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

        self.model = BertForMaskedLM.from_pretrained('bert-large-uncased')
        self.model.eval()
        self.probmodel = my_BertForMaskedLM.from_pretrained('bert-large-uncased')
        self.probmodel.eval()

        self.word_dic = get_full_word_dict()
        self.bert_dic = [self.tokenizer.vocab[x] for x in self.word_dic]

        self.mask_tensor = torch.zeros(len(self.tokenizer.vocab))
        self.mask_tensor[self.tokenizer.vocab["[MASK]"]]=1


    def convert_prior_to_weights_tensor(self,prior,top_n):
        weights_tensor = torch.zeros((len(prior)+2,len(self.tokenizer.vocab)))
        weights_tensor[0][self.tokenizer.vocab["[CLS]"]]=1
        weights_tensor[len(prior)+1][self.tokenizer.vocab["[SEP]"]]=1
        for i,p in enumerate(prior):
            top_indices = np.flip(np.argsort(p))[:top_n]
            mysum = np.sum(p[top_indices])
            for j in top_indices:
                weights_tensor[i+1][self.bert_dic[j]] = p[j]/mysum
        return weights_tensor


    def calc_baseline(self,weights_tensor,mask_id):
        # Might become useless soon
        base = torch.tensor(np.ones((1,weights_tensor.size(0),weights_tensor.size(1)))/weights_tensor.size(1),dtype=torch.long)
        base[0][mask_id] = mask_tensor
        predictions = self.probmodel(base)
        preds = predictions[0, mask_id].numpy()
        return softmax(np.array([preds[self.bert_dic[i]] for i in range(len(self.bert_dic))]),theta=0.4)

    def showprobs(self,probs,amount=20):
        inds = np.flip(np.argsort(probs))
        return [[self.word_dic[inds[i]],probs[inds[i]]] for i in range(amount)]

    def calc_probabilistic_likelihood(self,weights_tensor,mask_id,theta=0.5):
        inner_tensor = weights_tensor.clone().reshape((1,weights_tensor.size(0),weights_tensor.size(1)))
        inner_tensor[0][mask_id] = self.mask_tensor
        predictions = self.probmodel(inner_tensor)
        preds = predictions[0, mask_id].numpy()
        return softmax(np.array([preds[self.bert_dic[i]] for i in range(len(self.bert_dic))]),theta=theta)

    def bert_posterior_probabilistic_rounds(self,prior,top_n,iterations_left,theta=0.5):
        if iterations_left <= 0:
            return prior
        with torch.no_grad():
            weights_tensor = self.convert_prior_to_weights_tensor(prior,top_n)
            likelihood = np.empty_like(prior)
            for mask_id in range(1,len(prior)+1):
                likelihood[mask_id-1] = self.calc_probabilistic_likelihood(weights_tensor,mask_id,theta)
        posterior_numerator = prior*likelihood
        posterior = (posterior_numerator.T/np.sum(posterior_numerator,axis=1)).T
        print("likelihood",get_most_likely_sentence(likelihood,self.word_dic))
        print("posterior",get_most_likely_sentence(posterior,self.word_dic))
        return self.bert_posterior_probabilistic_rounds(posterior,top_n,iterations_left-1)

    def bert_posterior_probabilistic_live(self,prior,top_n,iterations_left,alreadys=None,orig_prior=None):
        if iterations_left <= 0:
            return prior
        if alreadys is None:
            alreadys = np.zeros(len(prior))
        mask_id = self.get_most_uncertain_index(prior,alreadys)+1
        alreadys[mask_id-1] += 1

        with torch.no_grad():
            weights_tensor = self.convert_prior_to_weights_tensor(prior,top_n)
            likelihood = self.calc_probabilistic_likelihood(weights_tensor,mask_id)
        if orig_prior is None:
            numerator = prior[mask_id-1] * likelihood
        else:
            numerator = orig_prior[mask_id-1] * likelihood
        prior[mask_id-1] = numerator/np.sum(numerator)
        print("masked",mask_id-1)
        print("likelihood",self.showprobs(likelihood,10))
        print("posterior",get_most_likely_sentence(prior,self.word_dic))
        return self.bert_posterior_probabilistic_live(prior,top_n,iterations_left-1,alreadys,orig_prior)

    def get_most_uncertain_index(self,prior,alreadys):
        my_min = np.inf
        for i,p in enumerate(prior):
            s = np.partition(-p,1)
            diff = s[1]-s[0]
            if diff == 1 or s[0]==0:
                diff = 100
            if diff+alreadys[i]<my_min:
                lowest = i
                my_min = diff+alreadys[i]
        return lowest

    def bert_posterior_old(self,prior,maxdepth):
        return self.bert_posterior_recur_old(prior.copy(),prior,np.zeros(len(prior)),maxdepth)

    def bert_posterior_recur_old(self,orig_prior,prior,alreadys,maxdepth):
        if maxdepth <= 0:
            return prior
        lowest = self.get_most_uncertain_index(prior,alreadys)
        alreadys[lowest] += 1
        sent_ray = []
        for i,p in enumerate(prior):
            pmaxin = np.argmax(p)
            if i==lowest:
                old_word = self.bert_dic[pmaxin]
                sent_ray.append(self.tokenizer.vocab["[MASK]"])
            sent_ray.append(self.bert_dic[pmaxin])

        indexed_tokens = [self.tokenizer.vocab["[CLS]"]]+sent_ray+[self.tokenizer.vocab["[SEP]"]]
        
        # Create the segments tensors.
        segments_ids = [0] * len(indexed_tokens)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Predict all tokens
        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)
        
        masked_index = indexed_tokens.index(self.tokenizer.vocab["[MASK]"])
        preds = predictions[0, masked_index].numpy()
        likelihood = softmax(np.array([preds[self.bert_dic[i]] for i in range(len(self.bert_dic))]),theta=0.5)

        best_indexied = list(reversed(np.argsort(likelihood)))
        best_scores = [likelihood[index] for index in best_indexied[:10]]
        predicted_tokens = list(zip(self.tokenizer.convert_ids_to_tokens([self.bert_dic[x] for x in best_indexied[:5]]),best_scores[:5]))

        numerator = orig_prior[lowest] * likelihood
        posterior = deepcopy(prior)
        posterior[lowest] = numerator/np.sum(numerator)
        return self.bert_posterior_recur_old(orig_prior,posterior,alreadys, maxdepth-1)