import os
import numpy as np
import pickle
from tqdm import tqdm
import sys
from functools import lru_cache
sys.path.append("..")
from util.utility import fast_allmin,load_pickle,get_full_word_dict,load_dictionary, softmax, smallest_n_permutations
from util.letter_stuff import annoying_boys,letters,small_letters,numbers,all_chars

class Sub_dist():
    def __init__(self):
        self.vowls = set("AaEeOoUuiI")
        self.sim_matrix = np.load(os.path.join(os.path.dirname(__file__),"../binaries/vis_sim.npy"))
        self.letter_map = {x:i for i,x in enumerate(small_letters+numbers)}
        self.full_word_dic = get_full_word_dict()
        self.morph_dic = load_dictionary(os.path.join(os.path.dirname(__file__), "../../DATA/dictionaries/bert_morphemes.txt"))
        self.punc_dic = load_dictionary(os.path.join(os.path.dirname(__file__),"../../DATA/dictionaries/bert_punctuations.txt"))
        with open(os.path.join(os.path.dirname(__file__),"../binaries/freq_ranking.pkl"), "rb") as f:
            self.freq_dict = pickle.load(f)

        self.del_scaler = 0.75
        self.cheap_actions = {
            "ins":True,
            "sub":True,
            "del":True,
            "tp":True,
            "ana":True
        }
        self.num_hyps = 10
        self.min_prob = 0.07
        self.prob_softmax = 1
        self.hyp_softmax = 10
        self.freq_scale = 2
        self.hyperparams = ["prob_softmax","hyp_softmax","cheap_actions","del_scaler","num_hyps","min_prob","freq_scale"]
        self.char_distribs = {word:self.get_char_distribution(word)[0] for word in self.full_word_dic}
    
    def set_hyperparams(self,**kwargs):
        if "prob_softmax" in kwargs:
            self.prob_softmax = kwargs["prob_softmax"]
        if "hyp_softmax" in kwargs:
            self.hyp_softmax = kwargs["hyp_softmax"]
        if "del_scaler" in kwargs:
            self.del_scaler = kwargs["del_scaler"]
        if "cheap_actions" in kwargs:
            self.cheap_actions = kwargs["cheap_actions"]
        if "num_hyps" in kwargs:
            self.num_hyps = kwargs["num_hyps"]
        if "min_prob" in kwargs:
            self.min_prob = kwargs["min_prob"]
        if "freq_scale" in kwargs:
            self.freq_scale = kwargs["freq_scale"]

    def one_dist(self,source,target,no_vowls,appearance_table):
        matrix = np.zeros((len(target)+1,len(source)+1))
        startmatrix = [[set() for _ in range(len(source)+1)] for _ in range(len(target)+1)]
        for num in range(len(source)+1):
            startmatrix[0][num].add(num)

        i = 0
        for num in range(1, len(target)+1):
            startmatrix[num][0].add(0)
            i += self.in_cost(target[num-1],no_vowls)
            matrix[num][0] = i

        for i in range(1, len(target)+1):
            for j in range(1, len(source)+1):
                eq_to_last = target[i-1] == source[j-1]
                can_trans = i > 2 and j > 2 and target[i-1] == source[j-2] and target[i-2] == source[j-1]
                possibilities = [matrix[i-1][j] + self.in_cost(target[i-1],no_vowls),# insertion von target_i in source
                                    matrix[i-1][j-1] + self.sub_cost(target[i-1], source[j-1]),# substituition von target in source
                                    matrix[i][j-1] + self.del_cost(source[j-1], appearance_table)]# deletion  source_j
                if eq_to_last:
                    possibilities.append(matrix[i-1][j-1])
                if can_trans:
                    possibilities.append(matrix[i-2][j-2] + self.trans_cost())
                mins = fast_allmin(possibilities)
                for mini in mins:
                    if mini == 0:
                        startmatrix[i][j].update(startmatrix[i-1][j])
                    elif mini == 1:
                        startmatrix[i][j].update(startmatrix[i-1][j-1])
                    elif mini == 2:
                        startmatrix[i][j].update(startmatrix[i][j-1])
                    elif mini == 3:
                        if eq_to_last:
                            startmatrix[i][j].update(startmatrix[i-1][j-1])
                        elif can_trans:
                            startmatrix[i][j].update(startmatrix[i-2][j-2])
                    elif mini == 4:
                        startmatrix[i][j].update(startmatrix[i-2][j-2])
                matrix[i][j] = possibilities[mins[0]]
        endpoints = fast_allmin(matrix[-1])
        combos = []
        for endpoint in endpoints:
            for startpoint in startmatrix[-1][endpoint]:
                combos.append((startpoint,endpoint))
        return matrix[-1,endpoints[0]],combos
    
    def _insert_into_hyps(self,combo,dist,hyps):
        to_ins = (combo,dist)
        for i in range(len(hyps)):
            if to_ins[1]<hyps[i][1]:
                to_ins,hyps[i] = hyps[i],to_ins


    def find_best_hypothesis(self,cur_comb,cur_dist,combo_parts,best_hyps):
        if len(cur_comb)>5:
            return best_hyps
        fill_dist = cur_dist + len(combo_parts)-cur_comb[-1]
        if fill_dist < best_hyps[-1][1]:
            self._insert_into_hyps(cur_comb,fill_dist,best_hyps)
        if cur_comb[-1]<len(combo_parts):
            for targ_in,dist in combo_parts[cur_comb[-1]]:
                new_dist = cur_dist+dist
                if new_dist < best_hyps[-1][1]:
                    self.find_best_hypothesis(cur_comb+[targ_in],new_dist,combo_parts,best_hyps)
        return best_hyps

    def get_char_distribution(self,word):
        dic = {x:0 for x in letters}
        unknowns = 0
        for char in word:
            if char in dic:
                dic[char]+=1
            else:
                unknowns+=1
        return dic,unknowns

    def score_anagramness(self,source,target,source_dict,targ_dict,unknowns):
        ana_dist = 1+unknowns*2
        for letter in letters:
            ana_dist += abs(source_dict[letter]-targ_dict[letter])*2
        if source[0] == target[0]:
            ana_dist*=0.7
        if source[-1] == target[-1]:
            ana_dist*=0.8
        return ana_dist

    def word_to_prob(self,source,progress=False):
        no_vowls = True
        for char in source:
            if char in self.vowls or char not in all_chars:
                no_vowls = False
        appearance_table = self.char_appearence(source)

        comb_parts = [{} for _ in range(len(source))]
        combo_words = {}
        def enter_combos(comb,dist,sample_word,ismorph):
            dist += 0.5
            if sample_word in self.freq_dict and not ismorph:
                dist += ((self.freq_dict[sample_word]/len(self.freq_dict))*self.freq_scale)/len(sample_word)
            else:
                if ismorph:
                    dist+=0.3
                else:
                    if sample_word not in self.punc_dic:
                        dist+=self.freq_scale/len(sample_word)
            for c in comb:
                if c[1]-c[0] < len(source) and dist<c[1]-c[0]:
                    if ismorph and c[0]==0:
                        continue
                    for targ_in in range(max(c[0]+1,c[1]-1),min(c[1]+2,len(source)+1)):
                        true_dist = dist
                        if targ_in<c[1]:
                            for char in source[targ_in:c[1]]:
                                true_dist+=self.del_cost(char,appearance_table)
                        elif targ_in>c[1]:
                            for char in source[c[1]:targ_in]:
                                true_dist+=self.in_cost(char,no_vowls)
                        put_word = "##"+sample_word if ismorph else sample_word
                        targ_comb = (c[0],targ_in)
                        if targ_comb in combo_words:
                            combo_words[targ_comb].append((true_dist,put_word))
                        else:
                            combo_words[targ_comb] = [(true_dist,put_word)]
                        if targ_in in comb_parts[c[0]]:
                            if comb_parts[c[0]][targ_in]<true_dist:
                                continue
                        comb_parts[c[0]][targ_in] = true_dist
        distance = np.ones(len(self.full_word_dic))*100
        mindist = np.inf
        char_distrib,unknowns = self.get_char_distribution(source)
        for i,sample_word in (tqdm(enumerate(self.full_word_dic)) if progress else enumerate(self.full_word_dic)):
            if not no_vowls and len(sample_word)-len(source)>mindist+1:
                break
            dist,comb = self.one_dist(source,sample_word,no_vowls,appearance_table)
            fill_cost = len(source)-max([x[1]-x[0] for x in comb])
            real_dist = dist + fill_cost
            if fill_cost > 0:
                enter_combos(comb,dist,sample_word,False)
            if self.cheap_actions["ana"]:
                ana_score = self.score_anagramness(source,sample_word,char_distrib,self.char_distribs[sample_word],unknowns)
                distance[i] = min(real_dist,ana_score)
            else:
                distance[i] = real_dist
            mindist = min(mindist,distance[i])
        combo_words[(0,len(source))] = [distance,self.full_word_dic]
        for sample_word in (tqdm(self.morph_dic) if progress else self.morph_dic):
            dist,comb = self.one_dist(source,sample_word,no_vowls,appearance_table)
            for c in comb:
                enter_combos(comb,dist,sample_word,True)
        bestdist = np.min(distance)
        comb_parts = [list(sorted(filter(lambda x:x[1]<bestdist,co.items()),key=lambda x:x[1])) for co in comb_parts]
        orig_hyps = [((0,len(source)),bestdist)]+[(None,np.inf) for _ in range(self.num_hyps-1)]
        best_hyps = self.find_best_hypothesis([0],0,comb_parts,orig_hyps)
        hyps_with_words = []
        for hyp in best_hyps:
            if hyp[0] is not None:
                word_prob_list = []
                for i in range(len(hyp[0])-1):
                    if (hyp[0][i],hyp[0][i+1]) == (0,len(source)):
                        cw = (distance,self.full_word_dic)
                    else:
                        cw = tuple(zip(*combo_words[(hyp[0][i],hyp[0][i+1])]))
                    word_prob_list.append(cw)
                hyps_with_words.append(hyp+(tuple(word_prob_list),))
        return hyps_with_words

    def get_sentence_hypothesis(self,sentence,progress=False):
        word_hyps = [self.word_to_prob(x,progress=progress) for x in sentence]
        permuts = smallest_n_permutations([[y[1] for y in x] for x in word_hyps],self.num_hyps)
        hyps = []
        for edit_dist,cur_ind in permuts:
            full_hyp = [x[j] for x,j in zip(word_hyps,cur_ind)]
            newhyp = sum((x[2] for x in full_hyp),tuple())
            newhyp = [(softmax(-np.array(x[0]),self.prob_softmax),x[1]) for x in newhyp]
            hyps.append((edit_dist,newhyp))
        unp_hyps = list(zip(*hyps))
        smax = softmax(-np.array(unp_hyps[0]),self.hyp_softmax)
        max_prob = max(smax)
        hyps = tuple(sorted(filter(lambda x:x[0]>self.min_prob*max_prob,zip(smax,unp_hyps[1])),key=lambda x:-x[0]))
        return hyps[:self.num_hyps]

    def show_hyp_max(self,hyp):
        print(hyp[0]," ".join([x[1][np.argmax(x[0])] for x in hyp[1]]))

    def in_cost(self, in_char, no_vowls):
        if (not self.cheap_actions["ins"]):
            return 1
        if no_vowls and in_char in self.vowls:
            return 0.3
        else:
            return 1

    def sub_cost(self,letnum,unic):
        if (not self.cheap_actions["sub"] or unic in all_chars or letnum not in self.letter_map):
            return 1
        return max(0,min(1,(0.8-self.sim_matrix[ord(unic),self.letter_map[letnum]])*3))

    def del_cost(self, del_char, table):
        if (not self.cheap_actions["del"]):
            return 1
        scal_cost = self.del_scaler**(table[del_char]-1)
        return scal_cost

    def trans_cost(self):
        return 1

    def char_appearence(self,word):
        table = {}
        for c in word:
            if c in table:
                table[c] = table[c] + 1
            else:
                if c in letters:
                    table[c] = 1
                else:
                    table[c] = 3
        return table

if __name__ == "__main__":
    sd = Sub_dist()
    #print(sd.sub_cost("b","ǟ"))
    #exit()
    res = sd.word_to_prob(sys.argv[1],progress=True)
    for wuff in res:
        print(wuff[0],wuff[1],[x[1][np.argmin(x[0])] for x in wuff[2]])
    #res = sd.get_sentence_hypothesis(sys.argv[1].split(" "),progress=True)
    #for x in res:
    #    sd.show_hyp_max(x)