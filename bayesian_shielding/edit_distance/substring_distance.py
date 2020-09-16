import numpy as np
import pickle
from tqdm import tqdm
import sys
sys.path.append("..")
from util.utility import fast_allmin,load_pickle,get_full_word_dict,load_dictionary, softmax
from util.letter_stuff import annoying_boys

class Sub_dist():
    def __init__(self):
        self.vowls = set("AaEeOoUu")
        self.del_scaler = 0.75
        self.cheap_actions = {
            "ins":True,
            "sub":True,
            "del":True,
            "tp":True
        }
        self.word_embedding = load_pickle("../binaries/visual_embeddings.pkl")
        self.full_word_dic = get_full_word_dict()
        self.morph_dic = load_dictionary("../../DATA/dictionaries/bert_morphemes.txt")
        self.punc_dic = load_dictionary("../../DATA/dictionaries/bert_punctuations.txt")
        self.prob_softmax = 27
        self.hyp_softmax = 30
        with open("../binaries/freq_ranking.pkl", "rb") as f:
            self.freq_dict = pickle.load(f)

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
                possibilities = [matrix[i-1][j] + self.in_cost(target[i-1],no_vowls),# insertion von target_i in source
                                    matrix[i-1][j-1] + self.sub_cost(target[i-1], source[j-1]),# substituition von target in source
                                    matrix[i][j-1] + self.del_cost(source[j-1], appearance_table)]# deletion  source_j
                if target[i-1] == source[j-1]:
                    possibilities.append(matrix[i-1][j-1])
                mins = fast_allmin(possibilities)
                for mini in mins:
                    if mini == 0:
                        startmatrix[i][j].update(startmatrix[i-1][j])
                    elif mini == 1 or mini == 3:
                        startmatrix[i][j].update(startmatrix[i-1][j-1])
                    elif mini == 2:
                        startmatrix[i][j].update(startmatrix[i][j-1])
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
        fill_dist = cur_dist + len(combo_parts)-cur_comb[-1]
        if fill_dist < best_hyps[-1][1]:
            self._insert_into_hyps(cur_comb,fill_dist,best_hyps)
        if cur_comb[-1]<len(combo_parts):
            for targ_in,dist in combo_parts[cur_comb[-1]]:
                new_dist = cur_dist+dist
                if new_dist < best_hyps[-1][1]:
                    self.find_best_hypothesis(cur_comb+[targ_in],new_dist,combo_parts,best_hyps)
        return best_hyps

    def word_to_prob(self,source,num_hyps=10,progress=False,freq_scale=2):
        no_vowls = True
        for char in source:
            if char in self.vowls:
                no_vowls = False
        appearance_table = self.char_appearence(source)

        comb_parts = [{} for _ in range(len(source))]
        combo_words = {}
        def enter_combos(comb,dist,sample_word,ismorph):
            dist += 0.5
            if sample_word in self.freq_dict and not ismorph:
                dist += ((self.freq_dict[sample_word]/len(self.freq_dict))*freq_scale)/len(sample_word)
            else:
                if ismorph:
                    dist+=0.4
                else:
                    if sample_word not in self.punc_dic:
                        dist+=freq_scale/len(sample_word)
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
                        if targ_in in comb_parts[c[0]]:
                            if comb_parts[c[0]][targ_in]<true_dist:
                                continue
                        comb_parts[c[0]][targ_in] = true_dist
                        put_word = "##"+sample_word if ismorph else sample_word
                        targ_comb = (c[0],targ_in)
                        if targ_comb in combo_words:
                            combo_words[targ_comb].append((true_dist,put_word))
                        else:
                            combo_words[targ_comb] = [(true_dist,put_word)]
        distance = np.zeros(len(self.full_word_dic))
        for i,sample_word in (tqdm(enumerate(self.full_word_dic)) if progress else enumerate(self.full_word_dic)):
            dist,comb = self.one_dist(source,sample_word,no_vowls,appearance_table)
            fill_cost = len(source)-max([x[1]-x[0] for x in comb])
            real_dist = dist + fill_cost
            if fill_cost > 0:
                enter_combos(comb,dist,sample_word,False)
            distance[i] = real_dist
        combo_words[(0,len(source))] = [distance,self.full_word_dic]
        for sample_word in (tqdm(self.morph_dic) if progress else self.morph_dic):
            dist,comb = self.one_dist(source,sample_word,no_vowls,appearance_table)
            for c in comb:
                enter_combos(comb,dist,sample_word,True)
        bestdist = np.min(distance)
        comb_parts = [list(sorted(filter(lambda x:x[1]<bestdist,co.items()),key=lambda x:x[1])) for co in comb_parts]
        orig_hyps = [((0,len(source)),bestdist)]+[(None,np.inf) for _ in range(num_hyps-1)]
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
                hyps_with_words.append(hyp+(word_prob_list,))
        return hyps_with_words

    def get_sentence_hypothesis(self,sentence,max_hyps=10,max_worse_percent=0.2,progress=False):
        word_hyps = [self.word_to_prob(x,num_hyps=max_hyps,progress=progress) for x in sentence]
        cur_ind = [0]*len(word_hyps)
        best_edit_dist = None
        hyps = []
        for i in range(max_hyps):
            full_hyp = [x[i] for x,i in zip(word_hyps,cur_ind)]
            print(full_hyp[0])
            newhyp = sum((x[2] for x in full_hyp),tuple())
            newhyp = [(softmax(1/(x[0]+1),self.prob_softmax),x[1]) for x in newhyp]
            edit_dist = sum(x[1] for x in full_hyp)
            if best_edit_dist is None:
                best_edit_dist = edit_dist
            hyps.append((edit_dist,newhyp))
            best_new_dist = best_edit_dist*(1+max_worse_percent)
            bestj = None
            for j in range(len(cur_ind)):
                if len(word_hyps[j])>cur_ind[j]+1:
                    maydist = edit_dist - word_hyps[j][cur_ind[j]] + word_hyps[j][cur_ind[j]+1]
                    if maydist < best_new_dist:
                        bestj = j
                        best_new_dist = maydist
            if bestj is None:
                break
            cur_ind[j] += 1
        unp_hyps = list(zip(*hyps))
        smax = softmax(1/(unp_hyps[0]+1),self.hyp_softmax)
        hyps = tuple(zip(smax,unp_hyps[1]))
        return hyps

    def show_hyp_max(hyp):
        print(hyp[0]," ".join([x[1][np.argmax(x[0])] for x in hyp[1]]))

    def in_cost(self, in_char, no_vowls):
        if (not self.cheap_actions["ins"]):
            return 1
        if no_vowls and in_char in self.vowls:
            return 0.3
        else:
            return 1

    def sub_cost(self, char1, char2):
        if (not self.cheap_actions["sub"]):
            return 1
        vek1 = self.word_embedding[ord(char1)]
        vek2 = self.word_embedding[ord(char2)]
        return min((1 - vek1@vek2)*2,1)

    def del_cost(self, del_char, table):
        if (not self.cheap_actions["del"]):
            return 1
        scal_cost = self.del_scaler**(table[del_char]-1)
        return scal_cost

    def char_appearence(self,word):
        table = {}
        for c in word:
            if c in table:
                table[c] = table[c] + 1
            else:
                if c in annoying_boys:
                    table[c] = 3
                else:
                    table[c] = 1
        return table

if __name__ == "__main__":
    sd = Sub_dist()
    res = sd.get_sentence_hypothesis(sys.argv[1].split(" "),progress=True)
    for x in res:
        sd.show_hyp_max(x)