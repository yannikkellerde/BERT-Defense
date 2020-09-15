import numpy as np
from tqdm import tqdm
import sys
sys.path.append("..")
from util.util import fast_allmin,load_pickle,get_full_word_dict,load_dictionary
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

    def one_dist(self,source,target):
        no_vowls = True
        for char in source:
            if char in self.vowls:
                no_vowls = False
        appearance_table = self.char_appearence(source)

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
    
    def find_best_combo(self,cur_comb,cur_index,cur_dist,combo_parts,best_dist):
        best_choice = None
        fill_dist = cur_dist + len(combo_parts)-cur_index
        if fill_dist < best_dist:
            best_choice = cur_comb
            best_dist = fill_dist

        for ind in (cur_index,cur_index+1,cur_index-1):
            if ind>=0 and ind<len(combo_parts):
                for targ_in,sample_word,dist in combo_parts[ind]:
                    new_dist = cur_dist+dist+abs(ind-cur_index)
                    if len(cur_comb)>0:
                        new_dist+=0.9
                    if new_dist < best_dist:
                        comb_dist = self.find_best_combo(cur_comb+[sample_word],targ_in,new_dist,combo_parts,best_dist)
                        if comb_dist is not None:
                            best_dist = comb_dist[1]
                            best_choice = comb_dist[0]
        if best_choice is None:
            return None
        return (best_choice, best_dist)

    def word_to_prob(self,source,progress=False):
        comb_parts = [{} for _ in range(len(source))]
        distance = np.zeros(len(self.full_word_dic))
        for i,sample_word in (tqdm(enumerate(self.full_word_dic)) if progress else enumerate(self.full_word_dic)):
            dist,comb = self.one_dist(source,sample_word)
            fill_cost = len(source)-max([x[1]-x[0] for x in comb])
            real_dist = dist + fill_cost
            if fill_cost > 0:
                for c in comb:
                    if c[1]-c[0] < len(source) and c[1]>c[0] and dist<c[1]-c[0]:
                        if c[1] in comb_parts[c[0]]:
                            if comb_parts[c[0]][c[1]][1]<dist:
                                continue
                        comb_parts[c[0]][c[1]] = (sample_word,dist)
            distance[i] = real_dist
        """for sample_word in (tqdm(self.morph_dic) if progress else self.morph_dic):
            dist,comb = self.one_dist(source,sample_word)
            for c in comb:
                if c[1]-c[0] < len(source) and c[1]>c[0] and dist<c[1]-c[0]:
                        if c[1] in comb_parts[c[0]]:
                            if comb_parts[c[0]][c[1]][1]<dist:
                                continue
                        comb_parts[c[0]][c[1]] = (sample_word,dist)"""
        bestdist = np.min(distance)
        comb_parts = [list(sorted(filter(lambda x:x[2]<bestdist,[(key,val[0],val[1]) for key,val in co.items()]),key=lambda x:x[2])) for co in comb_parts]
        best_combo = self.find_best_combo([],0,0,comb_parts,bestdist)
        print(best_combo,self.full_word_dic[np.argmin(distance)],bestdist)

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
    print(sd.word_to_prob(sys.argv[1],progress=True))