import numpy as np
import sys
sys.path.append("..")
from util.util import fast_allmin,load_pickle

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

    def __call__(self,source,target):
        appearance_table = self.char_appearence(source)

        matrix = np.zeros((len(target)+1,len(source)+1))
        startmatrix = [[set() for _ in range(len(source)+1)] for _ in range(len(target)+1)]
        for num in range(len(source)+1):
            startmatrix[0][num].add(num)

        i = 0
        for num in range(1, len(target)+1):
            i += self.in_cost(target[num-1],vowls_in)
            matrix[num][0] = i

        for i in range(1, len(target)+1):
            for j in range(1, len(source)+1):
                possibilities = [matrix[i-1][j] + self.in_cost(target[i-1]),# insertion von target_i in source
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
        return matrix,combos
    
    def in_cost(self, in_char):
        if (not self.cheap_actions["ins"]):
            return 1
        if in_char in self.vowls:
            return 0.5
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

    def char_apparence(self,word):
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