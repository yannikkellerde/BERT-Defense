import sys,os
basepath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(basepath)
sys.path.append(os.path.join(basepath,"..","bayesian_shielding"))
from phonetic_attacks.phonetically_attack import Phonetic_attacker
from visual_attacks.viper_dces import Viper_decs
from simple_attacks.simple_attacks import simple_perturb
from simple_attacks.segmentations import manip_segmentations
from util.utility import read_labeled_data
from tqdm import tqdm
import random

class Adversarial_attacker():
    def __init__(self):
        self.phonetic_attacker = Phonetic_attacker(stats_folder=os.path.join(os.path.realpath(os.path.dirname(__file__)),"phonetic_attacks/statistics"))
        self.visual_attacker = Viper_decs()
        self.methods = ['visual','phonetic','full-swap','inner-swap','disemvowel','truncate','keyboard-typo','natural-typo','intrude','segmentation']

    def do_one_attack(self,sentence,method,severity):
        if method not in self.methods:
            raise ValueError("Invalid method")
        if method == "visual":
            return self.visual_attacker(sentence,severity)
        elif method == "phonetic":
            return self.phonetic_attacker(sentence,severity)
        elif method == "segmentation":
            return manip_segmentations(sentence,severity)
        else:
            return simple_perturb(sentence,method,severity)

    def multiattack(self, sentence, attacks_with_severity):
        """Perform multiple consecutive attacks
        :param sentence: Sentence to attack
        :param attacks_with_severity: List of tuples containing method and severity
        :return: The attacked sentence
        """
        for i,(attack, severity) in enumerate(attacks_with_severity):
            if attack=="rand":
                attack = random.choice(self.methods)
                while attack=="intrude" and i!=len(attacks_with_severity)-1:
                    attack = random.choice(self.methods)
            sentence = self.do_one_attack(sentence, attack, severity)
        return sentence

    def multiattack_document(self, infile, outfile, attacks_with_severity):
        """Attack all sentences in document with a set of attacks
        :param infile: A file expected to be in the STS-B format.
        :param outfile: The output will be written to this file, with a comment about
                        the attacks in the first line
        :param attacks_with_severity: List of tuples containing method and severity
        """
        scores,first_sentences, second_sentences = read_labeled_data(infile,do_float=False)
        pert_first = [self.multiattack(sentence,attacks_with_severity) for sentence in tqdm(first_sentences)]
        pert_second = [self.multiattack(sentence,attacks_with_severity) for sentence in tqdm(second_sentences)]
        with open(outfile, 'w') as f:
            f.write("# "+str(attacks_with_severity)+"\n")
            f.write("\n".join(f"{sc}\t{fs}\t{ss}" for sc,fs,ss in zip(scores,pert_first,pert_second)))

if __name__ == "__main__":
    attack = Adversarial_attacker()
    attacks_with_severity = [("visual",1)]
    print(attack.multiattack(sys.argv[1],attacks_with_severity))
    #doc = "../bayesian_shielding/benchmark_tasks/STSB/400_sentences.csv"
    #doc = "../bayesian_shielding/benchmark_tasks/MNLI/mnli_dataset.csv"
    #attack.multiattack_document(doc,"../evaluation/attacked_mnli/rand2.txt",attacks_with_severity)