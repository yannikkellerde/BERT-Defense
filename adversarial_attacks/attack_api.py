import sys,os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from phonetic_attacks.phonetically_attack import Phonetic_attacker
from visual_attacks.viper_dces import Viper_decs
from simple_attacks.simple_attacks import simple_perturb
from simple_attacks.segmentations import manip_segmentations

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
        for attack, severity in attacks_with_severity:
            sentence = self.do_one_attack(sentence, attack, severity)
        return sentence

if __name__ == "__main__":
    attack = Adversarial_attacker()
    """for method in attack.methods:
        res = attack.do_one_attack(sys.argv[1],method,1)
        print(res.count(" "),method,res)"""
    
    attacks_with_severity = [('keyboard-typo',0.2),('intrude',0.5),('segmentation',0.6)]
    res = attack.multiattack(sys.argv[1],attacks_with_severity)
    print(res.count(" "),"multiattack", res)