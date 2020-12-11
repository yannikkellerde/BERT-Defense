import os,sys
import random
import pandas as pd
sys.path.append("../bayesian_shielding")
from util.utility import read_labeled_data


def create_human_test_data(ground_truth, attack_path, attacked_documents, sentence_per_attacker ,num_per_sent, num_cond):
    test_data = []
    _, tfirst_sentences, tsecond_sentences = read_labeled_data(ground_truth)
    correct_sentences = tfirst_sentences + tsecond_sentences
    samples = random.sample(range(len(correct_sentences)), sentence_per_attacker)
    for sample in samples:
        for _ in range(num_per_sent):
            test_data.append((correct_sentences[sample], correct_sentences[sample], "correct.txt"))
    for path in attacked_documents:
        complete_path = os.path.join("attacked_documents",path)
        _, first_sentences, second_sentences = read_labeled_data(complete_path)
        attacked_sentences = first_sentences + second_sentences
        samples = random.sample(range(len(attacked_sentences)), sentence_per_attacker)
        for sample in samples:
            for _ in range(num_per_sent):
                test_data.append((attacked_sentences[sample], correct_sentences[sample], path))
    random.shuffle(test_data)
    cond_list = []
    cond_list = [[] for i in range(num_cond)]
    for i in range(len(test_data)):
        cond_list = list(sorted(cond_list, key=lambda x: len(x), reverse=False))
        for cond in range(len(cond_list)):
            if not(cond_list[cond]):
                cond_list[cond].append(test_data[i])
                break
            else:
               stim, truth, catagory = zip(*cond_list[cond])
               if not(test_data[i][1] in truth):
                   cond_list[cond].append(test_data[i])
                   break
            if cond == len(cond_list)-1:
                print("Warning Element not added")
    for condition, i in zip(cond_list, range(len(cond_list))):
        stim, truth, catagory = zip(*condition)
        data = {"stimulus":stim, "Truth":truth, "Class":catagory}
        df = pd.DataFrame(data=data)
        df.to_csv(f"exp_data/csv_data_cond{i}", index=False)


def get_all_attacked_docs(attack_path):
    all_attacks = os.listdir(attack_path)
    for attack in all_attacks:
        print(attack)
    

if __name__ == "__main__":
    ground_truth = "../bayesian_shielding/benchmark_tasks/STSB/400_sentences.csv"
    attack_path = "attacked_documents"
    attacked_documents = ["rand2.txt", "phonetic.txt", "phonetic_70.txt", "visual.txt",
                          "full_swap.txt", "natural_typo.txt", "segment_typo.txt",
                          "disemvowel.txt", "rand_hard.txt"]
    sentence_per_attacker = 40
    get_all_attacked_docs(attack_path)
    create_human_test_data(ground_truth, attack_path, attacked_documents, sentence_per_attacker, 4, 27)