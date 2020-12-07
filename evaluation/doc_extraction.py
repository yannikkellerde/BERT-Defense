import os,sys
import random
import pandas as pd
sys.path.append("../bayesian_shielding")
from util.utility import read_labeled_data


def create_human_test_data(ground_truth, attack_path, attacked_documents, sentence_per_doc ,num_per_sent):
    test_data = []
    _, tfirst_sentences, tsecond_sentences = read_labeled_data(ground_truth)
    correct_sentences = tfirst_sentences + tsecond_sentences
    for path, sent_num in zip(attacked_documents, sentence_per_doc):
        complete_path = os.path.join("attacked_documents",path)
        _, first_sentences, second_sentences = read_labeled_data(complete_path)
        attacked_sentences = first_sentences + second_sentences
        samples = random.sample(range(len(attacked_sentences)), sent_num)
        for sample in samples:
            for _ in range(num_per_sent):
                test_data.append((attacked_sentences[sample], correct_sentences[sample], path))
    random.shuffle(test_data)
    stim, truth, catagory = zip(*test_data)
    data = {"stimulus":stim, "Truth":truth, "Class":catagory}
    df = pd.DataFrame(data=data)
    df.to_csv("experiment_data", index=False)







def get_all_attacked_docs(attack_path):
    all_attacks = os.listdir(attack_path)
    for attack in all_attacks:
        print(attack)
    




if __name__ == "__main__":
    ground_truth = "../bayesian_shielding/benchmark_tasks/STSB/400_sentences.csv"
    attack_path = "attacked_documents"
    attacked_documents = ["phonetic_10.txt", "phonetic_50.txt", "phonetic_70.txt", "phonetic_90.txt", "rand1.txt"]
    sentence_per_doc = [20, 20, 20, 20, 40]
    get_all_attacked_docs(attack_path)
    create_human_test_data(ground_truth, attack_path, attacked_documents, sentence_per_doc, 3)