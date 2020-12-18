import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import metrics
import matplotlib.pyplot as plt

def create_evaluation_file(filename, new_file_dire):
    black_list = ["A1Z6F6EIUJFC6T:3E1QT0TDFWCVBZ86WP6F3THQXX28IA", "A1WUFHQ1YGHK3C:37U1UTWH92P84YBPMC0OHLIHKTKR8W"]
    data = pd.read_csv(filename, header=None)
    versuchspersonen = data[0].tolist()
    trail_data = data[3].tolist()
    trail_data = [trail.replace("false", "False")for trail in trail_data]
    trail_data = [trail.replace("true", "True")for trail in trail_data]
    trail_data = [eval(trail) for trail in trail_data]
    experiment_data = list(zip(versuchspersonen, trail_data))
    versuchspersonen = list(set(versuchspersonen))
    for vp_code in versuchspersonen:
        vp_data = list(filter(lambda x: x[0] == vp_code, experiment_data))
        submit = list(filter(lambda x:"status" in x[1] and x[1]["status"] == "submit", vp_data))
        if not(submit):
            experiment_data = list(filter(lambda x: x[0] != vp_code, experiment_data))
    for vp_code in black_list:
        experiment_data = list(filter(lambda x: x[0] != vp_code, experiment_data))
    experiment_data = list(filter(lambda x: x[1]["phase"] == "Experiment", experiment_data))
    conditions = [d[1]["cond"] for d in experiment_data]
    for cond in set(conditions):
        cond_data = list(filter(lambda x: x[1]["cond"] == cond, experiment_data))
        vp, _ = zip(*cond_data)
        if len(set(vp)) != 1:  
            del_vp = random.sample(set(vp),1)
            experiment_data = list(filter(lambda x: x[0] != del_vp[0], experiment_data))
    experiment_data = [(d[0], d[1]["sent"], d[1]["cleanSent"], d[1]["response"], d[1]["attack"]) for d in experiment_data]
    vp, sent, cleanSent, response, attack = zip(*experiment_data)
    sent = [_clean_data(s) for s in sent]
    cleanSent = [_clean_data(s) for s in cleanSent]
    response = [_clean_data(s) for s in response]
    attack = [a.split(".")[0] for a in attack]
    data = {"vp":vp, "sent":sent, "cleanSent":cleanSent, "response":response, "attack": attack}
    df = pd.DataFrame(data)
    df.to_csv(new_file_dire, index=False, sep="\t")

def evaluate_data(datafile):
    mover = []
    bleu = []
    edit_distance = []
    rough = []
    data = pd.read_csv(datafile, sep="\t")
    vp = data["vp"].tolist()
    reference = data["cleanSent"].tolist()
    cleaned_sent = data["response"].tolist()
    attack_typ = data["attack"].tolist()
    all_attacks = set(attack_typ)
    for ref, cleaned in tqdm(zip(reference, cleaned_sent)):
        mover.append(metrics.mover_score(ref,cleaned)[0])
        bleu.append(metrics.bleu_score(ref,cleaned))
        edit_distance.append(metrics.edit_distance(ref,cleaned,lower=True))
        rough.append(metrics.rouge_score(ref,cleaned)["rouge-1"]["p"])
    evaled_data = list(zip(vp, attack_typ, mover, bleu, edit_distance, rough))
    results = {"mover":[], "bleu":[], "edit_distance":[], "rough":[]}
    for atk in all_attacks:
        one_attack_data = list(filter(lambda x: x[1] == atk, evaled_data))
        _ , _, mover, bleu, edit_distance, rough = zip(*one_attack_data)
        results["mover"].append((atk, np.mean(mover)))
        results["bleu"].append((atk, np.mean(bleu)))
        results["edit_distance"].append((atk, np.mean(edit_distance)))
        results["rough"].append((atk, np.mean(rough)))
    build_plots(results)
    return results

def build_plots(results):
    plot_length = len(results)
    fig, axs = plt.subplots(plot_length, sharex=True)
    for key, i in zip(results, range(plot_length)):
        if key != "edit_distance":
            axs[i].set_ylim(0, 1)
        axs[i].set_title(key)
        attack, value = zip(*results[key])
        x = np.arange(len(attack))
        axs[i].bar(x, value, color="grey", tick_label=attack)
    plt.show()
        


    



def _clean_data(sentence):
    sentence = sentence.lower()
    sentence = sentence.strip()
    if sentence[len(sentence)-1] == ".":
        sentence = sentence[:-1]
    return sentence

    





if __name__ == "__main__":
    print(_clean_data("   Halo Welt.   \n"))
    #create_evaluation_file("evaluation/human_data/trialdata.csv", "evaluation/human_data/eval_data.csv")
    evaluate_data("human_data/eval_data.csv")
