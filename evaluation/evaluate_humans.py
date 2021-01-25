import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import metrics
from scipy.stats import ttest_ind
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
    return results, evaled_data

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
        

def evaluate_vp(scored_data):
    data = pd.read_csv("human_data/eval_data.csv", sep="\t")
    question_data = pd.read_csv("human_data/questiondata.csv", header=None)
    vp = data["vp"].tolist()
    vp_q = question_data[0].tolist()
    question_typ = question_data[1].tolist()
    ans = question_data[2].tolist()
    questions = zip(vp_q, question_typ, ans)
    print("Total subjects: ", len(set(vp_q)))
    print("Data used subjects: ", len(set(vp)))
    print("Rejected Subjects: ", len(set(vp_q))-len(set(vp)))
    print("Sentences per Subject: ", len(list(filter(lambda x: x == vp_q[0], vp))))
    relevant_questions = list(filter(lambda x: x[0] in vp, questions))
    age_questions = list(filter(lambda x: x[1] == "age", relevant_questions))
    sex_questions = list(filter(lambda x: x[1] == "Sex", relevant_questions))
    native_questions = list(filter(lambda x: x[1] == "EnglishLevel", relevant_questions))
    _ ,_ , age = zip(*age_questions)
    age = [int(a) for a in age]
    print("Mean age: ", np.mean(age))
    males = len(list(filter(lambda x: x[2] == "M", sex_questions)))
    females = len(list(filter(lambda x: x[2] == "F", sex_questions)))
    print(f"Females: {females}, Males: {males}")
    native_speaker = list(filter(lambda x: x[2] == "NS", native_questions))
    non_native_speaker = list(filter(lambda x: x[2] == "NNS", native_questions))
    print(f"Native speaker: {len(native_speaker)}, Non-native speaker: {len(non_native_speaker)}")
    eds = []
    for speaker_class in [native_speaker, non_native_speaker]:
        vp_code, _, s_class= zip(*speaker_class)
        important_evals = list(filter(lambda x: x[0] in vp_code, scored_data))
        _, _, mover, bleu, edit_distance, rough = zip(*important_evals)
        eds.append(edit_distance)
        print(f"{s_class[0]}: Mover Score = {np.mean(mover)}, Bleu Score = {np.mean(bleu)}, Edit distance = {np.mean(edit_distance)}, rouge = {np.mean(rough)}")
    print(f"Unequal variance t-test {ttest_ind(eds[0],eds[1],equal_var=False)}")

def _clean_data(sentence):
    sentence = sentence.lower()
    sentence = sentence.strip()
    if sentence[len(sentence)-1] == ".":
        sentence = sentence[:-1]
    return sentence


if __name__ == "__main__":
    print(_clean_data("   Halo Welt.   \n"))
    #create_evaluation_file("evaluation/human_data/trialdata.csv", "evaluation/human_data/eval_data.csv")
    _, evals = evaluate_data("human_data/eval_data.csv")
    evaluate_vp(evals)
