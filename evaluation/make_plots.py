import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import numpy as np
import os,sys
import re

ev = pd.read_csv("evaluation.csv")

name_map = {"visual":"vi","phonetic":"ph","full-swap":"fs","inner-swap":"is","disemvowel":"dv","truncate":"tr",
            "keyboard-typo":"kt","natural-typo":"nt","intrude":"in","segmentation":"sg","rand":"rd"}

def set_xmargin(ax, left=0.0, right=0.0):
    # Source https://stackoverflow.com/questions/49382105/matplotlib-set-different-margins-for-left-and-right-side/49382894#49382894
    ax.set_xmargin(0)
    ax.autoscale_view()
    lim = ax.get_xlim()
    delta = np.diff(lim)
    left = lim[0] - delta*left
    right = lim[1] + delta*right
    ax.set_xlim(left,right)

def attack_map(attack_str):
    out = ""
    singled_attacks = re.findall(r"\([0-9a-zA-Z,'\-\. ]*\)", attack_str)
    for i,s in enumerate(singled_attacks):
        name = name_map[re.findall(r"'.*'",s)[0][1:-1]]
        number = re.findall(r"[0-9\.]+",s)[0]
        out+=f"{name}:{number}"
        if i!=len(singled_attacks)-1:
            out+=","
    return out

df = pd.read_csv("evaluation.csv")
df = df[~df["document"].str.contains("all_attacks")]
methods = []
attacks = []
for i,row in df.iterrows():
    doc = row["document"]
    if "nocheap_priors" in doc:
        method = "ours bp (only priors)"
    elif "nocheap_bayesian_shielding" in doc:
        method = "ours bp (full pipeline)"
    elif "/bayesian_shielding" in doc:
        method = "ours fp (full pipeline)"
    elif "/priors" in doc:
        method = "ours fp (only priors)"
    elif "attacked_documents" in doc:
        method = "no cleaning"
    elif "pyspellchecker" in doc:
        method = "pyspellchecker"
    elif "Adversarial_Misspellings" in doc:
        method = "danishpruthi" 
    elif "human" in doc:
        method = "human"
    if method == "human":
        attacks.append(doc.split("/")[1].replace(";",","))
    else:
        with open(os.path.join("attacked_documents",os.path.basename(doc.replace(":40",""))), "r", encoding="utf-8") as f:
            attacks.append(attack_map(f.readline()[2:]))
    if doc.endswith(":40"):
        method+=":40"
    methods.append(method)

df["method"] = methods
df["attacks"] = attacks

#at_uniq = list(pd.unique(df["attacks"]))
#at_uniq = ["vi:0.3","tr:0.3","dv:0.3","sg:0.3","is:0.3","fs:0.3","in:0.3","kt:0.3","nt:0.3","ph:0.1","ph:0.3","ph:0.5","ph:0.7","ph:0.9","sg:0.5,kt:0.3","vi:0.3,in:0.3","rd:0.3","rd:0.3,rd:0.3","rd:0.6,rd:0.6"]
#print(set(at_uniq_old)-set(at_uniq))
at_uniq = ["vi:0.3","nt:0.3","dv:0.3","fs:0.3","ph:0.3","ph:0.7","sg:0.5,kt:0.3","rd:0.3,rd:0.3","rd:0.6,rd:0.6"]
me_uniq = pd.unique(df["method"])
colormap = {"danishpruthi":"#0000ff","pyspellchecker":"#6600cc","no cleaning":"#808080","ours bp (only priors)":"#cc9900",
            "ours fp (only priors)":"#ffcc00","ours bp (full pipeline)":"#ff3300","ours fp (full pipeline)":"#800000",
            "human":"#006600"}
me_uniq = list(colormap.keys())
measures = ["sts-b","bleu","rouge-1","rouge-4","rouge-l","rouge-w","editdistance","mover","perfect_cleaned"]

def scatter_plots():
    for measure in measures:
        fig = plt.figure(figsize=(10,8))
        fig.subplots_adjust(bottom=0.2,right=0.75)
        for method in me_uniq:
            df_inner = df[df["method"]==method]
            stupid = []
            for at in at_uniq:
                uff = {}
                uff["attack"] = at
                pdval = df_inner.loc[df_inner["attacks"]==at][measure]
                if len(pdval)==0:
                    uff["value"] = None
                else:
                    uff["value"] = float(pdval)
                stupid.append(uff)
            df_inner = pd.DataFrame(stupid)
            npval = df_inner["value"].to_numpy()
            avg = np.mean(npval[~np.isnan(npval)])
            plt.plot([len(at_uniq),len(at_uniq)-0.3], [avg, avg], 'k-', color=colormap[method])
            plt.scatter(range(len(df_inner)),df_inner["value"],label=method,color=colormap[method])
            plt.xticks(range(len(df_inner)),at_uniq,rotation=90)
        plt.title(measure)
        plt.legend(title="Methods",bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        set_xmargin(fig.axes[0],left=0.03,right=0.0)
        plt.savefig(f"plots/scatter/{measure}.svg")
        plt.cla()

def ablation_plot():
    docs_with_names = {
        "attacked_documents/rand2.txt":"no shielding",
        "cleaned/no_lev_bayesian_shielding/rand2.txt":"No lev.",
        "cleaned/no_bert_bayesian_shielding/rand2.txt":"No BERT",
        "cleaned/no_gpt_bayesian_shielding/rand2.txt":"No GPT",
        "cleaned/nocheap_bayesian_shielding/rand2.txt":"attack-agnostic",
        "cleaned/bayesian_shielding/rand2.txt":"domain-specific"
    }
    colormap = ("#0000ff","#6600cc","#808080","#cc9900","#ffcc00","#ff3300")
    width = 0.8
    move = np.array([-0.35+width/2+width*i for i in range(len(docs_with_names))])
    plt.xticks(move,docs_with_names.values(),rotation=90)
    scores = []
    for doc,name in docs_with_names.items():
        scores.append(float(df[df["document"]==doc]["mover"]))
    print(move,scores)
    plt.bar(np.array(move),np.array(scores),color=colormap,width=0.5)
    plt.tight_layout()
    plt.title("Mover Score")
    plt.show()

def bar_plots(methods,do_40=False):
    home_path = "plots/bar/"+"_".join(methods)
    os.makedirs(home_path, exist_ok=True)
    width = 0.8/len(methods)
    move = [-0.35+width/2+width*i for i in range(len(methods))]
    for measure in measures:
        fig = plt.figure(figsize=(10,2.5))
        fig.subplots_adjust(bottom=0.2,right=0.75)
        for i,method in enumerate(methods):
            df_inner = df[df["method"]==method+(":40" if do_40 and not "human" in method else "")]
            stupid = []
            for at in at_uniq:
                uff = {}
                uff["attack"] = at
                pdval = df_inner.loc[df_inner["attacks"]==at][measure]
                if len(pdval)==0:
                    uff["value"] = None
                else:
                    uff["value"] = float(pdval)
                stupid.append(uff)
            df_inner = pd.DataFrame(stupid)
            npval = df_inner["value"].to_numpy(dtype="float")
            avg = np.mean(npval[~np.isnan(npval)])
            if measure == "perfect_cleaned":
                df_inner["value"]*=100
            plt.bar(np.array(range(len(df_inner)))+move[i],df_inner["value"],color=colormap[method],label=method,width=width)
            plt.locator_params(axis='y', nbins=4)
            plt.xticks(range(len(df_inner)),at_uniq,rotation=90)
        plt.title(measure)
        plt.legend(title="Methods",bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        set_xmargin(fig.axes[0],left=0.03,right=0.0)
        plt.savefig(os.path.join(home_path,f"{measure}.svg"))
        plt.cla()
#scatter_plots()
#line_plots(methods=["ours bp (full pipeline)","ours fp (full pipeline)","danishpruthi","pyspellchecker"])
#bar_plots(methods=["ours bp (full pipeline)","ours fp (full pipeline)","danishpruthi","pyspellchecker"],do_40=False)
#bar_plots(methods=["ours bp (full pipeline)","ours fp (full pipeline)","human"],do_40=True)
#bar_plots(methods=["ours bp (full pipeline)","ours fp (full pipeline)","danishpruthi","pyspellchecker"])
ablation_plot()