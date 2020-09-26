import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../DATA/hyperparam_tune/tune_info.csv")

params = ["top_n","bert_theta","gtp_theta"]
evals = ["bleu","mover","rouge-1","rouge-4","rouge-l","rouge-w"]

for param in params:
    df = df.sort_values(param)
    for e in evals:
        plt.plot(df[param],df[e],label=e)
    plt.xlabel(param)
    plt.ylabel("score")
    plt.legend()
    plt.show()