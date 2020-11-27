import pandas as pd

df = pd.read_csv("mnli/dev_matched.tsv",sep="\t")
df = df[["gold_label","sentence1","sentence2"]]
df = df.iloc[:200]
df.to_csv("mnli_dataset.csv",sep="\t",index=False,header=False)