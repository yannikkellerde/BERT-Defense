import pandas as pd
import sys,os
from easy_table import EasyTable
import re

class MyEasyTable(EasyTable):
    def displayTable(self):
        # Get the max length
        self.getMaxLengthPerColumn()
        # Print the reader
        self.printTableHeader()
        # Print the rows
        for row_num, row in enumerate(self.table_data):
            self.printDataRow(row)
        self.printTableFooter()

name_map = {"visual":"vi","phonetic":"ph","full-swap":"fs","inner-swap":"is","disemvowel":"dv","truncate":"tr",
            "keyboard-typo":"kt","natural-typo":"nt","intrude":"in","segmentation":"sg","rand":"rd"}

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
    with open(os.path.join("attacked_documents",os.path.basename(doc)), "r", encoding="utf-8") as f:
        attacks.append(attack_map(f.readline()[2:]))
    if "bayesian_shielding" in doc:
        method = "ours"
    elif "priors" in doc:
        method = "ours (only priors)"
    elif "attacked_documents" in doc:
        method = "no cleaning"
    elif "pyspellchecker" in doc:
        method = "pyspellchecker"
    elif "Adversarial_Misspellings" in doc:
        method = "Adversarial_Misspellings" 
    methods.append(method)

df["method"] = methods
df["attacks"] = attacks

evals = ["mover","sts-b","bleu","rouge-1"]

table_data = {x:[] for x in evals}

for method in df["method"].unique():
    for key in evals:
        table_data[key].append({"method":method})
    only_method = df[df["method"]==method]
    for i,row in only_method.iterrows():
        for ev in evals:
            table_data[ev][-1][row["attacks"]] = str(round(row[ev],3))

tables = {}
for ev in evals:
    table = MyEasyTable(ev)
    table.setOuterStructure("|", "")
    table.setInnerStructure("|", "-", "|")
    table.setData(table_data[ev])
    tables[ev] = table

original_stdout = sys.stdout
with open("tables.txt", "w") as f:
    sys.stdout = f
    name_table = MyEasyTable("names")
    name_table.setData([name_map])
    name_table.setOuterStructure("|", "")
    name_table.setInnerStructure("|", "-", "|")
    print("## Abbrevation map")
    name_table.displayTable()

    for name,table in tables.items():
        print("\n##",name)
        table.displayTable()
with open("tables.txt","r") as f:
    text = f.read().replace("\n ","\n")
with open("tables.txt","w") as f:
    f.write(text)
sys.stdout = original_stdout
print(text)