import os
import torch
print("cuda is avaliable" if torch.cuda.is_available() else "CUDA NOT AVALIABLE, USING CPU")
for fname in os.listdir("attacked_mnli"):
    if not os.path.isfile(os.path.join("cleaned_mnli/nocheap_bayesian_shielding",fname)):
        print("starting mnli basic prior",fname)
        os.system(f"python clean_document.py {os.path.join('attacked_mnli',fname)} false")
    if not os.path.isfile(os.path.join("cleaned_mnli/bayesian_shielding",fname)):
        print("starting mnli full prior",fname)
        os.system(f"python clean_document.py {os.path.join('attacked_mnli',fname)} true")
for fname in os.listdir("attacked_documents"):
    if not os.path.isfile(os.path.join("cleaned/nocheap_bayesian_shielding",fname)):
        print("starting basic prior",fname)
        os.system(f"python clean_document.py {os.path.join('attacked_documents',fname)} false")
    if not os.path.isfile(os.path.join("cleaned/bayesian_shielding",fname)):
        print("starting full prior",fname)
        os.system(f"python clean_document.py {os.path.join('attacked_documents',fname)} true")