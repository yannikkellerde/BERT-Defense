import os
for fname in os.listdir("attacked_documents"):
    if os.path.isfile(os.path.join("cleaned/nocheap_bayesian_shielding",fname)):
        print("skipping",fname)
        continue
    print("starting",fname)
    os.system(f"python clean_document.py {os.path.join('attacked_documents',fname)}")