# BERT-Defense: A Probabilistic Model Based on BERT toCombat Cognitively Inspired Orthographic Adversarial Attacks

## Requirements
There is an environment.yml provided.  
To summarize, the main packages are:
+ python >= 3.6
+ numpy
+ torch==1.5.0
+ nltk
+ tqdm
+ pillow
+ pytorch_pretrained_bert
+ pandas

## Usage
You can use BERT-Defense to clean a single sentence using the script at `bayesian_shielding/frontend/clean_sentences.py`. Run for example `python clean_sentences.py "Egyp's Muslim BrotherhooRefuse toBackDown"` to clean the sentence and obtain the output *egypt muslim brotherhood refuse to back down*.

If you want to clean multiple sentences, you should probably make use of batched processing and use the `evaluation/clean_document.py` script. However, currently, you will need to format your document of sentences in a format similar to the files in `evaluation/attacked_documents`.