# BERT-Defense: A Probabilistic Model Based on BERT to Combat Cognitively Inspired Orthographic Adversarial Attacks

Code and data for our [paper](https://arxiv.org/pdf/2106.01452.pdf)


```
@inproceedings{keller:bert-defnse,
   title = "BERT-Defense: A Probabilistic Model Based on BERT to Combat Cognitively Inspired Orthographic Adversarial Attacks",
   author = "Yannik Keller and Jan Mackensen and Steffen Eger",
   year = "2021",
   booktitle = "Findings of the Association for Computational Linguistics: ACL 2021",
   publisher = "Association for Computational Linguistics",
   note="accepted"
}
```

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

## Notes

If you cite the Zéroe benchmark, please also cite this paper:

```
@inproceedings{eger-benz-2020-hero,
    title = "From Hero to Z{\'e}roe: A Benchmark of Low-Level Adversarial Attacks",
    author = "Eger, Steffen  and
      Benz, Yannik",
    booktitle = "Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing",
    month = dec,
    year = "2020",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.aacl-main.79",
    pages = "786--803",
    abstract = "Adversarial attacks are label-preserving modifications to inputs of machine learning classifiers designed to fool machines but not humans. Natural Language Processing (NLP) has mostly focused on high-level attack scenarios such as paraphrasing input texts. We argue that these are less realistic in typical application scenarios such as in social media, and instead focus on low-level attacks on the character-level. Guided by human cognitive abilities and human robustness, we propose the first large-scale catalogue and benchmark of low-level adversarial attacks, which we dub Z{\'e}roe, encompassing nine different attack modes including visual and phonetic adversaries. We show that RoBERTa, NLP{'}s current workhorse, fails on our attacks. Our dataset provides a benchmark for testing robustness of future more human-like NLP models.",
}
```
