## Substring Levenshtein Distance
Inspired by [http://ginstrom.com/scribbles/2007/12/01/fuzzy-substring-matching-with-levenshtein-distance-in-python/](http://ginstrom.com/scribbles/2007/12/01/fuzzy-substring-matching-with-levenshtein-distance-in-python/). The Idea is to perform fuzzy substring matching to get back source word indicies between which the target word fits best. Multiple of those can then be used to puzzle together long source strings without word segmentations.

* len source word := n, len target word := m  
* Initialize distance matrix D of size (m+1)x(n+1)
* **Fill top row with zeros instead of with increasing values**
* Fill first column with increasing values based on insertion cost.
* **Initialize startmatrix S of size (m+1)x(n+1) that keeps track of from which startpoints each point in the distance matrix can be reached in shortest distance**
* **Initialize the entrys of the top row of S with sets, that only contain their respective column index. Every other entry will be initalized with the empty set.**
* Iterate n and m.
    + For each entry, calculate the cost of getting to that entry by inserting, deleting, swapping, substituting or if possible by doing nothing (If source and target word match up at that point).
    + Enter the lowest cost into the distance Matrix
    + **Update S, by updating the set for the current entry with the sets from the entrys that led to this entry with lowest cost**
* **Check the bottom row of D for a lowest cost C. For all bottom row entrys of D equal to C: Store combinations of respective column index together with the set entrys of S at that location in 2-tuples into a list L**
* **Return C,L**

- Insertion cost: reduced if vowl and no vowls in word
- Sub_cost: Reduced for high cosine similarity between swapped chars (visual embeddings)
- Del_cost: Reduced for frequent characters in source word
- Swap_cost: 1

## Dictionarys
* Full-word dict: Bert's default Word piece dictionary, but only the tokens that are real english words on their own.
* Morpheme dict: Morphemes from Bert's Word piece dictionary. They are marked for BERT with '##' in the beginning.

## Sentence cleanup pipeline
1. "Soft" tokenization (Tokenize, but don't split up /,$, or sth.)
2. For each source word S in the Sentence:
    * Create combination dictionary *C*
    * Create single-word distance Matrix **D**
    * For each target word T from Full-word dict + Morpheme dict:
        + Calculate Substring Levenshtein distance to obtain distance D and combinations **M**.
        + If T in Full-word dict, store D in **D**, while accounting for fillup costs for substring matches.
        + For each U in **M**: Store T,D in *C\[U\]*
    * Puzzle together S from combinations in *C*, so that the start and end indices match up the full source word S. Assume lowest distance from the target words of each combinations key in *C*. Increase costs for using many word parts. Keep N lowest cost hypothesis in list **H**.
    * Keep the entries of *C* at the selected combinations for each hypothesis as dictionaries for later use. (The lowest distance word part will not necessarily be the final result after using BERT)
    * Insert **D** into **H**.
3. Use the cross product of all word hypothesis to form sentence hypothesis. Only keep the N sentence Hypothesis with the lowest distances in a list **S**.
4. Use softmax to convert sentence Hypothesis distances to probabilities. Use softmax to convert word part distances to probabilities for each word part in each word in each sentence hypothesis.
5. Return **S**, consisting of sentence hypothesis with probabilities associated, that in each consist of words out of word parts. Each word part has an own dictionary as well as a probability vector associated with it.
6. For each Hypothesis **H** in **S**:
    * Select most uncertain word part U in **H** based on difference of highest 2 probabilities.
    * Perform weighted average of the word embeddings top N words (based on probability) for each other word part in **H**.
    * Mask U and put it, [('keyboard-typo',0.2),('intrude',0.5),('segmentation',0.6)][('keyboard-typo',0.2),('intrude',0.5),('segmentation',0.6)]oghether with the avg. word embeddings into bert, to obtain a probability distribution P over words for the masked word part U.
    * Only select the words in P that are in the dictionary associated with U and then use softmax to call it a likelihood L.
    * Obtain Posterior *P* from prior probabilitys for U and L.
    * Repeat until each word has been covered once (or even for more iterations).
7. Select the Maximum-a-posteriori (MAP) for each word-part in each sentence hypothesis
8. Call the probabilities associated with each sentence hypothesis prior P
9. Use OpenAI's GTP on each Sentence to calculate a sentence likelihood L.
10. Calculate Sentence Posterior *P* for the hypothesis from L and P.
11. Select Maximum-a-posteriori to get a final cleaned up sentence.
12. Puzzle segmentations back together using self implemented detokenization.

## Cleanup example:
The sentence `A sma|ll w#hiteca:t witg,l,o,win,g e[h[e[sstajdinginderneath ac*h*a*ir.` has been attacked with 3 adversarial attacks. The attack parameters where `[('keyboard-typo',0.2),('intrude',0.5),('segmentation',0.6)]`.
After tokenizing the sentence, it looks like this: `['a', 'sma|ll', 'w#hiteca:t', 'witg,l,o,win,g', 'e[h[e[sstajdinginderneath', 'ac*h*a*ir', '.']`
When we now calculate the substring levenshtein distances for each word, we can form the following prior sentence hypothesis:  
`Hypothesis 1, Probability: 0.5591098924966913, Prior:a small white cat wit glowing eyes standing underneath chair.`
`Hypothesis 2, Probability: 0.4300367473288713, Prior:a small white cat wit glowing eyes standing underneath a chair.`  
Note, that here, only the most probable words according to the prior are shown. Actually, each word index has a whole dictionary with probabilities associated with it. We see, that we have 2 hypothesis, one that the `a` in front of chair does not belong there and the actual word is only `chair` (it could be a typo or something) and the other one is that `a` is it's own word that belongs in front of `chair`.
Let's now put our hypothesis through the BERT posterior and see how that changes our sentences.
`Hypothesis 1, Posterior: a small white cat with glowing eyes standing underneath chair.`
`Hypothesis 2, Posterior: a small white cat with glowing eyes standing underneath a chair.`  
In both cases, BERT fixed the `wit-with` error, as `wit` would not make any sense gramatically.  
To figure our which of the 2 Hypothesis we should finally choose, let's put the sentences into GTP to calculate sentence Likelihoods.  
`Hypothesis 1, Prior prob: 0.5591098924966913, Sentence: a small white cat with glowing eyes standing underneath chair. GTP Likelihood: 0.31197451, Posterior prob: 0.3708833644581104`
`Hypothesis 2, Prior prob: 0.4300367473288713, Sentence: a small white cat with glowing eyes standing underneath a chair. GTP Likelihood: 0.68802549, Posterior prob: 0.6291166355418897`
GTP decided, that the sentence ending with `a chair` is much more likely than without the `a`. After combining likelihood with prior to Posterior, we only have to choose the MAP now, to get our final cleaned output sentence: `a small white cat with glowing eyes standing underneath a chair.`.