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
        + For each U in **M**: Store T in *C\[U\]*
    * Puzzle together S from combinations in *C*, so that the start and end indices match up the full source word S. Assume lowest distance from the target words of each combinations key in *C*. Increase costs for using many word parts. Keep N lowest cost hypothesis in list **H**.
    * Keep the entries of *C* at the selected combinations for each hypothesis as dictionaries for later use. (The lowest distance word part will not necessarily be the final result after using BERT)
    * Insert **D** into **H**.
3. Use the cross product of all word hypothesis to form sentence hypothesis. Only keep the N sentence Hypothesis with the lowest distances in a list **S**.
4. Use softmax to convert sentence Hypothesis distances to probabilities. Use softmax to convert word part distances to probabilities for each word part in each word in each sentence hypothesis.
5. Return **S**, consisting of sentence hypothesis with probabilities associated, that in each consist of words out of word parts. Each word part has an own dictionary as well as a probability vector associated with it.
6. For each Hypothesis **H** in **S**:
    * Select most uncertain word part U in **H** based on difference of highest 2 probabilities.
    * Perform weighted average of the word embeddings top N words (based on probability) for each other word part in **H**.
    * Mask U and put it, toghether with the avg. word embeddings into bert, to obtain a probability distribution P over words for the masked word part U.
    * Only select the words in P that are in the dictionary associated with U and then use softmax to call it a likelihood L.
    * Obtain Posterior *P* from prior probabilitys for U and L.
    * Repeat until each word has been covered once (or even for more iterations).
7. Select the Maximum-a-posteriori (MAP) for each sentence hypothesis and call their associated probability the sentence prior P
8. Use OpenAI's GTP on each Sentence to calculate a sentence likelihood L.
9. Calculate Sentence Posterior *P* for the hypothesis from L and P.
10. Select Maximum-a-posteriori to get a final cleaned up sentence.