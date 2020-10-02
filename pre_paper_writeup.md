## Manipulated Levenshtein Distance
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

* Insertion cost: reduced if vowl and no vowls in word
* Sub_cost: Reduced for high cosine similarity between swapped chars (visual embeddings)
* Del_cost: Reduced for frequent characters in source word
* Swap_cost: 1


## Sentence cleanup pipeline
1. "Soft" tokenization (Tokenize, but don't split up /,$, or sth.)
2. Use manipulated Levenshtein distance to 