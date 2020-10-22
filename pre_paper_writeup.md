## Visual similarity
To asses the similarity of the first 30000 Unicode glyphs with english letters and numbers, we create a 30000x36 similarity matrix. Each glyph is drawn via PIL in 20pt using a fitting font from the [Noto](https://www.google.com/get/noto/) font collection. The bitmap is then cropped to contain only the glyph. Then the image is resized and padded on the right and bottom to be of size (30px, 30px). When comparing the bitmap of a unicode glyph image and a letter/number glyph, multiple versions of the letter/number bitmap are created. For letters, the small as well as the large versions of each letter are taken. The bitmap get's downsized to 5 different sizes ((30px, 30px) to (15px, 15px)), rotated and flipped in all 8 unique ways and then padded to (30px, 30px) again, such that the glyph is either placed at the top-left or the bottom left. The percentage of matching black pixels between bitmaps is converted into a similarity score and the highest similarity of all versions is chosen as a final result.

## New Phonetic attacks
Use [Many to many aligner](https://github.com/letter-to-phoneme/m2m-aligner) to align letters and phones of [cmudict](https://github.com/cmusphinx/cmudict). Calculate statistics from aligned cmudict: To what letters are phones most likely to correspond to? Weight this statistic via a word frequency. To attack a word, transcribe it to it's phonetic representation via cmudict and then use the statistics to transform the phonetic representation back to letters. For higher attack levels, choose statistically less likely letters. Cherry picked example: `Heavy rain raises threat of Christmas Day flooding => Heavie rayn razes threit of Cristmise Day flooding`. [List of not cherry picked examples](adversarial_attacks/phonetic_attacks/examples/lil_random_with_freq.txt).

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
- Sub_cost: Reduced for high similarity values
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
        + Calculate Anagram matching distance A, by evaluating letter frequencies by comparing character frequencies in source and target word.
        + If T in Full-word dict, store min(A,D) in **D**, while accounting for fillup costs for substring matches.
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
    * Mask U and put it, together with the avg. word embeddings into bert, to obtain a probability distribution P over words for the masked word part U.
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

## Evaluation
## Abbrevation map
                                                                                                            
visual|phonetic|full-swap|inner-swap|disemvowel|truncate|keyboard-typo|natural-typo|intrude|segmentation|rand 
------|--------|---------|----------|----------|--------|-------------|------------|-------|------------|---- 
  vi  |   ph   |   fs    |    is    |    dv    |   tr   |     kt      |     nt     |  in   |     sg     | rd  
                                                                                                            

## mover
                                                                                                            
      method      |dv:0.3|ph:0.3|kt:0.2,nt:0.2,tr:0.2|fs:0.3|is:0.3|sg:0.5,kt:0.3|vi:0.3,in:0.3|rd:0.3,rd:0.3 
------------------|------|------|--------------------|------|------|-------------|-------------|------------- 
       ours       |0.795 |0.588 |       0.577        |0.822 |0.831 |    0.687    |    0.816    |    0.643     
ours (only priors)|0.657 |0.419 |       0.307        |0.689 |0.843 |    0.538    |    0.68     |    0.451     
   no cleaning    |0.317 |0.265 |        0.19        |0.291 |0.307 |    0.154    |    0.012    |    0.162     
                                                                                                            

## sts-b
                                                                                                            
      method      |dv:0.3|ph:0.3|kt:0.2,nt:0.2,tr:0.2|fs:0.3|is:0.3|sg:0.5,kt:0.3|vi:0.3,in:0.3|rd:0.3,rd:0.3 
------------------|------|------|--------------------|------|------|-------------|-------------|------------- 
       ours       |0.776 |0.654 |       0.587        |0.802 |0.818 |    0.63     |    0.799    |    0.654     
ours (only priors)|0.788 |0.556 |        0.52        |0.744 |0.872 |    0.698    |    0.715    |    0.53      
   no cleaning    |0.473 |0.588 |       0.581        |0.402 |0.574 |    0.573    |    0.33     |    0.322     
                                                                                                            

## bleu
                                                                                                            
      method      |dv:0.3|ph:0.3|kt:0.2,nt:0.2,tr:0.2|fs:0.3|is:0.3|sg:0.5,kt:0.3|vi:0.3,in:0.3|rd:0.3,rd:0.3 
------------------|------|------|--------------------|------|------|-------------|-------------|------------- 
       ours       |0.515 |0.339 |       0.293        | 0.54 | 0.55 |    0.398    |    0.531    |    0.365     
ours (only priors)|0.392 |0.173 |       0.044        |0.434 |0.568 |    0.238    |    0.384    |    0.179     
   no cleaning    |0.149 |0.098 |       0.016        |0.155 |0.154 |    0.008    |    0.002    |    0.016     
                                                                                                            

## rouge-1
                                                                                                            
      method      |dv:0.3|ph:0.3|kt:0.2,nt:0.2,tr:0.2|fs:0.3|is:0.3|sg:0.5,kt:0.3|vi:0.3,in:0.3|rd:0.3,rd:0.3 
------------------|------|------|--------------------|------|------|-------------|-------------|------------- 
       ours       | 0.9  |0.748 |       0.733        |0.902 |0.913 |    0.843    |    0.91     |    0.781     
ours (only priors)|0.829 |0.648 |       0.538        |0.846 |0.928 |    0.761    |    0.835    |    0.675     
   no cleaning    |0.603 |0.514 |       0.404        | 0.6  |0.623 |    0.322    |    0.144    |    0.357     

## Comparison to other work
We compared to the following other work [https://github.com/danishpruthi/Adversarial-Misspellings](https://github.com/danishpruthi/Adversarial-Misspellings): This implementation is only targeted at defending swaps, additions, drops and keyboard typos, but even for this kind of errors it only barely works. For example, their demonstration example is `nicset atcing I have ever witsesed` => `nicest acting i have ever witnessed`, but already for something slightly different, generated by our natural-typo adversarial attack (strength: 0.2) `nicest acting i have evere withnessed` it fails and produces `nicest acting i have every witnessed`. For complicated stuff like `A sma|ll w#hiteca:t witg,l,o,win,g e[h[e[sstajdinginderneath ac*h*a*ir.` for which to be fair it was built for, it does not work at all: `a small w#hiteca:t witg,l,o,win,g e[h[e[sstajdinginderneath ac*h*a*ir.`.

## Human Similarty clues (How humans handle adversarial attackes)
First of all, you have to distinguish between two basic ways of dealing with words. 
1.  Word recognition: Instantaneous automatic cognitive process to recognize a word, for very familiar words (often function words). Cannot be used for     adversarial attacked words.
2. Word identification: needs a specific strategy to recognize the word, because it is an unknown or rarely seen word. The literature discusses this case almost only in the context of children learning to read. Nevertheless, a brief summary of the basic strategies:
    * context clues (implemented with Bert)
    * word order and grammer (implemented with Bert → could perhaps use grammer better)
    * word parts → Splitting compound words into individual parts
    * morphemic analysis → Break word into morphemes to analyze meaning

### Typoglycemia

Typoglycemia is a neologism that describes the effect that a text is still perfectly readable, even though the letters in the words have been swapped, as long as the first and last letter remain the same. According to an Internet meme, this is said to have been scientifically confirmed by the University of Cambridge. However, this is not the case. Nevertheless, Matt Davis from the University of Cambridge has written an interesting article about this. (https://www.mrc-cbu.cam.ac.uk/people/matt.davis/cmabridge/). Important points from the article for our paper:

* text reading speed decrase by 11% when internal letters are reorderd, so the order of letters is important
* transposition of adjacent letters are easy to read (we model this with your levenstein distance)
* transpostion are easy to read if you do not change the sound of a word
* Context is very important this is given by function words that do not change e.g. the/and
* Shape of a word plays a big role in recognizing words, hence the idea that only the first and last letter would play a role. Nevertheless these do not determine alone the shape of a word.

The meme: Aoccdrnig to a rscheearch at Cmabrigde Uinervtisy, it deosn't mttaer in waht oredr the ltteers in a wrod are, the olny iprmoetnt tihng is taht the frist and lsat ltteer be at the rghit pclae. The rset can be a toatl mses and you can sitll raed it wouthit porbelm. Tihs is bcuseae the huamn mnid deos not raed ervey lteter by istlef, but the wrod as a wlohe.