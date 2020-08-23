def combine_known_transpos(dataset,combo_transpo,normal_transpo):
    out = [[[] for _sentence in line] for line in dataset]
    for i,line in enumerate(dataset):
        for j,sentence in enumerate(line):
            for word in sentence:
                word=mylower(word)
                print(word in combo_transpo, word in normal_transpo)
                if len(word) > 20:
                    out[i][j].append(combo_transpo[word])
                else:
                    out[i][j].append([normal_transpo[word],combo_transpo[word][1]])
    return out

def eval_cosine_similarity(embedding):
    similarity_scores = []
    sim_mean = []
    sim_max = []
    sim_min = []
    size = []
    bar = progressbar.ProgressBar(max_value=len(embedding)**2)
    i = 0
    j = 0
    for key1 in embedding:
        for key2 in embedding:
            similarity_scores.append(cosine_similarity(embedding[key1], embedding[key2]))
            j +=1
            i +=1
            if j == 100000:
                similarity_scores = np.array(similarity_scores)
                sim_max.append(np.max(similarity_scores))
                sim_mean.append(np.mean(similarity_scores))
                sim_min.append(np.min(similarity_scores))
                size.append(j)
                j = 0
                similarity_scores = []
            bar.update(i)
    if j !=0:
        similarity_scores = np.array(similarity_scores)
        sim_max.append(np.max(similarity_scores))
        sim_mean.append(np.mean(similarity_scores))
        sim_min.append(np.min(similarity_scores))
        size.append(j)
    sim_mean = calc_mean(sim_mean, size)
    sim_max = np.max(sim_max)
    sim_min = np.min(sim_min)
    with open("sim_eval.txt", "w") as f:
        f.write(str(sim_mean) + "\n")
        f.write(str(sim_max) + "\n")
        f.write(str(sim_min) + "\n")

def create_pre_mapping(distribution,orig_words,dic):
    pre_map = {}
    for i,(p,c) in enumerate(distribution):
        pmaxin = np.argmax(p)
        if len(c) == 0:
            if len(dic[pmaxin])>3:
                blub = np.zeros_like(p)
                blub[pmaxin] = 1
                pre_map[orig_words[i]] = [blub,tuple()]
        else:
            csum = sum(x[1] for x in c)
            if p[pmaxin]/(1+csum)>c[0][1]:
                if len(dic[pmaxin])>3:
                    blub = np.zeros_like(p)
                    blub[pmaxin] = 1
                    pre_map[orig_words[i]] = [blub,tuple()]
            else:
                pre_map[orig_words[i]] = [np.zeros_like(p),((c[0][0],1),)]
    return pre_map

def get_word_vec_from_distribution(word_distribution, dic, embeddings):
    word_vecs = []
    weights = []
    for i in range(len(word_distribution)):
        if word_distribution[i] !=0:
            word_vecs.append(embeddings[dic[i]])
            weights.append(word_distribution[i])
    word_vecs = np.array(word_vecs)
    average_word = np.average(word_vecs, axis=0, weights=weights)
    return average_word


def sentence_average_from_word_embeddings(posterior, dic, embeddings):
    sentence_embedding =[]
    word_vecs = []
    for word_distribution in posterior:
        word_vec = get_word_vec_from_distribution(word_distribution, dic, embeddings)
        word_vecs.append(word_vec)
    sentence_embedding.append(np.average(np.array(word_vecs), axis=0))
    return sentence_embedding

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data