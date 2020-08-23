def load_freq_dict():
    freq_dict = {}
    with open("../../DATA/dictionaries/word_frequencies.txt","r") as f:
        lines = f.read().splitlines()
        for i,line in enumerate(lines):
            parts = line.split("\t")
            freq_dict[parts[0].lower()] = int(len(lines)-i)
    return freq_dict

def get_prior(dictionary, theta = 0.00001, freq_dict=None, not_in_val=1):
    if freq_dict is None:
        freq_dict = load_freq_dict()
    prior = []
    for word in dictionary:
        if word in freq_dict:
            prior.append(freq_dict[word])
        else:
            prior.append(not_in_val)
    return softmax(np.array(prior),theta)