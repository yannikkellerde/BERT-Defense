# Some useful character constants
small_vocals = "aeiou"
big_vocals = small_vocals.upper()
vocals = small_vocals+big_vocals
small_consonants = "bcdfghjklmnpqrstvwxyz"
big_consonants = small_consonants.upper()
consonants = small_consonants+big_consonants
letters = vocals+consonants
sentence_ends = "?!;.\""
singular_punctuations = ",:&"+sentence_ends
non_singular_punctuations = "-' #/\\(){}()%$€@"
trenner_punctuations = "-'/&"
punctuations = non_singular_punctuations+singular_punctuations
all_chars = letters+punctuations