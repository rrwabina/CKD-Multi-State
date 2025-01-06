import re 


def get_pair_stats(word_freq):
    '''
    Count frequency of adjacent symbol pairs across all words.
    Args: 
        word_freq (dict): A dictionary of words and their frequencies.
    
    Returns: 
        dict: A dictionary of symbol pairs and their frequencies.
    '''
    pair_stats = {}
    for symbols, freq in word_freq.items():
        for i in range(len(symbols) - 1):   
            pair = (symbols[i], symbols[i+1])
            pair_stats[pair] = pair_stats.get(pair, 0) + freq
    return pair_stats 

def merge_pair_in_words(best_pair, word_freq):
    '''
    Merge the best_pair in every word in word_freq.
    If best_pair = ('a', 'n'), then in the word ('b', 'a', 'n', 'a', '</w>'),
        we merge occurrences of adjacent 'a' 'n' into ('an').

    Args:   best_pair (tuple): The pair of symbols to merge.
            word_freq (dict): A dictionary of words and their frequencies.
    Returns: dict: A dictionary of merged words and their frequencies.
    '''
    merged_word_freq = {}
    bigram = best_pair
    for symbols, freq in word_freq.items():
        new_symbols = []
        i = 0
        while i < len(symbols):
            if (i < len(symbols) - 1) and (symbols[i], symbols[i+1]) == bigram:
                new_symbols.append(symbols[i] + symbols[i+1])  # e.g. 'a'+'n' -> 'an'
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_symbols = tuple(new_symbols)
        merged_word_freq[new_symbols] = merged_word_freq.get(new_symbols, 0) + freq
    return merged_word_freq

def apply_bpe_to_word(word, merges):
    ''' 
    Applies the learned merges to a single word (tuple of chars) in sequential order.
        e.g., merges = [(('a','n'), 'an'), (('an','a'), 'ana'), ...].
        We keep merging until no merge can be applied or we've exhausted merges.

    Args:   word (tuple): The word to tokenize as a tuple of characters.
            merges (list): A list of merges learned from the training data.

    Returns: list: A list of subword units after applying BPE merges
    '''
    symbols = list(word)
    i = 0
    for (bigram, merged_symbol) in merges:
        i = 0
        while i < (len(symbols) - 1):
            if (symbols[i], symbols[i+1]) == bigram:
                symbols[i] = merged_symbol
                del symbols[i+1]  
            else:
                i += 1
    return symbols


def tokenize_bpe(text, num_merges = 5, special_end_token = '<END>'):
    ''' 
    Tokenize an input text using a more realistic (less naive) Byte-Pair Encoding (BPE).
    This function both learns and applies the BPE merges from the input text.

    Args:
        text (str): The input text (can be multiple sentences) to tokenize.
        num_merges (int): The maximum number of BPE merges to learn.
        special_end_token (str): A marker to indicate the end of a word.

    Returns:
        list: A list of BPE-tokenized words (subwords)
    '''
    words = re.findall(r'\S+', text)
    word_freq = {}
    
    for w in words:
        chars = tuple(list(w) + [special_end_token])
        word_freq[chars] = word_freq.get(chars, 0) + 1

    merges = []  
    for _ in range(num_merges):
        pair_stats = get_pair_stats(word_freq)
        if not pair_stats:
            break
        best_pair = max(pair_stats, key = pair_stats.get)
        best_pair_count = pair_stats[best_pair]

        if best_pair_count == 0:
            break
        merges.append((best_pair, ''.join(best_pair)))
        word_freq = merge_pair_in_words(best_pair, word_freq)
    tokenized_text = []
    for w in words:
        chars = tuple(list(w) + [special_end_token])
        subword_units = apply_bpe_to_word(chars, merges)
        if subword_units and subword_units[-1] == special_end_token:
            subword_units = subword_units[:-1]
        tokenized_text.append(' '.join(subword_units)) 
    return tokenized_text

sample_text = ''' 
                Note: This test is to gauge your knowledge in Python programming and NLP architecture and theory. 
                It is expected that no AI or Internet assistance will be used to answer questions in this test. 
                By completing this test, you swear that you have not used AI assistance and have provided your own, unaided, responses. 
                If discovered that you have used AI at any point during the recruitment process, you will not be considered for the position. 
                Give your honest opinion on the questions, wrong answers will not be disqualifying.
                You may use the internet, but not AI, for questions 4.
              '''
tokenized = tokenize_bpe(sample_text, num_merges = 10)
print(tokenized)