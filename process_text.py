from string import punctuation
from bs4 import BeautifulSoup
from collections import Counter
import numpy as np
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 
import re

def remove_puchuations(text):
    return ''.join([c for c in text if c not in punctuation])

def preprocess_text(text):
    
    #import pdb;pdb.set_trace()
    # lower case
    text = text.lower()
    
    # Remove all non-word characters (everything except numbers and letters)
    text = re.sub(r"[^\w\s]", '', text)
    # Replace all runs of whitespaces with no space
    text = re.sub(r"\s+", ' ', text)
    # replace digits with no space
    text = re.sub(r"\d", '', text)
    

    # remove html tags
    text = BeautifulSoup(text, "lxml").text
    
    # remove puchuations
    text = remove_puchuations(text)
    
    # remove stopwords
    text = " ".join([w for w in text.split(' ') if w not in stop_words])
    
    return text
    
def get_vocab_to_int(data):
    
    text = ' '.join(data)
    words = text.split()
    count_words = Counter(words)

    total_words = len(words)
    sorted_words = count_words.most_common(total_words)
    
    vocab_to_int = {w:i for i, (w,c) in enumerate(sorted_words)}
    
    # to avoid padding conflict of the word with index 0
    vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
    
    return vocab_to_int
    
    
    
def encode_sent(sent,vocab_to_int):
     return [vocab_to_int[w] for w in sent.split()]

def pad_features(l_sent, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    review_len = len(l_sent)
    if review_len <= seq_length:
        zeroes = list(np.zeros(seq_length-review_len, dtype=int))
        new = zeroes+l_sent
    elif review_len > seq_length:
        new = l_sent[0:seq_length]
    return new