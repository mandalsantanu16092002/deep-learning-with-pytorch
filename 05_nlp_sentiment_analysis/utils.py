
import re
from collections import Counter

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.split()

def build_vocab(sentences, min_freq=1):
    counter = Counter()
    for sentence in sentences:
        counter.update(tokenize(sentence))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab

def encode_sentence(sentence, vocab, max_len):
    tokens = tokenize(sentence)
    encoded = [vocab.get(word, vocab["<UNK>"]) for word in tokens]

    if len(encoded) < max_len:
        encoded += [vocab["<PAD>"]] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]

    return encoded
