


def loadVocab(vocab_file):
    vocab = {}
    vocabFile = open(vocab_file, "r")
    for line in vocabFile:
        # Clean the line
        line = line.strip()
        
        # Get the key and value
        line = line.split(',')
        
        # Add to the vocab
        vocab[int(line[0])] = line[1]
        
    return vocab