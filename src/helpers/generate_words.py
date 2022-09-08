from .helpers import get_clean_words
import random


# Generate a vocab file csv with two columns, a key
# and a value
def generate():
    vocabFile = "data/Text2/data.txt"
    outFile = "vocab_text2.csv"
    limit = 68600
    randomize = True # Randomize the vocab to stray from bias
    
    
    # The created vocab as each word is seen
    vocab = {"<START>": 0, "<PAD>": 1, "<END>": 2, "<NEXT>": 3}
    
    i = len(vocab)
    sents = 0
    
    # Iterate over all lines in the file
    file = open(vocabFile, "r", encoding="utf-8")
    for line in file:
        sents += 1
        
        # Get the words
        words = get_clean_words(line)
        
        # Breakup the line into spaces and store any new words
        for word in words:
            # Make the word lowercased
            word = word.lower()
            
            # If the word is blank, don't add it
            if len(word) == 0:
                continue
            
            # If the word is not already in the dictionary,
            # add it only
            if word not in vocab:
                vocab[word] = i
                i += 1
            
            # Check if the limit has been reached
            if len(vocab.keys()) > limit:
                break
        
        # Check if the limit has been reached
        if len(vocab.keys()) > limit:
            break
    file.close()

    # Shuffle the vocab
    if randomize:
        k = list(vocab.keys())
        random.shuffle(k)
        vocab = {k[i]:i for i in range(0, len(k))}

    
    # Save the vocab as a csv
    file = open(outFile, "w", encoding="utf-8")
    for key in vocab.keys():
        file.write(f"{vocab[key]},{key}\n")
        
    # Close the file
    print(f"Number of sentences loaded: {sents}")
    file.close()



generate()