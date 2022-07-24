from .helpers import get_clean_words


# Generate a vocab file csv with two columns, a key
# and a value
def generate():
    vocabFile = "data/Fortunes/data.txt"
    outFile = "vocab_fortunes.csv"
    limit = 1000000000000
    
    
    # The created vocab as each word is seen
    vocab = {"<START>": 0, "<PAD>": 1, "<END>": 2, "<UNKNOWN>": 3}
    
    i = len(vocab)
    
    # Iterate over all lines in the file
    file = open(vocabFile, "r", encoding="utf-8")
    for line in file:
        
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
            # add it
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
    
    # Save the vocab as a csv
    file = open(outFile, "w", encoding="utf-8")
    for key in vocab.keys():
        file.write(f"{vocab[key]},{key}\n")
        
    # Close the file
    file.close()



generate()