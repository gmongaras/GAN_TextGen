from helpers import get_clean_words




# Generate a vocab file csv with two columns, a key
# and a value
def generate():
    vocabFile = "data/Text/data.txt"
    outFile = "vocab_text.csv"
    limit = 1000000000000
    
    
    # The created vocab as each word is seen
    vocab = {"<START>": 0, "<PAD>": 1, "<END>": 2, "<UNKNOWN>": 3}
    
    i = 0
    
    # Iterate over all lines in the file
    file = open(vocabFile, "r")
    for line in file:
        
        # Get the words
        words = get_clean_words(line)
        
        # Breakup the line into spaces and store any new words
        for word in words:
            # If the word is blank, don't add it
            if len(word) == 0:
                continue
            
            # If the word is not already in the dictionary,
            # add it
            if word not in vocab:
                vocab[word.lower()] = i
                i += 1
            
            # Check if the limit has been reached
            if len(vocab.keys()) > limit:
                break
        
        # Check if the limit has been reached
        if len(vocab.keys()) > limit:
            break
    file.close()
    
    # Save the vocab as a csv
    file = open(outFile, "w")
    for key in vocab.keys():
        file.write(f"{vocab[key]},{key}\n")
        
    # Close the file
    file.close()



generate()