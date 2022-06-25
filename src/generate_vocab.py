from helpers import get_clean_words




# Generate a vocab file csv with two columns, a key
# and a value
def generate():
    vocabFile = "data/data.txt"
    outFile = "vocab.csv"
    limit = 10000000
    
    
    # The created vocab as each word is seen
    vocab = {0: "<START>", 1: "<PAD>", 2: "<END>", 3: "<UNKNOWN>"}
    
    # Iterate over all lines in the file
    file = open(vocabFile, "r")
    for line in file:
        
        # Get the words
        words = get_clean_words(line)
        
        # Breakup the line into spaces and store any new words
        for word in words:
            if word.lower() not in vocab.values():
                # If the word is blank, don't add it
                if len(word) == 0:
                    continue
                
                vocab[len(vocab.keys())] = word.lower()
                
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
        file.write(f"{key},{vocab[key]}\n")
        
    # Close the file
    file.close()



generate()