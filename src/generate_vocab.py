# Generate a vocab file csv with two columns, a key
# and a value




def generate():
    vocabFile = "data/data.txt"
    unwanted = ["!", "\"", "#", "$", "%", "&", 
                "(", ")", "*", "+", ",", "-", ".",
                "/", ":", ";", "<", "=", ">", "?", "@",
                "[", "\\", "]", "^", "_", "`", "{", "|",
                "}", "~", "\n", "\t"]
    outFile = "vocab.csv"
    limit = 100000
    
    
    # The created vocab as each word is seen
    vocab = {0: "<START>", 1: "<PAD>", 2: "<END>", 3: "<UNKNOWN>"}
    
    # Iterate over all lines in the file
    file = open(vocabFile, "r")
    for line in file:
        # Replace any characters in the line we don't want
        for i in unwanted:
            line = line.replace(i, "")
        
        # Breakup the line into spaces and store any new words
        words = line.split(" ")
        for word in words:
            if word.lower() not in vocab.values():
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