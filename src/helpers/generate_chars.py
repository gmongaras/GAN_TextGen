import sys



# Generate a vocab file csv with two columns, a key
# and a value
def generate():
    vocabFile = "data/Fortunes/data.txt"
    outFile = "vocab_chars.csv"
    limit = 1000000000000
    startTok = "¶"
    endTok = "∅"
    padTok = "↔"
    lowercase = False
    
    
    # The created vocab as each word is seen
    vocab = {startTok: 0, endTok: 1, padTok: 2}
    
    i = len(vocab)
    
    # Iterate over all lines in the file
    file = open(vocabFile, "r", encoding='utf-8')
    for line in file:
        # Iterate over each character in the line:
        for char in line:
            # Make the character lowercased
            if lowercase:
                char = char.lower() 
            
            # If the character is not already in the dictionary,
            # add it
            if char not in vocab:
                vocab[char] = i
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