from .helpers import get_clean_words
import random


# Generate a vocab file csv with two columns, a key
# and a value
def generate():
    vocabFile = "data/Text2/data.txt"
    outFile = "vocab_text2.csv"
    outTxtFile = "data/Text2/data2.txt"
    limit = 70000
    randomize = True # Randomize the vocab to stray from bias
    
    
    # The created vocab as each word is seen
    vocab = {"<START>": 0, "<END>": 2}
    
    i = len(vocab)
    sents = 0
    
    # Iterate over all lines in the file
    file = open(vocabFile, "r", encoding="utf-8")
    txtFile = open(outTxtFile, "w", encoding="utf-8")
    for line in file:
        # Get the words
        line = line.strip()
        words, prop = get_clean_words(line)

        # Has the limit been reached?
        limitIssue = False
        
        # Breakup the line into spaces and store any new words
        for word in words:
            # Make the word lowercased
            word = word.lower().strip()
            
            # If the word is blank, don't add it
            if len(word) == 0:
                continue
            
            # Check if the limit has been reached
            if len(vocab.keys()) >= limit:
                # Is the word in the vocab?
                if word not in vocab:
                    limitIssue = True
                    break

            # If the limit hasn't been reached, add the word to
            # the vocab if it isn't already in there
            else:
                if word not in vocab:
                    vocab[word] = i
                    i += 1

        # If there isn't a sentence issue, save the senence
        if not limitIssue:
            txtFile.write(" ".join(words) + "\n")
            sents += 1
        
        # Check if the limit has been reached
        # if len(vocab.keys()) > limit:
        #     break
    file.close()
    txtFile.close()

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