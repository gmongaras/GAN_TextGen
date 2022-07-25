



# Given a filename which stores data to load in, load in the
# data as a set of characters in batches of words.
# Input:
#   filename - The filename with the data to load in
#   maxLen - Max number of words to load in for each input sentence
#   vocab - Dictionary mapping characters to their integer encoding
#   input_size - Size of the vector to embed each word to. This is
#                the max number of chararacters each word can have
#   lower - Should the words be lowercased when embedding?
#   limit - Limit on the number of sentences to load in
# Output:
#   X - A tensor of shape (N, maxLen, input_size) which is the word
#       encoding for each sentence
#   y - A tensor of shape (N, maxLen, vocabSize) which is the one-hot
#       encoding for each sentence
def load_both(filename, maxLen, vocab, input_size, lower):
    # Encoded start and end tokens. Note that the sentences
    # don't use pad tokens, just end tokens
    start_tok = [vocab["<"] + vocab["S"] + vocab["T"] + vocab["A"] + vocab["R"] + vocab[">"]] + \
        [vocab["∅"]] + [vocab["~"] for i in range(0, input_size-8)]
    # The end token is one end character token with
    # a bunch of pad tokens. Essentially, a blank word.
    end_tok = [vocab["∅"]] + [vocab["~"] for i in range(0, input_size-1)]
    
    # The encoded setences
    X = []
    
    # Prepare the data as a sequence of characters for each sentence
    with open(filename, encoding='utf-8') as file:
        # Iterate over all lines in the file
        for line in file:
        
            # Breakup the sentence by spaces to get each word
            words = line.split(" ")