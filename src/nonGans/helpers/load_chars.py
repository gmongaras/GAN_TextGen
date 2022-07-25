import torch




# Given a filename which stores data to load in, load in the
# data as a set of characters.
# Input:
#   filename - The filename with the data to load in
#   maxLen - Max number of characters to load in for each input sentence
#   vocab - Dictionary mapping characters to their integer encoding
# Output:
#   X - A tensor of shape (N, maxLen, 1) which is the character encoding
#       for each sentence
#   y - A tensor of shape (N, maxLen, vocabSize) which is the one-hot
#       encoding for each sentence
def load_chars(filename, maxLen, vocab):
    # Does the vocabulary have capital letters?
    lower = True
    for char in vocab:
        if char.isupper() and char.isalpha():
            lower = False
    
    # Encoded end and pad tokens
    end_tok = [vocab["∅"]]
    pad_toks = [vocab["~"] for i in range(0, maxLen-1)]
    
    # The encoded setences
    X = []
    y = []

    # Prepare the data as a sequence of characters for each sentence
    with open(filename, encoding='utf-8') as file:
        for f in file:
            # Clean f a little
            f = f.strip()
            if lower:
                f = f.lower()

            # Get the line as a list of encoded characters
            try:
                c = [vocab[i] for i in f]
            except:
                continue

            # Add the end token to the set of characters
            c += end_tok

            # If the length is too long, skip this sentence
            if len(c) > maxLen -1:
                continue
            
            # Add padding to the sentence
            c += pad_toks[len(c)-1:maxLen]

            # Convert the data to a tensor
            c = torch.tensor(c, dtype=torch.float32, requires_grad=False)

            # One hot encode the sentence
            c_onthot = torch.nn.functional.one_hot(c.long(), num_classes=len(vocab))

            # Normalize the tensor between 0 and 1
            #c = c/n_vocab
            
            # Save the sentence as a torch tensor
            X.append(c.unsqueeze(-1))
            y.append(c_onthot)
    
    # Get the data as a complete tensor
    return torch.stack(X).float(), torch.stack(y).float()