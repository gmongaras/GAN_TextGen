import torch
from ...helpers.helpers import get_clean_words




# Given a filename which stores data to load in, load in the
# data as a set of words.
# Input:
#   filename - The filename with the data to load in
#   maxLen - Max number of words to load in for each input sentence
#   vocab - Dictionary mapping words to their integer encoding
#   input_size - Size of the vector to embed each word to
#   lower - Should the words be lowercased when embedding?
#   limit - Limit on the number of sentences to load in
# Output:
#   X - A tensor of shape (N, maxLen, input_size) which is the word
#       encoding for each sentence
#   y - A tensor of shape (N, maxLen, vocabSize) which is the one-hot
#       encoding for each sentence
def load_words(filename, maxLen, vocab, input_size, lower, limit):
    # Used to encode words to their vector form
    embedder = torch.nn.Embedding(len(vocab.keys()), input_size)
    
    # Encoded end and pad tokens
    start_tok = [embedder(torch.tensor(vocab["<START>"]))]
    start_tok_num = [vocab["<START>"]]
    end_tok = [embedder(torch.tensor(vocab["<END>"]))]
    end_tok_num = [vocab["<END>"]]
    pad_tok = embedder(torch.tensor(vocab["<PAD>"]))
    pad_toks = [pad_tok for i in range(0, maxLen-1)]
    pad_toks_nums = [vocab["<PAD>"] for i in range(0, maxLen-1)]
    
    # The encoded setences
    X = []
    y = []

    # Prepare the data as a sequence of characters for each sentence
    with open(filename, encoding='utf-8') as file:
        # Iterate over all sentences in the file
        for sentence in file:
            
            # Control how many sentences are loaded
            if limit == 0:
                break
            limit -= 1
            
            # Clean the sentence a little
            sentence = sentence.strip()
            if lower:
                sentence = sentence.lower()
            sentence, prop = get_clean_words(sentence)

            # Get the line as a list of encoded characters
            try:
                enc = [embedder(torch.tensor(vocab[i])) for i in sentence]
                enc_nums = [vocab[i] for i in sentence]
            except:
                continue
            
            # Add the start token to the beginning of the words
            enc = start_tok + enc
            enc_nums = start_tok_num + enc_nums

            # Add the end token to the set of words
            enc += end_tok
            enc_nums += end_tok_num

            # If the length is too long, skip this sentence
            if len(enc) > maxLen -1:
                continue
            
            # Add padding to the sentence
            enc += pad_toks[len(enc)-1:maxLen]
            enc_nums += pad_toks_nums[len(enc_nums)-1:maxLen]

            # Convert the data to a tensor
            enc = torch.stack(enc).long()
            enc_nums = torch.tensor(enc_nums, dtype=torch.long, requires_grad=False)

            # One hot encode the sentence
            enc_onthot = torch.nn.functional.one_hot(enc_nums, num_classes=len(vocab))

            # Normalize the tensor between 0 and 1
            #c = c/n_vocab
            
            # Save the sentence as a torch tensor
            X.append(enc)
            y.append(enc_onthot)
    
    # Get the data as a complete tensor
    return torch.stack(X).float(), torch.stack(y).float()