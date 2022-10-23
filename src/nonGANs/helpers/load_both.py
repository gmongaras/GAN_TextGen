import re
import torch



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
def load_both(filename, maxLen, vocab, input_size, lower, limit):
    # Encoded start and end tokens. Note that the sentences
    # don't use pad tokens, just end tokens
    start_tok = [vocab["¶"]] + [vocab["↔"] for i in range(0, input_size-1)]
    # The end token is one end character token with
    # a bunch of pad tokens. Essentially, a blank word.
    end_tok = [vocab["∅"]] + [vocab["↔"] for i in range(0, input_size-1)]
    
    # Space token. Each word will have a space at the end of it
    space_tok = [vocab[" "]]
    
    # The starting sentence is a sentence without anything
    # but start tokens in it and a start token at the beginning. 
    # It is of shape (maxLen, input_size)
    # as there will be 'maxLen' number of words and
    # 'input_size' number of characters per word.
    sentence_template = torch.cat((torch.tensor([start_tok]), torch.broadcast_to(torch.tensor(end_tok), (maxLen-1, input_size))))
    
    # The encoded setences.
    y = []
    
    # Prepare the data as a sequence of characters for each sentence
    with open(filename, encoding='utf-8') as file:
        # Iterate over all lines in the file
        for line in file:
            
            if limit == 0:
                break
            limit -= 1
            
            # Remove whitespace from the beginning and end of the sentence
            line = line.strip()
            if lower:
                line = line.lower()
        
            # Breakup the sentence by spaces or ...
            # to get each word
            words = re.split(r"\.{3}|[ ]", line)
            
            # If the number of words is too long,
            # skip this sentence
            if len(words) > maxLen-2:
                continue
            
            # Sentence tensor that contains a tensor of
            # encoded words.
            sentence = sentence_template.clone()
            
            # Was a word too large?
            wordTooLarge = False
            
            # Iterate over each word
            i = 1
            idx = 1
            for word in words:
                # Add a space to the word if it's not
                # the last word in the sentence
                if i < len(words):
                    word += " "
                i += 1
                
                # Encode the word to it's numerical form
                word = [vocab[w] for w in word]
                
                # Add the, start, end, and pad tokens to the word
                word = start_tok[:1] + word + end_tok[:input_size-len(word)-1]
                
                # If the length of the word is too large,
                # don't encode the sentence
                if (len(word) > input_size) or (word[-1] != end_tok[0] and word[-1] != vocab["↔"]):
                    wordTooLarge = True
                    break
                
                # Save the word to the sentence
                sentence[idx] = torch.tensor(word)
                idx += 1
            
            # If a word was too long, skip this sentence
            if wordTooLarge:
                continue
            
            # Save the sentence
            y.append(sentence)
    
    # Return the encoded sentences
    return torch.stack(y).float()