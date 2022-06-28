import torch




def loadVocab(vocab_file):
    vocab = {}
    vocabFile = open(vocab_file, "r")
    for line in vocabFile:
        # Clean the line
        line = line.strip()
        
        # Get the key and value
        line = line.split(',')
        
        # Add to the vocab
        vocab[int(line[0])] = line[1]
        
    return vocab



# Given a sentence, return a list of cleaned words
# Input:
#   Sentence - A sentence to clean
# Output:
#   A list of cleaned words
def get_clean_words(Sentence):
    unwanted = ["!", "\"", "#", "$", "%", "&", 
                "(", ")", "*", "+", ",", "-", ".",
                "/", ":", ";", "<", "=", ">", "?", "@",
                "[", "\\", "]", "^", "_", "`", "{", "|",
                "}", "~", "\n", "\t"]
    
    # Replace any characters in the line we don't want
    for i in unwanted:
        Sentence = Sentence.replace(i, " ")
    Sentence = Sentence.lower()
    
    # Breakup the string and return it
    return Sentence.split(" ")




# Encode the sentences so it can be used by the Discriminator
# Input:
#   X - list of Strings where each string is a cleaned
#   vocab_inv - Inverted vocab where words map to their values
#   sequence_length - Length to encode each sequence to
#   encoder - Encoder object that excepts a word as
#             input and returns a vector form of that word
#   device - Device to put the tensors on
# Output:
#   A list of the same size with encoded sentences as tensors
def encode_sentences(X, vocab_inv, sequence_length, encoder, device):
    # Get the encoder on the correct device
    encoder.to(device)

    # Final tensor of encoded sentences
    encoded = []
    
    # Get the encoded form of <END>
    end_enc = encoder(torch.tensor(vocab_inv["<END>"], device=device))
    
    # Get the encoded form of <START>
    start_end = encoder(torch.tensor(vocab_inv["<START>"], device=device))
    
    # Iterate over all sentences
    for sentence in X:
        # List of encoded words starting with <START>
        enc_words = [start_end]
        
        # Get the words from the sentence
        words = get_clean_words(sentence)
        
        # Iterate over each word
        for word in words:
            # If the word is blank, skip it
            if len(word) == 0:
                continue

            # Encode the word
            word_enc = encoder(torch.tensor(vocab_inv[word], device=device))
            
            # Save the encoded word
            enc_words.append(word_enc)
        
        # Add an <END> token to the sequence
        enc_words.append(end_enc)
        
        # Skip the sentence is it's too long
        if len(enc_words) > sequence_length:
            continue
        
        # Turn the encoded words into a list and save it
        encoded.append(torch.stack(enc_words).detach().to(device))
    
    
    # Return the list of encoded sentences
    return encoded



# Encode the sentences so it can be used by the Discriminator
# Input:
#   X - list of Strings where each string is a cleaned
#   vocab_inv - Inverted vocab where words map to their values
#   sequence_length - Length to encode each sequence to
#   device - Device to put the tensors on
# Output:
#   A list of the same size with encoded sentences as tensors
def encode_sentences_one_hot(X, vocab_inv, sequence_length, device):
    # Final tensor of encoded sentences
    encoded = []
    
    # Get the encoded form of <END>
    end_enc = torch.nn.functional.one_hot(torch.tensor(vocab_inv["<END>"]), len(vocab_inv))
    
    # Get the encoded form of <START>
    start_end = torch.nn.functional.one_hot(torch.tensor(vocab_inv["<START>"]), len(vocab_inv))
    
    # Iterate over all sentences
    for sentence in X:
        # List of encoded words starting with <START>
        enc_words = [start_end]
        
        # Get the words from the sentence
        words = get_clean_words(sentence)
        
        # Iterate over each word
        for word in words:
            # If the word is blank, skip it
            if len(word) == 0:
                continue

            # Encode the word
            word_enc = torch.nn.functional.one_hot(torch.tensor(vocab_inv[word]), len(vocab_inv))
            
            # Save the encoded word
            enc_words.append(word_enc)
        
        # Add an <END> token to the sequence
        enc_words.append(end_enc)
        
        # Skip the sentence is it's too long
        if len(enc_words) > sequence_length:
            continue
        
        # Turn the encoded words into a list and save it
        encoded.append(torch.stack(enc_words).detach().to(device))
    
    
    # Return the list of encoded sentences
    return encoded



# Given a list of encoded sentences, add <PAD> tokens
# to those sentences
#   X - Encoded list of sentences
#   vocab_inv - Inverted vocab where words map to their values
#   sequence_length - Length to encode each sequence to
#   encoder - Encoder object that excepts a word as
#             input and returns a vector form of that word
def addPadding(X, vocab_inv, sequence_length, encoder):
    # The padding tensor
    pad_enc = encoder(torch.tensor(vocab_inv["<PAD>"]))
    
    # The new padded tensor
    X_padded = torch.zeros(X.shape[0], sequence_length, pad_enc.shape[-1])
    
    # Iterate over all sentences
    for i in range(0, X.shape[0]):
        sentence = X[i]
        
        # Broadcast the pad encoding to the needed size
        pad = sequence_length-sentence.shape[0]
        pad_tensor = torch.broadcast_to(pad_enc, (pad, pad_enc.shape[-1]))
        
        # Add the pad tokens to the sentence
        sentence = torch.cat((sentence, pad_tensor.to(setence.device)))
        
        # Save the new sentence
        X_padded[i] = sentence
    
    return X_padded



# Given a list of encoded sentences, add <PAD> tokens
# to those sentences
#   X - Encoded list of sentences
#   vocab_inv - Inverted vocab where words map to their values
#   sequence_length - Length to encode each sequence to
def addPadding_one_hot(X, vocab_inv, sequence_length):
    # The padding tensor
    #pad_enc = encoder(torch.tensor(vocab_inv["<PAD>"]))
    pad_enc = torch.nn.functional.one_hot(torch.tensor(vocab_inv["<PAD>"]), len(vocab_inv))
    
    # The new padded tensor
    X_padded = torch.zeros(len(X), sequence_length, pad_enc.shape[-1])
    
    # Iterate over all sentences
    for i in range(0, len(X)):
        sentence = X[i]
        
        # Broadcast the pad encoding to the needed size
        pad = sequence_length-sentence.shape[0]
        pad_tensor = torch.broadcast_to(pad_enc, (pad, pad_enc.shape[-1]))
        
        # Add the pad tokens to the sentence
        sentence = torch.cat((sentence, pad_tensor.to(sentence.device)))
        
        # Save the new sentence
        X_padded[i] = sentence
    
    return X_padded
