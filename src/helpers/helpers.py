import torch
from nltk.tokenize import word_tokenize




def loadVocab(vocab_file):
    vocab = {}
    vocabFile = open(vocab_file, "r", encoding="utf-8")
    for line in vocabFile:
        # If the line is empty, skip it
        if len(line) < 2:
            continue
        
        # Clean the line
        if not line[2].isspace():
            line = line.strip()
        else:
            line = line[:3]
        
        # Get the key and value
        line = line.split(',', 1)
        
        # Add to the vocab
        vocab[int(line[0])] = line[1]
    vocabFile.close()
        
    return vocab



# Given a sentence, return a list of cleaned words
# Input:
#   Sentence - A sentence to clean
# Output:
#   A list of cleaned words
#   proportion of removed words in the sentence
def get_clean_words(Sentence):
    # Make the sentence lowercase and without redundant spacng
    Sentence = Sentence.lower()
    
    # Split the sentence up into words
    words = word_tokenize(Sentence)

    # Number of words before cleaning
    num_bef = len(words)
    
    # Remove all punctuation
    words = [word.lower() for word in words if word.isalnum()]

    # Number of words after cleaning
    num_af = len(words)
    
    # Return the words and proportion
    return words, num_af/num_bef




# Encode the sentences so it can be used by the Discriminator
# Input:
#   X - list of Strings where each string is a cleaned
#   vocab_inv - Inverted vocab where words map to their values
#   sequence_length - Length to encode each sequence to
#   encoder - Encoder object that excepts a word as
#             input and returns a vector form of that word
#   deleteOrig - True to delete X while encoding it, False otherwise
#   device - Device to put the tensors on
# Output:
#   A list of the same size with encoded sentences as tensors
def encode_sentences(X, vocab_inv, sequence_length, encoder, deleteOrig, device):
    # Get the encoder on the correct device
    try:
        encoder.to(device)
    except:
        pass

    # Final tensor of encoded sentences
    encoded = []

    # Get the encoded form of <END>
    end_enc = encoder(torch.tensor(vocab_inv["<END>"], device=device))

    # Get the encoded form of <START>
    start_enc = encoder(torch.tensor(vocab_inv["<START>"], device=device))
    
    # Iterate over all sentences
    i = 0
    while ((len(X) != 0 and deleteOrig == True) or (i < len(X) and deleteOrig == False)):
        # Get the sentence
        sentence = X[i]
        
        # Has the sentence been encoded?
        enc = True
        
        # List of encoded words
        enc_words = []
        
        # Get the words from the sentence
        words, prop = get_clean_words(sentence)
        
        # Iterate over each word
        for word in words:
            # If the word is blank, skip it
            if len(word) == 0:
                continue

            # Encode the word
            try:
                word_enc = encoder(torch.tensor(vocab_inv[word], device=device))
            except:
                enc = False
                break
            
            # Save the encoded word
            enc_words.append(word_enc)
        
        # If the word has not been encoded, an error happened, so
        # skip the sentence and delete it
        if enc == False or len(enc_words) == 0:
            del X[i]
            continue

        # Add an <START> token to the sequence
        enc_words = [start_enc] + enc_words

        # Add an <END> token to the sequence
        enc_words.append(end_enc)
        
        # Skip the sentence and delete it if it's too long
        if len(enc_words) > sequence_length:
            del X[i]
            continue
        
        # Turn the encoded words into a list and save it
        encoded.append(torch.stack(enc_words).detach().to(device))
        
        if deleteOrig == True:
            del X[i]
        else:
            i += 1
    
    
    # Return the list of encoded sentences
    return encoded



# Encode the sentences so it can be used by the Discriminator
# Input:
#   X - list of Strings where each string is a cleaned
#   vocab_inv - Inverted vocab where words map to their values
#   sequence_length - Length to encode each sequence to
#   deleteOrig - True to delete X while encoding it, False otherwise
#   device - Device to put the tensors on
# Output:
#   A list of the same size with encoded sentences as tensors
def encode_sentences_one_hot(X, vocab_inv, sequence_length, deleteOrig, device):
    # Final tensor of encoded sentences
    encoded = []

    # Get the encoded form of <END>
    end_enc = torch.nn.functional.one_hot(torch.tensor(vocab_inv["<END>"], device=device))

    # Get the encoded form of <START>
    start_enc = torch.nn.functional.one_hot(torch.tensor(vocab_inv["<START>"], device=device))
    
    # Iterate over all sentences
    i = 0
    while ((len(X) != 0 and deleteOrig == True) or (i < len(X) and deleteOrig == False)):
        # Get the sentence
        sentence = X[i]
        
        # Has the sentence been encoded?
        enc = True
        
        # List of encoded words
        enc_words = []
        
        # Get the words from the sentence
        words, prop = get_clean_words(sentence)
        
        # Iterate over each word
        for word in words:
            # If the word is blank, skip it
            if len(word) == 0:
                continue

            # Encode the word
            try:
                word_enc = torch.nn.functional.one_hot(torch.tensor(vocab_inv[word]), len(vocab_inv))
            except:
                enc = False
                break
            
            # Save the encoded word
            enc_words.append(word_enc)
            
        # If the word has not been encoded, an error happened, so
        # skip the sentence and delete it
        if enc == False or len(enc_words) == 0:
            del X[i]
            continue

        # Add a <START> token to the sequence
        enc_words = [start_enc] + enc_words

        # Add an <END> token to the sequence
        enc_words.append(end_enc)
        
        # Skip the sentence and delete it if it's too long
        if len(enc_words) > sequence_length:
            del X[i]
            continue
        
        # Turn the encoded words into a list and save it
        encoded.append(torch.stack(enc_words).detach().to(device).to(torch.bool))
        
        if deleteOrig == True:
            del X[i]
        else:
            i += 1
    
    
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
    # The padding tensor. Note <END> is used as a temp variable
    # since <PAD> is not in the vocab anymore
    pad_enc = encoder(torch.tensor(vocab_inv["<PAD>"]))

    # Instead of padding, use 0s
    pad_enc = pad_enc*0
    
    # The new padded tensor
    X_padded = torch.zeros(X.shape[0], sequence_length, pad_enc.shape[-1])
    
    # Iterate over all sentences
    for i in range(0, X.shape[0]):
        sentence = X[i]
        
        # Broadcast the pad encoding to the needed size
        pad = sequence_length-sentence.shape[0]
        pad_tensor = torch.broadcast_to(pad_enc, (pad, pad_enc.shape[-1]))
        
        # Add the pad tokens to the sentence
        sentence = torch.cat((sentence, pad_tensor.to(sentence.device)))
        
        # Save the new sentence
        X_padded[i] = sentence
    
    return X_padded



# Given a list of encoded sentences, add <PAD> tokens
# to those sentences
#   X - Encoded list of sentences
#   vocab_inv - Inverted vocab where words map to their values
#   sequence_length - Length to encode each sequence to
def addPadding_one_hot(X, vocab_inv, sequence_length):
    # The padding tensor. Note <END> is used as a temp variable
    # since <PAD> is not in the vocab anymore
    pad_enc = torch.nn.functional.one_hot(torch.tensor(vocab_inv["<END>"]), len(vocab_inv))

    # Instead of padding, use 0s
    # pad_enc = pad_enc*0
    
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
