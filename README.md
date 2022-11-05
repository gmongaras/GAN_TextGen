# Contents
- [Description](#description)
- [Requirements](#requirements)
- [Cloning This Repo](#cloning-the-repo)
- [Training a Model](#training-a-model)
- [Running the Model](#running-the-model)
- [Model Architecture](#model-architecture)
- [Creating a Vocabulary](#creating-a-vocabulary)
- [References](#references)


# Description
This project started as a side project for my [internship at Meta](https://github.com/gmongaras/MetaU_Capstone), but became a project I kind of became passionate in. Originally, I just wanted to generate text and I wanted to see if I could do so using a GAN. I was somewhat able to, but I wanted to try to make it better. Unfortunately, even though I tried for a while, I ran into many convergence issues when having the model estimate its own length.

The current project is my end result using GANs which resulted in text that looks better than random, but not entirely like a real sentence. Though, it's still fun to see what the generator produces.

While creating this project, I documented some of the problems I ran into, through these are defenitly not all of them and the documentation is very rough:
https://docs.google.com/document/d/1UA-dEF-IMr5-mOgkp6ORRtAERDEprKHw3K_Lw9A-vmU/edit?usp=sharing

# Requirements

This project was created using `python 3.9.12` but other versions should work too. You can download python [here](https://www.python.org/downloads/).
- Note that the latest version of python may not work with the project as the latest version may not support PyTorch.

Below are the package requirements for this project
```
PyTorch 1.12.1
Numpy 1.23.4
MatPlotLib 3.5.3
Click 8.0.4
Nltk 3.7
```

You can also paste the following into your terminal to download all the packages
```
pip install torch==1.12.1
pip install numpy==1.23.4
pip install matplotlib==3.5.3
pip install click==8.0.4
pip install nltk==3.7
```


# Cloning The Repo

To clone the repo, make sure you have git installed on your machine and run the following command:

`git clone https://github.com/gmongaras/PyTorch_TextGen.git`

The repo should be cloned to your machine, but some files are stored using git LFS. These files are the data files to train a model. If you do not wish to train a model, they are not needed, but if you want to train a model, they are required. First, make sure you have [git LFS](https://git-lfs.github.com/) installed on your machine. Then, 

Below is a list of all commands to run on the command line when pulling all data:
```
git clone https://github.com/gmongaras/GAN_TextGen.git
cd GAN_TextGen
```


# Constructing Data For Training

Before training the model, a dataset must first be created. This dataset can be custom or using the dataset I provide in this repo. I will be explaining how the dataset in this repo can be coonstructed for use by the model, but another dataset should be similar.

## Creating A Data File

The dataset I have in this repo contains many news articles and can be found in `data/Text/train_data`. To convert this dataset to a usable form, I created a single file with all the sentences I wanted to train the model which can be found in `data/Text/data.txt`. This file contains a bunch of sentences, each one of them cleaned and broken up by new line characters. To generate this file using the data I provide in the repo, you can use the `script.py` python file in `data/Text`. This file can be run using the following command in the root of the repo: `python -m data.Text.script.py`. The script has the following parameters:
- indir - Directory to load the news data from (this probably shouldn't be changed)
- outFile - File to write the output sentences to
- maxSentSize - Max number of words in a sentence. This corresponds to the length of sentences the model will generate.
- limit - Limit on the number of sentences to save to the file
- minLength - Limit on the minimum length of a sentence. A sentence with less words than this limit will not be saved to the output file.
- minProp - Minimum proportion or words that can be cleaned from a sentence
  - The proportion is calculated by `(number of words after cleaning)/(number of words before cleaning)`. So, this proportion represents how many words were not removed from a sentence and intuitively represents how clean a sentence was. A sentence with a high proprtion is "cleaner" as less words were removed from the sentence. I kept this proportion high so that dirty sentences with many words or characters removed were not used in the training data.

One note I want to make is that this script has a couple of important parts:
1. It generates a file of sentences where each word is broken up by a space and each sentence is broken up by a newline.
2. The sentences are cleaned so that there are no trash characters in the data.

The output of running `script.py` is an output file in the specified directory. This file has one more step before being ready to train the model with.

## Creating A Vocabulary

With a clean file of data, we can now generate a vocabulary for the model. The script to generate a vocabulary for the model is located in `src/helpers/generate_words.py`. To run this script, run the following command in the root directory of this repo: `python -m src.helpers.generate_words.py`. The script has the following parameters:
- vocabFile - File to load sentences and generate a vocabulary for (this is the output of [Creating A Data File](#creating-a-data-file))
- outFile - File to save the vocabulary to
- outTxtFile - A new text file that will be generated with the sentences that the vocabulary has words for.
- limit - Limit on the number of words that will be added to the vocabulary.
- randomize - When the vocabulary is generated, randomize the order when True, otherwise don't randomize it.

This script generates two files:
1. vocabFile - This is a dictionary-like file with the key as a numerical value in the range [0, limit] and a value of a word at that index value. This will be used to numerically encode each word.
2. outTxtFile - A new text file that stores the exact same sentence as vocabFile (the output of [Creating A Data File](#creating-a-data-file)), but all the sentences in this file have words in the generated vocabulary.
   - Why is this file generated? When the limit is met, new words cannot be added to the vocabulary. Instead of having unknown words in the input data, it is better to have data with words the model knows. This file only have sentences with words in the vocab while the original vocabFile has sentences with all words. This file is maximized though. If the max vocab cap is hit, only sentences without new words are added to this file as opposed to immediately stopping the creation of this new vocab file.

The outTxtFile will be used to train the model to learn how to generate sentences and the outFile will be used to map words to their numerical representation.


# Training A Model

Before going into training the model, ensure you have the data downloaded from git lfs from the [Cloning The Repo](#cloning-the-repo) secion. Or you can build your own dataset 



# Model Architecture

This section of the README goea over how the model works in detail. Before going into how the models work, there is some notation I am using to make it less cluttered:
- N - Batch size
- S - Sequence length
- E - Encoding size for each word/token
- V - Vocab size
- <i>N</i>(0, 1) - Represents a normal distribution with a mean of 0 and a variance of 1, but this can actually be any distribution we want to generator to model.
- Feed Forward - This layer is not a single linear layer, rather it is a linear layer projecting the data from E to a higher dimensional space, an activation like GELU, then another linear layer projecting the data back to E.

## Generator

Below is a diagram of the generator model

<p align="center">
   <img src="https://github.com/gmongaras/PyTorch_TextGen/raw/main/Model_Diagrams/Generator.png" alt="Generator" width="1000"/>
</p>

The generator is a recurrent model that generates a word one at a time from noise. Let's take a look at the steps for how the generator might generate a sentence:
1. First the generator samples noise of size (N, S, E) and encodes this noise using <i>M</i> linear layers to the same shape as done in the [StyleGAN model](https://arxiv.org/pdf/1812.04948.pdf).
   - Note: Like with the StyleGAN model, I was thinking the model may be able to control the overall style of the sentence using cross attention with the same noise throughout the entire sequence as this noise is applied globally to the sequence.
2. Next, the generator samples noise of shape (N, 1, E), preparing to generate the next word. If this is not the first word to be generated, append this noise token to the end of the currently generated sequence.
   - Note: There are a few options I could have picked here:
      1. Have the model generate a new token using the current last token in the sequence
      2. Have the model generate a new token using a predefined token as input
      3. Have the model generate a new token using random noise
   - I decided to go with the third option as it seemed to help the model generate better sentence with a high variety.
3. Positional encodings are added to the sampled noise
4. The sampled noise with positional encodings is sent through <i>B</i> number of generator backbone blocks with the following steps to generate an encoded form of the noise:
   1. Apply self-multihead attention to the input noise
   2. Apply layer norm and a skip connection
   3. Add noise of shape (N, 1, E) to the current embeddings
      - Quick note: I add noise here to as I wanted to give the model nosie that the it didn't have too much control over. There isn't any mathematical reason, I just thought it would help the model produce a higher variety of output.
   4. Use cross attention to add the global noise to the embeddings
   5. Apply layer normalization and a skip connection from the embeddings after the durect noise was added to the sequence
   6. Apply a feed forward layer
   7. Apply layer normalization and a skip connection from the embeddings before the feed forward layer
5. The embeddings are now sent through <i>O</i> number of output blocks with the following structure:
   1. Apply self-multihead attention to the input embeddings
   2. Apply layer norm and a skip connection
   3. Apply a feew forward layer
   4. Apply layer norm and a skip connection
   - Note: The model has two types of blocks: one that add noise and one that does not. The idea is that I wanted the model to encode the representation of the sentence and word through the noise blocks and then generate the next word prediction from those encodings using the second blocks without noise.
6. Slice the output of the generator to get the last token in the sequence. This token is what will be used to predict the next word. The rest of the sequence is not needed.
7. Apply a linear layer and a softmax layer to transform the embeddings from (N, 1, E) to (N, 1, V) to get the probabilities for the next word in the sequence
8. Save the current output from step 6 <b>while retaining the gradient of the current output</b> for the discriminator so that the generator can be updated from the discriminator.
9. Apply the argmax function and a one-hot encoding function to the output of step 7.
   - The output of these functions retains the shape of the output of step 7: (N, 1, V)
   - These function encode the
   - These functions also break the gradient computation for the generator, so the generator output is not updated sequence of sequential outputs, rather it is updated as a sequence of outputs generated at the same time.
   - Why did I do this? There is one main reason for doing so:
     - I didn't want to generator to adapt it's output of one token to help it generate the next token. I was afraid the generator may change how it was generating say the first token to help generate the second token as the first token directly effects the second token and the generator would be able to change the first toke nto help it generate the second. Breaking the gradient graph forces it to learn individual tokens without updating the others, but still allows it to learn how one token should come after the other.
     - Note: The sequence information is still retained throughout the generator output. It still have knowledge of previous outputs, it just can't update those previous outputs directly based on the future outputs.
10. Using the currently generated. Repeat from step 2 until there are a total of <i>S</i> word outputs.
11. When there are <i>S</i> word outputs, concatentate the outputs from step 6 into an embedded sequence of shape (N, S, V) for the discriminator to handle.


## Disciminator

Below is a diagram of the disciminator model

<p align="center">
  <img src="https://github.com/gmongaras/PyTorch_TextGen/raw/main/Model_Diagrams/Discriminator.png" alt="Discriminator" width="400"/>
</p>

The discriminator is a regressive model (not a classification model as it's based on the WGAN loss function) that scores a sequence based on how real it "thinks" it is. Like with the generator, let's step through how the discriminator scores a sequence:
1. The input into the model is either:
   1. A sequence of real sentences. These sentences must first be encoded to a vector form.
   2. A sequence of fake sentences from the generator which are already in the correct form.
2. The input sequences are fed into a linear layer to project the tokens from the vocab size, <i>V</i>, to the embedding size, <i>E</i>
3. Positional encodings of shape (N, S, E) are concatentated to the end of the sequences along the word embedding dimension to produce a tensor of shape (N, S, 2E).
   - Note: Unlike with the generator, the embeddings are not directly added to the embeddings. I found that adding the positional encodings to the word embeddings caused the discriminator to have trouble learning the format of a sequence. This makes sense as adding positional encodings to a sequence changes how each word is embedded which changes how it is represented in the embedding space and may lead the embeddings to represent something different than the word it actually should represent.
4. The embeddings are now sent through <i>B</i> number of backbone blocks with the following structure:
   1. Apply self-multihead attention to the input embeddings
   2. Apply layer norm and a skip connection
   3. Apply a feew forward layer
   4. Apply layer norm and a skip connection
5. Class tokens of shape (N, 1, 2E) are concatenated to the sequences to produce a tensor of shape (N, S+1, 2E). These tokens will be used to score each sequence.
6. The embeddings with the class token are now sent through <i>O</i> number of output blocks with the following structure:
   1. Apply self-multihead attention to the input embeddings
   2. Apply layer norm and a skip connection
   3. Apply a feew forward layer
   4. Apply layer norm and a skip connection
7. The class token is sliced from each of the sequences to get tensors of shape (N, 1, 2E)
8. A linear layer is applied to the class token outputs to project the outputs to a single dimension. The tensor is of shape (N, 1, 1). This tensor is flattened to the shape (N) representing the score for each sequence.


## Training Specifics

I used the [WGAN-GP](https://arxiv.org/abs/1704.00028) model to train my GAN. The original GAN paper used the following formula to train the model:

<p align="center">
   <img width="317" alt="image" src="https://user-images.githubusercontent.com/43501738/200096663-19c51883-7c49-42ba-b965-2b138de084e4.png">
</p>

Where G is the generator, D is the discriminator, x is the real data, and x_tilde is the fake data generated by the generator. However, this objective has a lot of convergence problems. The WGAN-GP model is similar to a normal GAN model, but it attempts to minimize the Earth-mover distance using the following objective:

<p align="center">
   <img width="252" alt="image" src="https://user-images.githubusercontent.com/43501738/200096732-7790fc78-f719-4b19-9246-10a299c149d6.png">
</p>

This is the objective I use in my model. The "-GP" part of WGAN-GP means I am using the gradient penalty variation of WGAN as opposed to the weight clipping variation.

I am not going to go into detail into how these loss function work as there are many articles that explain how they word in great detail.

## Length Esimation Failure

I want to add this section as most of my time spent was trying to make the model generate its own length in some way or another. Why would I want to do this? I though that having the model generate its own length would help it generate better sentences as opposed to having the model learn how to pad a sentence. My initial attempt masked the \<PAD\> tokens, but this immediately led to divergence in the discriminator. From what I remember, I tried the following (and probably much more):
- Having a separate model learn the lengths for the generator independently from the rest of the model.
- Having the independent model be trained by the discriminator instead of by a different loss function
- Having the generator produce its own length estimation, whether that's from another token prediction or some other way
- Having a second generator model generate the length of the sequence when the generator is done generating a sentence, then feed both the lengths and sequence into the generator and mask the \<PAD\> tokens
- Having a second generator and second discriminator which are independent GANs. One GAN produced sequences, the other produced lengths based on the sequences.

None of these attempts worked and all led to some type of divergence in the model.


# References
1. StyleGAN: https://arxiv.org/abs/1812.04948v3
2. WGAN: https://arxiv.org/abs/1701.07875
3. WGAN-GP: https://arxiv.org/abs/1704.00028


Data from the following sources:
- Random Text: https://www.kaggle.com/datasets/kashnitsky/hierarchical-text-classification
- Billion word language modeling benchmark: https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark
- Fortunes 1: https://github.com/ruanyf/fortunes
- Fortunes 2: http://www.fortunecookiemessage.com/
