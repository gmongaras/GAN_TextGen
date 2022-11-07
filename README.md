# Contents
- [Description](#description)
- [Requirements](#requirements)
- [Cloning This Repo](#cloning-the-repo)
- [Constructing Data For Training](#constructing-data-for-training)
  - [Creating A Data File](#creating-a-data-file)
  - [Creating A Vocabulary](#creating-a-vocabulary)
- [Training a Model](#training-a-model)
- [Running the Model](#running-the-model)
- [Model Results](#model-results)
- [Model Architecture](#model-architecture)
  - [Generator](#generator)
  - [Discriminator](#discriminator)
  - [Training Specifics](#training-specifics)
  - [Length Estimation Failure](#length-estimation-failure)
- [References](#references)


# Description
This project started as a side project for my [internship at Meta](https://github.com/gmongaras/MetaU_Capstone), but became a project I kind of became passionate about. Originally, I just wanted to generate text and I wanted to see if I could do so using a GAN. I was somewhat able to, but I wanted to try to make it better. Unfortunately, even though I tried for a while, I ran into many convergence issues when having the model estimate its own length.

The current project is my end result using GANs which resulted in text that looks better than random, but not entirely like a real sentence. Though, it's still fun to see what the generator produces.

While creating this project, I documented some of the problems I ran into, though these are definitely not all of them and the documentation is very rough:
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

The repo should be cloned to your machine, but some files are stored using git LFS. These files are the data files to train a model. If you do not wish to train a model, they are not needed, but if you want to train a model, they are required. First, make sure you have [git LFS](https://git-lfs.github.com/) installed on your machine. Then, in the root of the repo on your machine, run the command `git lfs pull`

Below is a list of all commands to run on the command line when pulling all data:
```
git clone https://github.com/gmongaras/GAN_TextGen.git
cd GAN_TextGen/
git lfs pull
```

Some pretrained models might be in Google Drive. For example, Model 2 is stored using Google Drive. To download those files, go into the drive folder and download them into the desired directory.


# Constructing Data For Training

Before training the model, a dataset must first be created. This dataset can be custom or use the dataset I provide in this repo. I will be explaining how the dataset in this repo can be constructed for use by the model, but another dataset should be similar.

## Creating A Data File

The dataset I have in this repo contains many news articles and can be found in `data/Text/train_data`. To convert this dataset to a usable form, I created a single file with all the sentences I wanted to train the model which can be found in `data/Text/data.txt`. This file contains a bunch of sentences, each one of them cleaned and broken up by new line characters. To generate this file using the data I provide in the repo, you can use the `script.py` python file in `data/Text`. This file can be run using the following command in the root of the repo: `python -m data.Text.script.py`. The script has the following parameters:
- indir - Directory to load the news data from (this probably shouldn't be changed)
- outFile - File to write the output sentences to
- maxSentSize - Max number of words in a sentence. This corresponds to the length of sentences the model will generate.
- limit - Limit on the number of sentences to save to the file
- minLength - Limit on the minimum length of a sentence. A sentence with fewer words than this limit will not be saved to the output file.
- minProp - Minimum proportion or words that can be cleaned from a sentence
  - The proportion is calculated by `(number of words after cleaning)/(number of words before cleaning)`. So, this proportion represents how many words were not removed from a sentence and intuitively represents how clean a sentence was. A sentence with a high proportion is "cleaner" as fewer words were removed from the sentence. I kept this proportion high so that dirty sentences with many words or characters removed were not used in the training data.

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
   - Why is this file generated? When the limit is met, new words cannot be added to the vocabulary. Instead of having unknown words in the input data, it is better to have data with words the model knows. This file only has sentences with words in the vocab while the original vocabFile has sentences with all words. This file is maximized though. If the max vocab cap is hit, only sentences without new words are added to this file as opposed to immediately stopping the creation of this new vocab file.

The outTxtFile will be used to train the model to learn how to generate sentences and the outFile will be used to map words to their numerical representation.


# Training A Model

Before going into training the model, ensure you have the data downloaded from git lfs from the [Cloning The Repo](#cloning-the-repo) section. Or you can build your own dataset by following [Constructing Data For Training](#constructing-data-for-training).

To train a model, you can run the following command in the root directory:

`python -m src.GANs.train`

to change the parameters of the model, add parameters flags like the following:

`python -m src.GANs.train --[flag 1] [flag value] --[flag 2] [flag value]`

Where [flag 1] is replaced by a parameter name and [flag value] is replaced by the value of the parameter. For example, I could change the number of epochs to 100 like the following:

`python -m src.GANs.train --epochs 100`

The training script has the following parameters:

<b>Required Parameters</b>:
- input_file [data/Text/data_clean.txt] - (String) File to load the data from. This is the output file from [Creating A Vocabulary](#creating-a-vocaulary)
- vocab_file [vocab_text.csv] - (String) Location of the csv file storing the vocabulary dictionary
- num_to_load [-1] - (Integer) Number of sentences to load from the input file. Use -1 to load all sentences from the input file

<b>Saving Parameters</b>:
- saveDir [models] - (String) Directory to save model checkpoints and graphs
- saveDefFile [model_def.json] - (String) File to save GAN defaults to so it can be easily loaded in
- genSaveFile [gen_model.pkl] - (String) File to save generator models to
- genSaveDefFile [gen_model_def.json] - (String) File to save generator defaults to so it can be easily loaded in when generating sentences
- discSaveFile [disc_model.pkl] - (String) File to save discriminator models to
- trainGraphFile [trainGraph.png] - (String) File to save the loss graph to

<b>Saved State Loading Parameters</b>:
- loadPreTrained [False] - (Boolean) True to load a pre-trained model, False to start from a random model
  - Note: All other parameters under this section are inactive if loadPreTrained is False, otherwise, they will be used
- loadDir [models] - (String) Path to load models from
- loadDefFile [model_def.json] - (String) File with GAN defaults (only used if loadPreTrained is True)
- genLoadFile [gen_model.pkl] - (String) File to load a pre-trained generator model from
- discLoadFile [disc_model.pkl] - (String) File to load a discriminator model from

<b>General Model Parameters</b>:
- M_gen [8] - (Integer) Number of input noise embedding blocks (number of linear layers to encode the noise)
- B_gen [6] - (Integer) Number of transformer blocks to encode the input sequence (Number of transformer blocks with cross attention)
- O_gen [6] - (Integer) Number of transformer blocks to get the output sequence (Number of transformer blocks without cross attention)
- embedding_size_gen [64] - (Integer) Word embedding size for the generator
- T_disc [2] - (Integer) Number of transformer blocks in each discriminator block
- B_disc [6] - (Integer) Number of discriminator blocks in the discriminator (Number of discriminator blocks before the class token)
- O_disc [6] - (Integer) Number of output transformer blocks in the discriminator (Number of transformer blocks after the class token)
- embedding_size_disc [64] - (Integer) Word embedding size for the discriminator
- hiddenSize [512] - (Integer) Hidden linear size in the transformer blocks (Projection size in the Feed Forward part of each transformer block)
- batchSize [64] - (Integer) Batch size used to train the model
- sequence_length [64] - (Integer) Max number of words in a sentence to train the model with
- num_heads [8] - (Integer) Number of heads in each MHA block
- noiseDist [unif] - (String) Distribution to sample noise from. Can be one of (\"norm\", \"unif\", \"trunc\" (for truncated normal))
- pooling [none] - (String) Pooling mode for the discriminator blocks (\"avg\" to use average pooling, \"max\" to use max pooling, or \"none\" to use no pooling)
- alpha [0.0001] - (Float) Model learning rate
- Beta1 [0.0] - (Float) Adam beta 1 term
- Beta2 [0.9] - (Float) Adam beta 2 term
- device [partgpu] - (String) Device to put the model on (\"cpu\", \"fullgpu\", or \"partgpu\")
- epochs [100000] - (Integer) Number of epochs to train the model
- n_D [5] - (Integer) Number of times to train the discriminator more than the generator for each epoch
- saveSteps [100] - (Integer) Number of steps until the model is saved
- loadInEpoch [False] - (Boolean) Should the data be loaded in as needed instead of before training? (True if so, False to load before training)
- delWhenLoaded [False] - (Boolean) Delete the data as it's loaded into free allocated memory? Note: This is automatically False if loadInEpoch is True

<b>GAN Parameters</b>
- Lambda [10] - (Integer) Lambda value used for gradient penalty in disc loss
- dynamic_n [False] - (Boolean) True to dynamically change the number of times to train the models. False otherwise
- Lambda_n [1] - (Integer) (Only used if dynamic_n is True) Amount to scale the generator over the discriminator to give the generator a higher weight (when \>1) or the discriminator a higher weight (when \<1)
- HideAfterEnd [False] - (Boolean) True to hide any tokens after the \<END\> token in the discriminator MHA with a mask, False to keep these tokens visible
  
As the model trains, save models and graphs of the model progress will be saved to the saveDir directory. Additionally, the loss of each of the models will be printed to the terminal.
  
  
# Running The Model
  
When a model has been trained, it can be loaded in to make some sentence predictions and generate new sentences. To do so, run the following command in the root of the repo:
  
`python -m src.GANs.generate --loadDir [loadDir] --loadDefFile [loadDefFile] --loadFile [loadFile]`

Where each parameter in [] is changed to the desired value. For example, I could load in a generator model named "gen_model - 15000.pkl" with the following line:

`python -m src.GANs.generate --loadDir "models/Model1/" --loadDefFile "gen_model_def.json" --loadFile "gen_model - 15000.pkl"`
  
  

This file has the following parameters:

<b>Required:</b>
- loadDir - (String) Path to load the model from
- loadDefFile - (String) File with generator defaults
- loadFile - (String) File to load the generator model from

<b>Optional:</b>
- batchSize [1] - (Integer) Number of sentences to generate at a time
- device [gpu] - (Integer) Device used to generate sentences with

Running this script will take a second or two and will print out the generated sentence along with the length of that sentence to the terminal.



# Model Results

The model resulted in output that looks better than random text and kind of has a sentence-like structure, but overall, the model didn't produce sentences that made any sense.

In the `models/` directory, you will find pretrained models. Model1 used a smaller discriminator while Model2 used a larger discriminator. The results when varying the generator ended with a poor model, so I didn't include those in the repo.

Along with the pretrained models, two graphs will be generated. One graph is called `trainGraph.png`. This graph is the loss values of the different models throughout training. The other graph is called `MDGraph.png` which displays the mean difference between the real and fake sentences. This value will never be 0, but we want this value to decrease over time, meaning the fake sentences are looking more like real sentences.


Here are some samples for model 1 when using the `gen_model - 10000.pkl` model:
```
swede in the has at in is more in at rational from people how and for and disciplines for
for if said in leaders from be to have and the do to with the for for to from
the the of is of the should was and at is which with the to to for their
has has and not to jo by was the children from from a of and for mcwilliams for have for clenched in quai hylton vapid kalkilya a reliant peltack a khazali he far
the in has made that him by and not conover not from dishearten not for and for the whitby subdivisions in pearl spaceflight trace
that his has and cartridges state the has have from to from him be not disciplines for the the for the for expressly the the are grounds furnell delingpole for play losties peltack the for the
```

Here are the training and mean difference graphs for this model:

<p align="center">
  <img src="https://github.com/gmongaras/GAN_TextGen/blob/main/models/Model1/trainGraph.png" alt="model 1 train graph" width="400"/>
  <img src="https://github.com/gmongaras/GAN_TextGen/blob/main/models/Model1/MDGraph.png" alt="model 1 mean difference graph" width="400"/>
</p>


Here are some samples for model 2 when using the `gen_model - 8000.pkl` model:
```
all choices was was all the the guberti by with have bouncer then was care the
the was was all caley investigative just greentech joined its was by up the byddai hudur on with incomptent information chalones was it was hanning by by to
was the was was was many exhibited game on with kikuyu the later the on on to that have scolding with msnbc its with ratings fitt 500gb cacao the brdo
ngs fitt 500gb cacao the brdothe was birgit will ppds viejo was ndiayes first was that by hospital
alfalfa was was was was yestreday have without just aqsa the was to tippetts have tippetts rompuy plagiarized septicaemia to vulgarity dominant will the
this was was was was its care the was yesterday emperor against
```

Here are the training and mean difference graphs for this model:

<p align="center">
  <img src="https://github.com/gmongaras/GAN_TextGen/blob/main/models/Model2/trainGraph1.png" alt="model 2 train graph" width="400"/>
  <img src="https://github.com/gmongaras/GAN_TextGen/blob/main/models/Model2/trainGraph2.png" alt="model 2 train graph continued" width="400"/>
</p>
<p align="center">
  <img src="https://github.com/gmongaras/GAN_TextGen/blob/main/models/Model2/MDGraph1.png" alt="model 2 mean difference graph" width="400"/>
  <img src="https://github.com/gmongaras/GAN_TextGen/blob/main/models/Model2/MDGraph2.png" alt="model 2 mean difference graph continued" width="400"/>
</p>


Here is an example of a failed model:
```
1. cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside cragside
2. olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur olafur
```

Here is an example graph of a failed model:

<p align="center">
  <img src="https://user-images.githubusercontent.com/43501738/200350064-91a6940c-8ee6-4204-a4c5-11ffe4f39942.png" alt="failed model train graph" width="400"/>
  <img src="https://user-images.githubusercontent.com/43501738/200349762-d246b352-0e4a-46b2-b5ab-e9c3e422fbca.png" alt="failed model mean difference graph" width="400"/>
</p>

Notice how the train graph converges on a large negative number and the mean difference graph stays at the max value meaning the fake sentences are as far away from the real sentences as possible. So, the model isn't learning at all.


# Model Architecture

This section of the README goes over how the model works in detail. Before going into how the models work, there is some notation I am using to make it less cluttered:
- N - Batch size
- S - Sequence length
- E - Encoding size for each word/token
- V - Vocab size
- <i>N</i>(0, 1) - Represents a normal distribution with a mean of 0 and a variance of 1, but this can actually be any distribution we want to generate to model.
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
   - I decided to go with the third option as it seemed to help the model generate better sentences with a high variety.
3. Positional encodings are added to the sampled noise
4. The sampled noise with positional encodings is sent through <i>B</i> number of generator backbone blocks with the following steps to generate an encoded form of the noise:
   1. Apply self-multihead attention to the input noise
   2. Apply layer norm and a skip connection
   3. Add noise of shape (N, 1, E) to the current embeddings
      - Quick note: I add noise here to as I wanted to give the model noise that it didn't have too much control over. There isn't any mathematical reason, I just thought it would help the model produce a higher variety of output.
   4. Use cross attention to add the global noise to the embeddings
   5. Apply layer normalization and a skip connection from the embeddings after the direct noise was added to the sequence
   6. Apply a feed-forward layer
   7. Apply layer normalization and a skip connection from the embeddings before the feed-forward layer
5. The embeddings are now sent through <i>O</i> number of output blocks with the following structure:
   1. Apply self-multihead attention to the input embeddings
   2. Apply layer norm and a skip connection
   3. Apply a feed forward layer
   4. Apply layer norm and a skip connection
   - Note: The model has two types of blocks: one that adds noise and one that does not. The idea is that I wanted the model to encode the representation of the sentence and word through the noise blocks and then generate the next word prediction from those encodings using the second blocks without noise.
6. Slice the output of the generator to get the last token in the sequence. This token is what will be used to predict the next word. The rest of the sequence is not needed.
7. Apply a linear layer and a softmax layer to transform the embeddings from (N, 1, E) to (N, 1, V) to get the probabilities for the next word in the sequence
8. Save the current output from step 6 <b>while retaining the gradient of the current output</b> for the discriminator so that the generator can be updated from the discriminator.
9. Apply the argmax function and a one-hot encoding function to the output of step 7.
   - The output of these functions retains the shape of the output of step 7: (N, 1, V)
   - These functions encode the words to a one-hot sparse representation of each word as opposed to a probability distribution.
   - These functions also break the gradient computation for the generator, so the generator output is not updated as a sequence of sequential outputs, rather it is updated as a sequence of outputs generated at the same time.
   - Why did I do this? There are two main reasons for doing so:
     - I didn't want the generator to adapt its output of one token to help it generate the next token. I was afraid the generator may change how it was generating say the first token to help generate the second token as the first token directly affects the second token and the generator would be able to change the first token to help it generate the second. Breaking the gradient graph forces it to learn individual tokens without updating the others, but still allows it to learn how one token should come after the other.
     - Note: The sequence information is still retained throughout the generator output. It still has knowledge of previous outputs, it just can't update those previous outputs directly based on the future outputs.
10. Using the currently generated. Repeat from step 2 until there are a total of <i>S</i> word outputs.
11. When there are <i>S</i> word outputs, concatenate the outputs from step 6 into an embedded sequence of shape (N, S, V) for the discriminator to handle.
     - I wanted the generator to model sparse one-hot distributions rather than having to deal with probability distributions.


## Discriminator

Below is a diagram of the discriminator model

<p align="center">
  <img src="https://github.com/gmongaras/PyTorch_TextGen/raw/main/Model_Diagrams/Discriminator.png" alt="Discriminator" width="400"/>
</p>

The discriminator is a regressive model (not a classification model as it's based on the WGAN loss function) that scores a sequence based on how real it "thinks" it is. Like with the generator, let's step through how the discriminator scores a sequence:
1. The input into the model is either:
   1. A sequence of real sentences. These sentences must first be encoded to a vector form.
   2. A sequence of fake sentences from the generator which are already in the correct form.
2. The input sequences are fed into a linear layer to project the tokens from the vocab size, <i>V</i>, to the embedding size, <i>E</i>
3. Positional encodings of shape (N, S, E) are concatenated to the end of the sequences along the word embedding dimension to produce a tensor of shape (N, S, 2E).
   - Note: Unlike with the generator, the embeddings are not directly added to the embeddings. I found that adding the positional encodings to the word embeddings caused the discriminator to have trouble learning the format of a sequence. This makes sense as adding positional encodings to a sequence changes how each word is embedded which changes how it is represented in the embedding space and may lead the embeddings to represent something different than the word it actually should represent.
4. The embeddings are now sent through <i>B</i> number of backbone blocks with the following structure:
   1. Apply self-multihead attention to the input embeddings
   2. Apply layer norm and a skip connection
   3. Apply a feed forward layer
   4. Apply layer norm and a skip connection
5. Class tokens of shape (N, 1, 2E) are concatenated to the sequences to produce a tensor of shape (N, S+1, 2E). These tokens will be used to score each sequence.
6. The embeddings with the class token are now sent through <i>O</i> number of output blocks with the following structure:
   1. Apply self-multihead attention to the input embeddings
   2. Apply layer norm and a skip connection
   3. Apply a feed forward layer
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

I am not going to go into detail into how these loss functions work as there are many articles that explain how they work in great detail.

## Length Estimation Failure

I want to add this section as most of my time spent was trying to make the model generate its own length in some way or another. Why would I want to do this? I thought that having the model generate its own length would help it generate better sentences as opposed to having the model learn how to pad a sentence. My initial attempt masked the \<PAD\> tokens, but this immediately led to divergence in the discriminator. From what I remember, I tried the following (and probably much more):
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
