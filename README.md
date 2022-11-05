# Contents
- [Description](#description)
- [Requirements](#requirements)
- [Cloning This Repo](#cloning-the-repo)
- [Creating a Vocabulary](#creating-a-vocabulary)
- [Training a Model](#training-a-model)
- [Running the Model](#running-the-model)
- [Model Architecture](#model-architecture)


# Description
This project started as a side project for for my [internship at Meta](https://github.com/gmongaras/MetaU_Capstone), but became a project I kind of became passionate in. Originally, I just wanted to generate text and I wanted to see if I could do so using a GAN. I was somewhat able to, but I wanted to try to make it better. Unfortunately, even though I tried for a while, I ran into many convergence issues, but decided to

The current project is my end result using GANs which resulted in text that looks better than random, but not entirely like a real sentence. Though, it's still fun to see what the generator produces.

# Requirements

This project was created using `python 3.9.12` but other versions should work too.

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

;



# Model Architecture

This section of the README goea over how the model works in detail. Before going into how the models work, there is some notation I am using to make it less cluttered:
- N - Batch size
- S - Sequence length
- E - Encoding size for each word/token
- <i>N</i>(0, 1) - Represents a normal distribution with a mean of 0 and a variance of 1, but this can actually be any distribution we want to generator to model.
- Feed Forward - This layer is not a single linear layer, rather it is a linear layer projecting the data from E to a higher dimensional space, an activation like GELU, then another linear layer projecting the data back to E.

## Generator

Below is a diagram of the generator model
<img src="https://github.com/gmongaras/PyTorch_TextGen/raw/main/Model_Diagrams/Generator.png" alt="Generator" width="1000"/>

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
   3. Add noise of shape (N, S, E) to the current embeddings
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
9. Apply the argmax function and a one-hot encoding function to the output of step 6.
   - The output of these functions retains the shape of the output of step 6: (N, 1, V)
   - These function encode the
   - These functions also break the gradient computation for the generator, so the generator output is not updated sequence of sequential outputs, rather it is updated as a sequence of outputs generated at the same time.
   - Why did I do this? There is one main reason for doing so:
     - I didn't want to generator to adapt it's output of one token to help it generate the next token. I was afraid the generator may change how it was generating say the first token to help generate the second token as the first token directly effects the second token and the generator would be able to change the first toke nto help it generate the second. Breaking the gradient graph forces it to learn individual tokens without updating the others, but still allows it to learn how one token should come after the other.
     - Note: The sequence information is still retained throughout the generator output. It still have knowledge of previous outputs, it just can't update those previous outputs based on the future outputs.
10. Using the currently generated. Repeat from step 2 until there are a total of <i>S</i> word outputs.
11. When there are <i>S</i> word outputs, concatentate the outputs from step 6 into an embedded sequence of shape (N, S, V) for the discriminator to handle.


# PyTorch_TextGen
A basic text generation model for a project I'm working on


Documentation/Log for GANs:
https://docs.google.com/document/d/1UA-dEF-IMr5-mOgkp6ORRtAERDEprKHw3K_Lw9A-vmU/edit?usp=sharing

Documentation/Log for networks that aren't GANs:
https://docs.google.com/document/d/1_2mnS_jRa5RixBPNazFUxKrX-_Y-Bz9zZHglUnGHpgo/edit?usp=sharing




# Problems and Solutions
- Problem: The output of the Generator was of shape (N, sequence length) where each batch sentence was the argmax of the softmax output. The problem is the argmax is not differentiable and a lot fo information is being lost with the argmax conversion. So, the generator isn't able to learn.
  - Solution: Instead of taking the argmax, the softmax outputs of the generator is directly used by the discriminator. So, the output of the generator is (N, sequence length, vocab size). The discriminator takes this output as input and uses a few linear layers to convert the tensor to the shape which was previously used of (N, sequence length, embedding size).
- Problem: The noise transformation in the generator was a heavy process that transformed a large latent space of data. The input used transformer blocks and MHA which is a lot of FLOPS (floating point operations) and very expensive. Additionally, since the output was 2-d, the generator had a hard time finding a good mapping for this noise.
  - Solution: Instead of trying to get 2-d noise from the generator, we can get 1-d and broadcast it to 2-d. So, we get N noise vector of size S (sequence length) and apply a few Linear transformations on that data. The output will also be NxS. Then, we broadcast the noise vector of size S by E (embedding size) so that the output is NxSxE. So now there are E number of noise vectors. This is the needed shape for the generator and since the embedding space is 1-d instead of 2-d, it seems to handle the noise a lot better.



Note: Data from the following README:
- Random Text: https://www.kaggle.com/datasets/kashnitsky/hierarchical-text-classification
- Billion word language modeling benchmark: https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark
- Fortunes 1: https://github.com/ruanyf/fortunes
- Fortunes 2: http://www.fortunecookiemessage.com/
