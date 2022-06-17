# PyTorch_TextGen
A basic text generation model for a project I'm working on




# Problems and Solutions
- Problem: The output of the Generator was of shape (N, sequence length) where each batch sentence was the argmax of the softmax output. The problem is the argmax is not differentiable and a lot fo information is being lost with the argmax conversion. So, the generator isn't able to learn.
  - Solution: Instead of taking the argmax, the softmax outputs of the generator is directly used by the discriminator. So, the output of the generator is (N, sequence length, vocab size). The discriminator takes this output as input and uses a few linear layers to convert the tensor to the shape which was previously used of (N, sequence length, embedding size).



Note: Data from the following README:

https://github.com/ruanyf/fortunes
