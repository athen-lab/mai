# Multilayer Authenticity Identifier (MAI)

MAI is a research project that attempts to train a machine learning model that identifies AI-generated images.

## Why?

i am bored.

## Architecture

i have experimented with the following model architectures with varying degrees of success.
they reside in their own directory.

### convolutional neural network (CNN)

the code for the CNN model is located at `./resnet`.

the CNN model looks like this:

1. 16-channel, 3x3 convolution layer -> 2x2 max pooling -> relu activation
2. 32-channel, 3x3 convolution layer -> 2x2 max pooling -> relu activation
3. 64-channel, 3x3 convolution layer -> 2x2 max pooling -> relu activation
4. 40,000-neuron layer -> relu -> 120-neuron layer -> relu -> 30 -> 1

the model expects a 200x200 image as an input and outputs a score, with 1 being that the input image is absolutely synthetic, and 0 being that it is absolutely authentic.

[BCEWithLogitLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) is used as the loss fn, and [RMSprop](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html) as the optimizer.

### vision language model (VLM)

the code for the vlm model finetuning is located at `./moondream`.

following [the Bi-LoRA paper](https://arxiv.org/abs/2404.01959) which suggests finetuning a vlm on real and ai-generated images, i decided to finetune [moondream](https://moondream.ai/), a small vlm.

the training data consists of 50% real images and 50% ai-generated images, along with a q&a pair.
the question is always "Is this image AI-generated?", followed by the answer "Yes." or "No.", depending on whether the image is ai-generated or not.

preliminary experiments show mixed results. to improve the training data, i decided to generate my own dataset using the following steps:

1. ask moondream to caption a real image.
2. use that caption to prompt a model (stable diffusion 3.5 large at the moment) to generate the equivalent ai image.
3. add the image to the row alongside the real image.

hopefully, this will help moondream better understand the difference between real and ai-generated images.

