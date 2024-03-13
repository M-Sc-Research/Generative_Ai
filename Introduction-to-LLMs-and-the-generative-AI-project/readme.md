## Generative AI is a subset of traditional machine learning. 
<br>


> Large language models have been trained on trillions of words over many weeks and months and with large amounts of computing power. These foundation models, as we call them, with billions of parameters, exhibit emergent properties beyond language alone, and researchers are unlocking their ability to break down complex tasks, reason, and problem-solve.

<br>
The more parameters a model has, the more memory, and as it turns out, the more sophisticated the tasks it can perform. By either using these models as they are or by applying fine-tuning techniques to adapt them to your specific use case, you can rapidly build customized solutions without the need to train a new model from scratch.<br>

> The way you interact with language models is quite different than other machine learning and programming paradigms. In those cases, you write computer code with formalized syntax to interact with libraries and APIs.
<br>

> In contrast, large language models can take natural language or human written instructions and perform tasks much as a human would. The text that you pass to an LLM is known as a **prompt**.

<br>
The space or memory that is available to the prompt is called the context window, and this is typically large enough for a few thousand words but differs from model to model.
<br>

In this example, you ask the model to determine where Ganymede is located in the solar system. The prompt is passed to the model, the model then predicts the next words, and because your prompt contained a question, this model generates an answer. The output of the model is called a **completion**, and the act of using the model to generate text is known as **inference**.

<br>
<br>

> The completion is comprised of the text contained in the original prompt, followed by the generated text. You can see that this model did a good job of answering your question. It correctly **identifies** that Ganymede is a moon of Jupiter and generates a reasonable answer to your question stating that the moon is located within Jupiter's orbit.
<br>

 You can use LLMs to carry out smaller, focused tasks like **information retrieval**. In this example, you ask the model to identify all of the people and places identified in a news article. This is known as named entity recognition, a word classification.

 <br>
 <br>
 
  The understanding of knowledge encoded in the model's parameters allows it:
  1. to correctly carry out this task
  2. return the requested information
  3.  Finally, an area of active development is augmenting LLMs by connecting them to external data sources or using them to invoke external APIs.
  4.   You can use this ability to provide the model with information it doesn't know from its pre-training and to enable your model to power interactions with the real world.
  <br>
  
 > Developers have discovered that as the scale of foundation models grows from hundreds of millions of parameters to billions, even hundreds of billions, the subjective understanding of language that a model possesses also increases. This language understanding stored within the parameters of the model is what processes, reasons, and ultimately solves the tasks you give it, but it's also true that smaller models can be fine-tuned to perform well on specific focused tasks.

 <br>
 
 <p>
	 
It's important to note that generative algorithms are not new. Previous generations of language models made use of an architecture called **recurrent neural networks** or RNNs. RNNs while powerful for their time, were limited by the amount of compute and memory needed to perform well at generative tasks.
</p>

example of an RNN:
- carrying out a simple next-word prediction generative task
-  With just one previous word seen by the model, the prediction can't be very good.
-   As you scale the RNN implementation to be able to see more of the preceding words in the text, you have to significantly scale the resources that the model uses.
<br>

- As for the prediction, the model failed here. Even though you scale the model, it still hasn't seen enough of the input to make a good prediction.
-  To successfully predict the next word, models need to see more than just the previous few words.
- Models need to have an understanding of the whole sentence or even the whole document.
-  The problem here is that language is complex.
-   In many languages, one word can have multiple meanings. These are homonyms.

> Transformers:

- can be scaled efficiently to use multi-core GPUs
-  it can parallel process input data
-   making use of much larger training datasets
-    it's able to learn to pay attention to the meaning of the words it's processing. And attention is all you need. 


<br>

1. The Transformer model uses self-attention to compute representations of input sequences, which allows it to capture long-term dependencies and parallelize computation effectively.
2.  The authors demonstrate that their model achieves state-of-the-art performance on several machine translation tasks and outperforms previous models that rely on RNNs or CNNs.

			


<br>







 > The attention map:
 can be defined as the weights assigned to each input feature when computing the context vector. ex:  can be useful to illustrate the attention weights between each word and every other word.


>  Self-attention: is a mechanism that allows the model to capture relationships between different positions of the input sequence. Self-attention is particularly useful in capturing long-range dependencies and identifying important elements within the sequence.


 self-attention and the ability to learn a tension in this way across the whole input significantly approve the model's ability to encode language. 

![Screenshot from 2024-03-08 01-51-14](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/5d882aeb-8194-4e92-9fe3-8c7e7b275da8)


The transformer architecture is split into two distinct parts:
1. the encoder
2.  the decoder
 These components work in conjunction with each other and they share several similarities.

 ![Screenshot from 2024-03-08 01-40-26](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/c9878c6e-eb32-41fe-8f34-0d7f6e0e824f)

 
 1. tokenize the words. Simply put, this converts the words into numbers
 2. with each number representing a position in a dictionary of all the possible words that the model can work with
 3. WE can choose from multiple tokenization methods. For example, token IDs matching two complete words, or using token IDs to represent parts of words.
 4.   What's important is that once you've selected a tokenizer to train the model, you must use the same tokenizer when you generate text.
 5.   Now that your input is represented as numbers, you can pass it to the embedding layer. This layer is a trainable vector embedding space, a high-dimensional space where each token is represented as a vector and occupies a unique location within that space.
 6.    Each token ID in the vocabulary is matched to a multi-dimensional vector, and the intuition is that these vectors learn to encode the meaning and context of individual tokens in the input sequence.
 7. Embedding vector spaces have been used in natural language processing for some time, previous generation language algorithms like Word2vec use this concept.
 8. if you imagine a vector size of just three, you could plot the words into a three-dimensional space and see the relationships between those words.
 9. As you add the token vectors into the base of the encoder or the decoder, you also add positional encoding.
 10. The model processes each of the input tokens in parallel.
 11.  Once you've summed the input tokens and the positional encodings, you pass the resulting vectors to the self-attention layer.
 12.   Here, the model analyzes the relationships between the tokens in your input sequence. As you saw earlier, this allows the model to attend to different parts of the input sequence to better capture the contextual dependencies between the words.
 13.   The self-attention weights that are learned during training and stored in these layers reflect the importance of each word in that input sequence to all other words in the sequence. But this does not happen just once, the transformer architecture actually has multi-headed self-attention.
 14.    This means that multiple sets of self-attention weights or heads are learned in parallel independently of each other.
 15. The number of attention heads included in the attention layer varies from model to model, but numbers in the range of 12-100 are common.
 16.  The intuition here is that each self-attention head will learn a different aspect of language.
 17.  The weights of each head are randomly initialized and given sufficient training data and time, each will learn different aspects of language.
 18.    Now that all of the attention weights have been applied to your input data, the output is processed through a fully connected feed-forward network.
 19. The output of this layer is a vector of logits proportional to the probability score for each and every token in the tokenizer dictionary. You can then pass these logits to a final softmax layer, where they are normalized into a probability score for each word.
 20.  This output includes a probability for every single word in the vocabulary, so there are likely to be thousands of scores here.
 21.   One single token will have a score higher than the rest. This is the most likely predicted token.


<br>

example: translate

![Screenshot from 2024-03-08 02-30-07](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/fb8fa72e-01da-4dd9-bf6f-45e49bf525eb)

## Transformer:
![Screenshot from 2024-03-08 02-32-54](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/8e6ed441-a729-4c5b-bb53-ed75a9242751)

<br>

![Screenshot from 2024-03-08 02-35-09](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/956e1aae-5893-4648-88bc-3937de3229ed)

<br>

The Transformer model uses self-attention to compute representations of input sequences, which allows it to capture long-term dependencies and parallelize computation effectively.
<br>

<strong>
The Transformer architecture consists of an encoder and a decoder, each of which is composed of several layers. Each layer consists of two sub-layers: a multi-head self-attention mechanism and a feed-forward neural network. The multi-head self-attention mechanism allows the model to attend to different parts of the input sequence, while the feed-forward network applies a point-wise fully connected layer to each position separately and identically. 
The Transformer model also uses residual connections and layer normalization to facilitate training and prevent overfitting.
</strong>


<br>

## Prompting and prompt engineering

**The text that you feed into the model is called the prompt, the act of generating text is known as inference, and the output text is known as the completion.**


#### context window:

The full amount of text or the memory that is available to use for the prompt is called the context window.
<br>

#### prompt engineering:

 you'll frequently encounter situations where the model doesn't produce the outcome that you want on the first try. <br>
 You may have to revise the language in your prompt or the way that it's written several times to get the model to behave in the way that you want. <br>
 This work to develop and improve the prompt is known as prompt engineering.

 <br>

 #### in-context learning:
 
  one powerful strategy to get the model to produce better outcomes is to include examples of the task that you want the model to carry out inside the prompt. Providing examples inside the context window is called in-context learning. 
  
  <br>
  
  With in-context learning, you can help LLMs learn more about the task being asked by including examples or additional data in the prompt. 


  <br>
  
#### zero-shot inference:

Zero-shot inference in Language Model (LLM) refers to the ability of the model to generate outputs for tasks or prompts that it has not been explicitly trained on. In other words, the model can perform a task without any specific training examples or fine-tuning for that task. This is achieved by leveraging the general knowledge and language understanding capabilities learned during pre-training.
<br>
In zero-shot inference, the model is provided with a task description or prompt, and it generates an output based on its understanding of the language and the context provided. This is made possible by the large-scale pre-training of LLMs on diverse text corpora, which enables them to generalize to new tasks or prompts.
<br>
Zero-shot inference is a powerful capability of LLMs as it allows them to adapt to new tasks without the need for extensive retraining, making them versatile and efficient in handling a wide range of natural language processing tasks.
<br>



<br>

The largest of the LLMs are surprisingly good at this, grasping the task to be completed and returning a good answer. <br>
Smaller models, on the other hand, can struggle with this.<br>

#### one-shot inference

One-shot inference refers to the ability of a model to make accurate predictions or classifications based on a single input or a small number of examples. In other words, the model can generalize well and make reliable decisions without requiring a large amount of training data.
<br><br>

Achieving one-shot inference typically requires the model to have strong generalization abilities, robust feature representation, and effective learning mechanisms. Techniques such as transfer learning, few-shot learning, and meta-learning can help improve a model's one-shot inference performance by leveraging information from related tasks or domains.
<br><br>
Overall, one-shot inference is a desirable trait in machine learning models as it enables efficient and accurate decision-making with minimal data requirements.
 <br><br>

 #### few-shot inference
 
 Sometimes a single example won't be enough for the model to learn what you want it to do. So you can extend the idea of giving a single example to include multiple examples. This is known as few-shot inference. 
 <br><br>
 
  So to recap, you can engineer your prompts to encourage the model to learn by example. While the largest models are good at zero-shot inference with no examples, smaller models can benefit from one-shot or few-shot inference that includes examples of the desired behavior. But remember the context window because you have a limit on the amount of in-context learning that you can pass into the model.
  <br><br>
  
  Generally, if you find that your model isn't performing well when, say, including five or six examples, you should try fine-tuning your model instead. 
 <br><br> 
  Fine-tuning performs additional training on the model using new data to make it more capable of the task you want it to perform.
  <br><br>
  
  As larger and larger models have been trained, it's become clear that the ability of models to perform multiple tasks and how well they perform those tasks depends strongly on the scale of the model.
  <br>
  
   > Models with more parameters can capture more understanding of language. The largest models are surprisingly good at zero-shot inference and can infer and complete many tasks that they were not specifically trained to perform. In contrast, smaller models are generally only good at a small number of tasks. Typically, those similar to the task they were trained on. You may have to try out a few models to find the right one for your use case. Once you've found the model that is working for you, there are a few settings that you can experiment with to influence the structure and style of the completions that the model generates.

<br><br>

### Greedy vs. random sampling

Most large language models by default will operate with so-called **greedy decoding**. This is the simplest form of next-word prediction, where the model will always choose the word with the highest probability. This method can work very well for short generations but is susceptible to repeated words or repeated sequences of words. If you want to generate text that's more natural, more creative, and avoids repeating words, you need to use some other controls. **Random sampling** is the easiest way to introduce some variability. Instead of selecting the most probable word every time with random sampling, the model chooses an output word at random using the probability distribution to weight the selection.

<br>





![Screenshot from 2024-03-08 23-53-42](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/4a9320f5-22b1-45da-bc09-355273f923d5)


<br><br>

## Two Settings, top p, and top k are sampling techniques that we can use to help limit the random sampling:

![Screenshot from 2024-03-08 23-59-02](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/2b7c706f-9f08-456e-9a24-c23669a42ee5)

<br>

![Screenshot from 2024-03-09 00-00-02](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/f620692f-52ae-4edb-a98a-100b6d7a2080)


<br><br>

#### temperature:
One more parameter that you can use to control the randomness of the model output is known as temperature. <br>
This parameter influences the shape of the probability distribution that the model calculates for the next token. Broadly speaking, the higher the temperature, the higher the randomness, and the lower the temperature, the lower the randomness. 

<br>


In contrast to the top k and top p parameters, changing the temperature alters the predictions that the model will make. If you choose a low value of temperature, say less than one, the resulting probability distribution from the softmax layer is more strongly peaked with the probability being concentrated in a smaller number of words.

<br><br>
If instead you set the temperature to a higher value, say, greater than one, then the model will calculate a broader flatter probability distribution for the next token.
<br>

> This leads the model to generate text with a higher degree of randomness and more variability in the output compared to a cool temperature setting. This can help you generate text that sounds more creative. If you leave the temperature value equal to one, this will leave the softmax function as default and the unaltered probability distribution will be used.
<br><br>

**Temperature is used to affect the randomness of the output of the softmax layer. A lower temperature results in reduced variability while a higher temperature results in increased randomness of the output.**

![Screenshot from 2024-03-09 01-28-30](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/683005ea-8868-49f8-9f63-2727c44c7da2)



<br><br>

 - LLMs are capable of carrying out many tasks, but their abilities depend strongly on the size and architecture of the model. 





## How large language models are trained

##### Considerations for choosing a model
1. foundation model: pre-trained model
2. train your model: custom LLM
<br>

#### Pre-training

- LLMs encode a deep statistical representation of language:
	-  This understanding is developed during the model's pre-training phase when the model learns from vast amounts of unstructured textual data. This can be gigabytes, terabytes, and even petabytes of text. (This data is pulled from many sources, including scrapes off the Internet and corpora of texts that have been assembled specifically for training language models.)
 -  In this self-supervised learning step, the model internalizes the patterns and structures present in the language.
 -  These patterns then enable the model to complete its training objective, which depends on the architecture of the model.
 -  During pre-training, the model weights get updated to minimize the loss of the training objective. 
 -  The encoder generates an embedding or vector representation for each token.
 -  Pre-training also requires a large amount of computing and the use of GPUs.
 -  when you scrape training data from public sites such as the Internet, you often need to process the data to increase quality, address bias, and remove other harmful content.
 -  As a result of this data quality curation, often only 1-3% of tokens are used for pre-training. (You should consider this when you estimate how much data you need to collect if you decide to pre-train your model.)


![Screenshot from 2024-03-09 02-57-56](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/33727850-ab81-4cb9-8331-7556cb8817f8)



<br><br>

 ## Three variances of the transformer model;
 1. encoder-only
 2. encoder-decoder models
 3. decoder-only

 <br>
 
 ### Encoder-only (Autoencoding models)
 - they are pre-trained using masked language modeling.<br>
 Here, tokens in the input sequence or randomly masked, and the training objective is to predict the mask tokens to reconstruct the original sentence. 
 This is also called a **denoising objective**.<br>
 
 ![Screenshot from 2024-03-09 03-11-58](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/833f205c-ed8e-44ce-83b2-fdb451f05b4f)

 - Autoencoding models spill bi-directional representations of the input sequence, meaning that the model has an understanding of the full context of a token and not just of the words that come before.
 - Encoder-only models are ideally suited to tasks that benefit from this bi-directional context. You can use them to carry out sentence classification tasks, for example, sentiment analysis or token-level tasks like named entity recognition or word classification.
 -  Some well-known examples of an autoencoder model are BERT and RoBERTa.
<br>
<br>

### Decoder-only(autoregressive models)

- they are pre-trained using causal language modeling.<br>
Here, the training objective is to predict the next token based on the previous sequence of tokens. Predicting the next token is sometimes called **full language modeling** by researchers.<br><br>
![Screenshot from 2024-03-09 03-38-55](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/5dc3e71c-fbf1-4f62-93ba-20b8412e1572)

- Decoder-based autoregressive models, mask the input sequence and can only see the input tokens leading up to the token in question.
-  The model does not know the end of the sentence. The model then iterates over the input sequence one by one to predict the following token.
-   In contrast to the encoder architecture, this means that the context is unidirectional.
-    By learning to predict the next token from a vast number of examples, the model builds up a statistical representation of language.
-  Models of this type make use of the decoder component of the original architecture without the encoder.
-   Decoder-only models are often used for text generation, although larger decoder-only models show strong zero-shot inference abilities, and can often perform a range of tasks well.
-   Well-known examples of decoder-based autoregressive models are GBT and BLOOM.


### encoder-decoder models(sequence-to-sequence model)
- The exact details of the pre-training objective vary from model to model.
- A popular sequence-to-sequence model T5, pre-trains the encoder using span corruption, which masks random sequences of input tokens.<br>
 Those mass sequences are then replaced with a unique Sentinel token, shown here as x. Sentinel tokens are special tokens added to the vocabulary, but do not correspond to any actual word from the input text.<br><br>
 ![Screenshot from 2024-03-09 03-40-30](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/7875cfe0-0868-4c52-9adc-a30474302ca6)

- The decoder is then tasked with reconstructing the mask token sequences auto-regressively.
- The output is the Sentinel token followed by the predicted tokens.
- You can use sequence-to-sequence models for translation, summarization, and question-answering.
- They are generally useful in cases where you have a body of texts as both input and output.
- Besides T5, which you'll use in the labs in this course, another well-known encoder-decoder model is BART, not bird.

   To summarize, here's a quick comparison of the different model architectures and the targets of the pre-training objectives.
### recap of transformer models:
>  Autoencoding models are pre-trained using masked language modeling. They correspond to the encoder part of the original transformer architecture and are often used with sentence classification or token classification.
>  Autoregressive models are pre-trained using causal language modeling. Models of this type make use of the decoder component of the original transformer architecture and are often used for text generation.
>  Sequence-to-sequence models use both the encoder and decoder part of the original transformer architecture. The exact details of the pre-training objective vary from model to model. The T5 model is pre-trained using span corruption. Sequence-to-sequence models are often used for translation, summarization, and question-answering.
<br><br>
![Screenshot from 2024-03-09 03-41-45](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/c44cebd7-0f46-42fb-972c-4a6d29f99333)

**Researchers have found that the larger a model, the more likely it is to work as you needed to without additional in-context learning or further training.**




<br>
<br>
<br>

### Computational challenges of training LLMs

1. running out of memory

  
#### CUDA(Compute Unified Device Architecture)
is a collection of libraries and tools developed for Nvidia GPUs. Libraries such as PyTorch and TensorFlow use CUDA to boost performance on metrics multiplication and other operations common to deep learning.


2. You'll encounter these out-of-memory issues because most LLMs are huge, and require a ton of memory to store and train all of their parameters.
<br>

![Screenshot from 2024-03-09 18-13-18](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/d75a7d2d-d757-4255-a67d-f69e3530d155)

<br>

3. If you want to train the model, you'll have to plan for additional components that use GPU memory during training.

![Screenshot from 2024-03-09 18-16-34](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/aba70ecb-4899-4461-b2a1-6b276b78a3d1)

<br><br>

4. This can easily lead to 20 extra bytes of memory per model parameter. To account for all of these overheads during training, you'll require approximately 6 times the amount of GPU RAM that the model weights alone take up. To train a one billion parameter model at 32-bit full precision, you'll need approximately 24 gigabytes of GPU RAM. This is too large for consumer hardware, and even challenging for hardware used in data centers if you want to train with a single processor.


### Reduce the memory required for training

1. Quantization: is to reduce the memory required to store and train models by reducing the precision of the model weights
data types used in deep learning framework:

- FP32 for 32-bit full position: The range of numbers you can represent with FP32 goes from approximately -3*10^38 to 3*10^38. By default, model weights, activations, and other model parameters are stored in FP32. 
<br>
<br>

![Screenshot from 2024-03-09 18-47-03](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/97f399d3-401e-40e2-91c8-6055b556eca5)

2.  FP16, or Bfloat16 for 16-bit half precision
3.  int8 eight-bit integers
   
**Quantization statistically projects the original 32-bit floating point numbers into a lower precision space, using scaling factors calculated based on the range of the original 32-bit floating point numbers.**

FP32: Floating point numbers are stored as a series of bits zeros and ones. The 32 bits to store numbers in full precision with FP32 consist of one bit for the sign where zero indicates a positive number and one for a negative number. Then eight bits for the exponent of the number, and 23 bits represent the fraction of the number.


## FP32 to FP16

![Screenshot from 2024-03-09 18-53-39](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/6c965a8d-5ea4-440f-bb13-035f192509ee)

> BFLOAT16 uses the full eight bits to represent the exponent but truncates the fraction to just seven bits. This saves memory and increases model performance by speeding up calculations. The downside is that BF16 is not well suited for integer calculations, which are relatively rare in deep learning.

## FP32 to BFLOAT16(Brain Floating Point Format)

![Screenshot from 2024-03-09 18-56-59](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/b9daf7de-799d-40b3-9da9-f00ac95b73de)

## FP32 into INT8

![Screenshot from 2024-03-09 18-59-21](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/9a95f163-5ac3-48d2-9e3b-a3518a6fb6e8)



<br>
<br>


![Screenshot from 2024-03-10 14-29-17](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/0799446d-92d8-4ace-8d1c-0dafa2dc2068)


<br>

### 1. Why do we use multiple GPUs?

you'll need to use multi-GPU computing strategies when your model becomes too big to fit in a single GPU. But even if your model does fit onto a single GPU, there are benefits to using multiple GPUs to speed up your training. 
<br>
### 2. how you can carry out this scaling across multiple GPUs in an efficient way?
<br>

- considering the case where your model still fits on a single GPU
1. scaling model training to distribute large data sets across multiple GPUs and process these batches of data in parallel
	- DDP(Distributed Date Parallel): DDP copyists your model onto each GPU and sends batches of data to each of the GPUs in parallel. Each data set is processed in parallel and then a synchronization step combines the results of each GPU, which in turn updates the model on each GPU, which is always identical across chips.=>results in faster training
	  ![Screenshot from 2024-03-12 02-00-46](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/f7480eb1-39b2-41a8-856c-0bb87c3b51ec)

	- model sharding => A popular implementation of modal sharding is Pi Torch is fully sharded data parallel or FSDP for short
<br>that proposed a technique called **ZeRO**.
 ZeRO stands for zero redundancy optimizer and the goal of ZeRO is to optimize memory by distributing or sharding model states across GPUs with ZeRO data overlap.
	  ![Screenshot from 2024-03-12 02-34-41](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/cf8ed4f4-fb6d-4914-9ca6-297c3b3ef80a)

   <br><br>
   ![Screenshot from 2024-03-12 02-37-08](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/9e6aef17-a3e2-4ea4-8c40-11506e608b37)

- ZeRO Stage 1, shots only optimizer states across GPUs, this can reduce your memory footprint by up to a factor of four.
-  ZeRO Stage 2 also shoots the gradients across chips. When applied together with Stage 1, this can reduce your memory footprint by up to eight times.
-  ZeRO Stage 3 shots of all components including the model parameters across GPUs.
 **When applied together with Stages 1 and 2, memory reduction is linear with a number of GPUs.**
   <br>
    For example, sharding across 64 GPUs could reduce your memory by a factor of 64.
   <br>
![Screenshot from 2024-03-12 02-43-32](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/a502787e-5fb4-4599-a009-4fcdc0be82cc)

   In contrast to GDP, where each GPU has all of the model states required for processing each batch of data available locally, FSDP requires you to collect this data from all of the GPUs before the forward and backward pass. Each CPU requests data from the other GPUs on-demand to materialize the sharded data into uncharted data for the duration of the operation. After the operation, you release the uncharted non-local data back to the other GPUs as original sharded data You can also choose to keep it for future operations during backward pass for example. Note, that this requires more GPU RAM again, this is a typical performance versus memory trade-off decision.the final step after the backward pass, FSDP synchronizes the gradients across the GPUs in the same way they were for DDP


   <br><br>

   ![Screenshot from 2024-03-12 02-46-32](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/75330e05-f299-4728-a386-b81682161fc0)

   ## Data parallelism in the context of training Large Language Models (LLMs) with GPUs
   Data parallelism is a strategy that splits the training data across multiple GPUs. Each GPU processes a different subset of the data simultaneously, which can greatly speed up the overall training time.

<br><br>

<br>

The goal during pre-training is to **maximize the model's performance of its learning objective**, which is minimizing the loss when predicting tokens. 

<br>

Two options you have to achieve better performance are
1. increasing the size of the dataset you train your model on
2. Increase the number of parameters in your model.
In theory, you could scale both of these quantities to improve performance.

<br><br>

![Screenshot from 2024-03-13 11-07-35](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/22258878-3a1c-4b04-bdef-a293431002c2)

Another factor to consider is your computing budget, which encompasses elements such as the quantity of GPUs at your disposal and the time allocated for model training. 
<br>
A petaFLOP per second day measures the floating point operations performed at a rate of one petaFLOP per second over a full day.<br>
![Screenshot from 2024-03-13 11-08-37](https://github.com/M-Sc-Research/Generative_Ai/assets/96652895/98a84775-82bf-40b7-844c-4aeecca24d49)

One petaFLOP per day is roughly equal to the computational power of eight NVIDIA V100 GPUs running at full capacity for an entire day.
<br>
If you have a more powerful processor that can carry out more operations at once, then a petaFLOP per second day requires fewer chips.
 <br>
 
> Larger models require more computing resources for training and typically need more data to achieve optimal performance.

there is a trade-off between training dataset size, model size, and compute budget. 


<br>

## Compute resource constraints
1. Hardware
2. project timeline
3. financial budget
 
### What's the ideal balance between these three quantities?
> The aim was to determine the best number of parameters and amount of training data within a specified compute budget. The Chinchilla paper suggests that numerous 100 billion parameter large language models, such as GPT-3, might be excessively parameterized, having more parameters than necessary for proficient language understanding and could benefit from increased training data. The authors theorized that smaller models could potentially match the performance of much larger ones if trained on more extensive datasets. 

#### The optimal training dataset size for a given model is about 20 times larger than the number of parameters in the model.
For a 70 billion parameter model, the ideal training dataset contains 1.4 trillion tokens or 20 times the number of parameters.




<br>

However, there's one situation where you may find it necessary to pre-train your model from scratch. <br>
If your target domain uses vocabulary and language structures that are not commonly used in day-to-day language. You may need to perform **domain adaptation** to achieve good model performance.
### Pretraining your model from scratch will result in better models for highly specialized domains like law, medicine, finance, or science.
 <br>
 recap:
  1. models are trained on vast amounts of text data during an initial training phase called pretraining.
  2. This is where models develop their understanding of language.
  3. In practice because of GPU memory limitations, you will almost always use some form of quantization when training your models. 
  4. Petaflops per second day is a useful measure for computing budget as it reflects the both hardware and time required to train the model.
To scale our model, we need to simultaneously increase dataset size and model size, as they can otherwise bottleneck each other.
