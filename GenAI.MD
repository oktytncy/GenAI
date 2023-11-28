
## Back to Basics: Core Concepts Explained

### What is Generative AI and LLM**
Generative AI refers to the type of artificial intelligence that can generate new content based on the data it has learned from. It's like teaching a computer to be creative. For example, after showing a generative AI lots of pictures of cats, it can create a brand new image of a cat that doesn't exist. Or, if you feed it lots of music, it might be able to compose a new song in the style of what it's heard.

LLM stands for **Large Language Model**, which is a type of AI that's been trained on a huge amount of text. It's designed to understand and generate human-like text by predicting what word comes next in a sentence. Because it's read so much text, it can do a lot of cool things with words.

### LLM Use Cases and Practical Tasks

Large Language Models (LLMs) like generative AI are commonly associated with chatbot functions due to their visibility and popularity. However, the underlying next-word prediction technology enables a wide range of text generation tasks, such as essay writing, summarizing dialogues, and translation between languages or into machine code. 

Moreover, LLMs can perform specific tasks like named entity recognition, extracting information such as people and places from texts. The potential of LLMs extends to interfacing with external data sources and APIs, enhancing their capabilities beyond their initial training. 

With the growth in the scale of foundation models, the nuanced understanding and processing ability of language by these models have significantly improved, making them adept at solving complex tasks, although smaller models can also be specialized for particular functions.

### The Foundations of Text Generation

Generative algorithms have been around for some time, with earlier language models relying on recurrent neural networks (RNNs). While RNNs were advanced for their era, they required substantial computational power and memory to perform generative tasks effectively. 

An RNN that tries to predict the next word with limited previous words tends to fail because it lacks sufficient context to make accurate predictions. The complexity of the language further compounds this difficulty.

The transformative moment for generative AI came with the 2017 paper [Attention is All You Need](https://arxiv.org/abs/1706.03762), introducing the transformer architecture. 

This new model is more efficient, capable of parallel processing, and uses larger datasets, which importantly allows it to understand the meaning of words through its attention mechanism, marking a significant advance in AI capabilities.

### Transformer Architecture

Transformers are a type of language model that performs much better than older ones on language tasks. They can understand the importance of each word in a sentence by using something called attention weights. 

This lets the model figure out complex relationships.

**For example:** The teacher taught math the student with a book.

Did the teacher teach using the book or did the student have the book or was it both?

In 2017, a groundbreaking paper called **Attention Is All You Need** was published by researchers from Google and the University of Toronto. 

This introduced the transformer architecture, which has led to the advancements in AI that we see today. Transformers are efficient because they can be scaled to run on many GPU cores, allowing them to handle big datasets all at once.

Most importantly, they can learn the meaning of words, which is essential for understanding and generating language.

Self-attention is a feature that helps a model understand how words in a sentence relate to each other. For instance, the model might recognize that the word "book" is closely related to "teacher" and "student" in a sentence. This ability lets the model see the full picture of the sentence, improving how it represents and processes language.

<p align="left">
  <img src="images/1.png" alt="drawing" width="500"/>
</p>

Transformers consist of encoders and decoders that process language by first converting words into numerical forms, a step known as tokenization. 

A specific layer then interprets the meanings of these numerical values. The model also maintains the sequence of words to make sense of sentences. It employs multiple **attention heads** to focus on various elements of language, which helps it predict what word comes next with high accuracy.

#### Working Principle

<p align="left">
  <img src="images/2.png" alt="drawing" width="500"/>
</p>

Machine learning models are essentially complex statistical tools that need to work with numbers instead of words. To prepare text for these models, you have to convert words into numerical values through tokenization. 

There are different ways to tokenize, some match whole words to numbers, and some match parts of words. It's crucial to use the same tokenization method throughout the training and text generation process. 

Once words are tokenized, they're fed into an embedding layer. This layer is a place where each number (representing a word or part of a word) gets a unique vector in a high-dimensional space. These vectors help the model understand the meaning and context of words. 

This concept isn't new; it's been used in earlier language processing tools like Word2vec.

The model processes each of the input tokens in parallel. So, by adding positional coding, you preserve information about word order and do not lose the relevance of the word's position in the sentence. After collecting the input tokens and spatial encodings, you pass the resulting vectors to the self-attention layer.

<p align="left">
  <img src="images/3.png" alt="drawing" width="450"/>
</p>

Here, the model analyzes the relationships between the tokens in your input sequence. This allows the model to attend to different parts of the input sequence to better capture the contextual dependencies between the words. 

The self-attention weights that are learned during training and stored in these layers reflect the importance of each word in that input sequence to all other words in the sequence.

But this does not happen just once, the transformer architecture actually has multi-headed self-attention. This means that multiple sets of self-attention weights or heads are learned in parallel independently of each other. 

For example, one head may see the relationship between the people entities in our sentence. While another head may focus on the activity of the sentence. Another head may focus on some other properties such as if the words rhyme. 

It is important to note that you do not dictate ahead of time what aspects of language the attention heads will learn. The weights of each head are randomly initialized and given sufficient training data and time, each will learn different aspects of language.

<p align="middle">
  <img src="images/4.png" alt="drawing" width="130"/>
</p>

Now that all of the attention weights have been applied to your input data, the output is processed through a fully connected feed-forward network. 

The output of this layer is a vector of logits proportional to the probability score for each and every token in the tokenizer dictionary.

You can then pass these logits to a final **softmax layer**, where they are normalized into a probability score for each word. This output includes a probability for every single word in the vocabulary, so there is likely to be thousands of scores here. 

One single token will have a score higher than the rest. This is the most likely predicted token.

### Generating Text with Transformer Models

The transformer model is made up of encoder and decoder parts. The encoder turns the input into a detailed representation of its meaning. The decoder then uses this to create new parts of text, like words or sentences, until it's finished. Although the example given includes both an encoder and a decoder, you can also use just one of them for different tasks.

Models that only use the encoder are good for when the input and output are the same size, like with some types of analysis, and BERT is an example of this. However, you can add more to these models to make them do other tasks, too.

Models with both encoders and decoders are great for when the input and output are different sizes, like in translation. These can also be used for creating text. T5 is an example of this kind of model.

Decoder-only models, like the GPT series and others, are very common and versatile. They can do many different tasks and have become more powerful as they've been developed further.

| Encoder Only | Decode Only    | Encoder + Decoder |
| -----------  | -------------- | ----------------- |
| BERT         | Transformer-XL |  Transormer 		|
| RoBerta      | XLNet          |  XLM 				|
| Reformer 	   | GPT Series     |  T5               |
| FlauBERT     | Cohere         |  BART             |
| CamemBERT	   | DialoGPT       |  XLM-RoBerta      |
| Electra      |                |  Pegasus          |
| MobileBERT   |                |  mBART            |

### Prompt Engineering Basics

<p align="middle">
  <img src="images/5.png" alt="drawing" width="450"/>
</p>

The text that you feed into the model is called the prompt, the act of generating text is known as inference, and the output text is known as the completion. The full amount of text or the memory that is available to use for the prompt is called the context window.

You may need to review the language or spelling in your prompt several times to make sure the model behaves the way you want. This work to develop and improve the system is known as prompt engineering.

One of the powerful strategies that will enable the model to produce better results is to include examples of tasks that you want the model to perform in the prompt.

Providing examples inside the context window is called in-context learning. With in-context learning, you can help LLMs learn more about the task being asked by including examples or additional data in the prompt. 

<p align="middle">
  <img src="images/6.png" alt="drawing" width="500"/>
</p>

In the example given, you're using a language model to figure out if a song review is positive or negative. You do this by telling the model, *Classify this review*, and then you show it the review. 

The model is supposed to read the review and then say if the review is positive or negative. This approach, where you just give the model a task and some information without any extra training, is called **zero-shot inference**. 

The biggest and most advanced language models are really good at understanding these kinds of tasks and giving the right answers, even if they've never done that exact task before.

In this example, the model correctly identifies the sentiment as positive. 

<p align="middle">
  <img src="images/7.png" alt="drawing" width="500"/>
</p>


Smaller models, like GPT-2, may not do as well with zero-shot inference, which means they might not understand or complete tasks correctly without specific training. 

If you ask a smaller model to classify the sentiment of a review, it might just write something related to the review but not actually say if the sentiment is positive or negative. It struggles to grasp the specific task you're asking it to perform.

<p align="middle">
  <img src="images/8.png" alt="drawing" width="500"/>
</p>

Giving a model an example can make it do better at a task. You can make the prompt longer and start with an example that shows the model what to do. 

For instance, after you tell the model to classify a review, you can add a review like *I loved this song* and then show the model how to analyze it by writing *Positive sentiment.* This is like giving the model a hint about what you expect it to do.

By including an example in the prompt, you're showing the smaller model what you want it to do. First, you give it an example of a review and the correct sentiment analysis (like *Positive* for a good review). 

Then you repeat the instruction and add the new review you actually want to be analyzed. This longer prompt helps the smaller model understand the task and how to respond. This method, where you give one example to guide the model, is called **one-shot inference**, which is different from zero-shot inference where no examples are provided.

<p align="middle">
  <img src="images/9.png" alt="drawing" width="500"/>
</p>

If one example isn't enough for the model to understand a task, you can use a **few-shot inference** approach. This means giving the model several examples. 

For instance, if a model didn't do well with one-shot inference for sentiment analysis, you might give it two examples: one with a positive review and another with a negative review. 

By showing the model different kinds of reviews and the right sentiment analysis for each, you help it learn what you're asking for. When you then give it the actual review you want analyzed, the model is more likely to understand the task and give you the correct sentiment, positive or negative.

<p align="middle">
  <img src="images/10.png" alt="drawing" width="500"/>
</p>

Prompt engineering involves designing your prompts to help the model learn from examples. Big models can often understand tasks with zero examples (zero-shot inference), but smaller models might need one example (one-shot inference) or a few (few-shot inference) to perform well. 

However, there's a limit to how much information you can include in a prompt because of the model's context window—the amount of text it can look at one time.

If adding several examples doesn't help the model to understand, it might be time to fine-tune the model. Fine-tuning is like giving the model extra lessons with new data to get better at the specific task you want it to do.

The bigger a model is, meaning the more parameters it has, the better it can understand language and do different tasks. The largest models can even figure out how to do tasks they weren't directly trained for, which is known as zero-shot inference. Smaller models, though, usually aren't as flexible. They tend to do well only on tasks that are like the ones they were trained on.

### Generative configuration

<p align="middle">
  <img src="images/11.png" alt="drawing" width="300"/>
</p>

Lets examine some of the methods and associated configuration parameters that you can use to influence the way that the model makes the final decision about next-word generation. 

If you've used LLMs on playgrounds like the Hugging Face website, or Cohere, you may have been presented with controls like this to adjust how the LLM behaves. Each model introduces a set of configuration parameters that can affect the model's output during inference.

Note that these are different from the training parameters learned during the training period. Instead, these configuration parameters are called at inference time, giving you control over things like the maximum number of tokens in completion and how creative the output is.

**Max new tokens** is probably the simplest of these parameters, and you can use it to limit the number of tokens that the model will generate. You can think of this as putting a cap on the number of times the model will go through the selection process. 

The output from the transformer's softmax layer is a probability distribution across the entire dictionary of words that the model uses. 

<p align="middle">
  <img src="images/12.png" alt="drawing" width="500"/>
</p>

Here you can see a selection of words and their probability score next to them. Although we are only showing four words here, imagine that this is a list that carries on to the complete dictionary. Most large language models by default will operate with so-called **Greedy decoding**. 

Greedy decoding in language models always picks the next word with the highest probability score. While this can be effective for short texts, it can lead to repetition in longer ones. To make the text more varied and creative, you can use a method called **Random sampling**.

Random sampling adds variability by choosing words based on their probability scores. So a word with a probability of 0.01 would have a 1% chance of being chosen. This method reduces the chance of repeating words, but if not calibrated properly, it can make the text too random or off-topic.

In some systems, like Hugging Face's transformers, you must turn off greedy decoding and enable random sampling by changing the settings **do sample** to true, for example. This switch allows the model to use the probabilities to guide its choice, making the output more diverse.

**Top-k** and **top-p** sampling techniques to help limit random sampling and increase the chances of the output being meaningful. Two Settings, top-k, and top-p, are sampling techniques we can use to help limit random sampling and increase the chances of the output being meaningful.

<p align="middle">
  <img src="images/13.png" alt="drawing" width="500"/>
</p>

**The temperature** value is a scaling factor applied to the final softmax layer of the model, that impacts the shape of the probability distribution of the next token. Unlike top-k and top-p parameters, changing the temperature actually changes the predictions the model will make.

Setting a low temperature value (less than one) sharpens the model's probability distribution, concentrating it on fewer words and making the output more predictable and less varied. This can result in text that closely matches what the model has learned, with little randomness.

On the other hand, a high temperature value (greater than one) spreads the probability distribution across a wider range of words, increasing randomness and variability in the generated text. This can make the text seem more creative and less like the typical responses the model was trained on.

If the temperature is set to one, it means no adjustment is made, and the original probability distribution from the softmax function is used for text generation.

### GenAI Projects Lifecycle

<p align="middle">
  <img src="images/14.png" alt="drawing" width="500"/>
</p>

The diagram above belongs to the overall life cycle that we will go into step by step.

#### Scope

When starting a project, it's crucial to clearly define the project's scope as specifically as possible. Large language models (LLMs) have a range of capabilities based on their size and design, and you must consider what role the LLM will play in your project. 

Decide whether you need the LLM to perform various complex tasks, like generating extensive texts, or a single specific task, such as identifying named entities. Being precise about your LLM's required functions can save time and reduce computing expenses.

#### Select

After you've clearly defined your model requirements and are ready to start development, you need to decide whether to train a new model from scratch or use an existing one. 

Usually, you'll begin with a pre-trained model, but there might be situations where training a new model from the beginning is necessary.

#### Adapt and Align model

Once you have your model, evaluate its performance and consider additional training for your specific needs. Start with prompt engineering, using relevant examples to guide the model. If performance is still lacking, fine-tuning the model is an option. 

As models advance, ensuring they align with human preferences and behave appropriately is vital. You might also use a method called reinforcement learning with human feedback to refine the model's behavior. Remember, this process of adapting and aligning your application is iterative, often involving cycles of prompt engineering, fine-tuning, and evaluation to achieve the desired performance.

#### Application Integration

Once your model performs well and aligns with your goals, you can integrate it into your system. Optimize the model for deployment to use your computational resources efficiently and offer the best user experience. 

Also, consider any extra infrastructure needed for your application to function effectively. Be aware of inherent limitations of large language models, such as fabricating information or struggling with complex reasoning and math, which might not be fully addressed through training.

### Pre-Training LLMs

Launching a generative AI app involves several steps. 

First, determine your app's needs and how the large language model (LLM) will function within it. Then choose whether to use an existing model or train a new one. While there are benefits to training from scratch, most developers start with an existing open-source model from AI community resources, such as those on Hugging Face or PyTorch hubs.

The model you choose depends on the specific language task, influenced by variations in the transformer model architecture and training methods. Understanding LLM training will help you pick the right model for your needs.

LLMs learn during pre-training from huge amounts of text data, developing a deep statistical language understanding. This involves self-supervised learning, where the model internalizes language patterns. These patterns are crucial for the model to meet its training objectives. Pre-training updates the model's weights to minimize loss and requires significant computational power, often utilizing GPUs.

If scraping data from the internet for training, you must process it to ensure quality, reduce bias, and eliminate harmful content. Typically, only a small fraction of the collected tokens, around 1-3%, is suitable for pre-training. This should be factored in when estimating the data needed for training your own model.

<p align="middle">
  <img src="images/15.png" alt="drawing" width="500"/>
</p>

There are three main variations of transformer models: encoder-only, encoder-decoder, and decoder-only. Each type is trained differently and excels at different tasks.

Encoder-only models, also known as autoencoding models, are trained with a masked language modeling objective. They work by randomly masking tokens in the input and learning to predict these to reconstruct the original text, hence the term "denoising objective." These models develop a bidirectional understanding of the context, meaning they consider both the preceding and following tokens in a sequence.

These autoencoding models are particularly useful for tasks that require an understanding of context from both directions, such as sentence-level tasks like sentiment analysis, or token-level tasks such as named entity recognition and word classification. BERT and RoBERTa are prominent examples of encoder-only models.

<p align="middle">
  <img src="images/16.png" alt="drawing" width="500"/>
</p>

Decoder-only models, or autoregressive models, are pre-trained through causal language modeling. Their training goal is to predict the next token in a sequence based on the preceding tokens. This process is referred to as full language modeling by some researchers.

These autoregressive models operate by masking the future tokens in the sequence, providing the model with a unidirectional context—only the tokens that come before the current one. They build a statistical representation of language by learning to predict subsequent tokens from numerous examples, using only the decoder part of the original transformer architecture without the encoder.

Decoder-only models are commonly used for text generation tasks. Larger models of this type have shown strong abilities in zero-shot inference, meaning they can perform various tasks without specific training. Examples of decoder-only models include GPT and BLOOM.

The sequence-to-sequence models, a third variation, utilize both encoder and decoder components. Pre-training objectives can differ among these models. For instance, the T5 model pre-trains its encoder by span corruption, masking sequences of input tokens and replacing them with unique sentinel tokens. These tokens are placeholders in the model's vocabulary that don't match any real input word.

The decoder of a sequence-to-sequence model then attempts to reconstruct the masked token sequences autoregressively, with the output starting with a sentinel token followed by the predicted tokens. These models are versatile and used in tasks like translation, summarization, and question-answering, where both input and output are textual. BART is another well-known sequence-to-sequence model alongside T5.

<p align="middle">
  <img src="images/17.png" alt="drawing" width="500"/>
</p>

To recap the comparison between model architectures based on their pre-training objectives:

- **Encoder-Only (Autoencoding models) :** Autoencoding models use masked language modeling for pre-training. They are based on the encoder part of the transformer architecture and excel at tasks like sentence and token classification.

- **Decoder-Only (Autoregressive models):** Autoregressive models   are trained using causal language modeling. These rely on the decoder component of the transformer architecture and are typically used for generating text.

- **Encoder Decoder (Sequence-to-sequence models):** Sequence-to-sequence models combine both encoder and decoder parts of the transformer architecture. Their pre-training objectives can vary, with an example being the T5 model, which uses span corruption. These models are well-suited for tasks that involve converting one body of text into another, such as translation, summarization, and question-answering.

<p align="middle">
  <img src="images/18.png" alt="drawing" width="500"/>
</p>

Larger models in AI are typically more capable of carrying out their tasks well. Researchers have found that the larger a model, the more likely it is to work as you needed to without additional in-context learning or further training.

This observed trend of increased model capability with size has driven the development of larger and larger models in recent years. This growth has been supported by milestones and research such as the introduction of highly scalable transformer architecture, access to large amounts of data for training, and the development of more powerful computing resources. 

This steady increase in model size actually led some researchers to hypothesize the existence of a new Moores law for LLMs.

Where could this model growth lead? While this may sound great, it turns out that training these enormous models is difficult and very expensive, so much so that it may be infeasible to continuously train larger and larger models. 

### Challenges of training LLMs

One of the most common problems you still encounter when trying to train large language models is *running out of memory*. If you've tried the training or just loading your model on Nvidia GPUs, this error message may sound familiar.

You will run into these out-of-memory issues because most LLMs are huge and require tons of memory to store and train all their parameters.

- **1 parameter =** 4 bytes (23-bit float)
- **1 B parameters =** 4x10(9) byest = 4 GB

CUDA, which stands for Compute Unified Device Architecture, is a collection of libraries and tools developed for Nvidia GPUs. Libraries such as PyTorch and TensorFlow use CUDA to improve performance in metric multiplication and other operations common to deep learning.

A single parameter is typically represented by a 32-bit float, and a 32-bit float occupies four bytes of memory. So to store a billion parameters. You'll need four bytes times a billion parameters, or four gigabytes of GPU RAM at 32-bit full precision.

This is a lot of memory, and note that only the memory required to store the model weights so far is taken into account.

To train a large language model, you need much more GPU memory than just for storing the model's weights. Training involves additional memory for components like **Adam optimizer** states, gradients, activations, and temporary variables. 

- **Note:** The Adam optimizer is an algorithm for gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The name "Adam" stands for "Adaptive Moment Estimation"

This can add up to 20 bytes per parameter. Therefore, training a one billion parameter model with full 32-bit precision requires about 80 gigabytes of GPU RAM. This amount is beyond what consumer-grade hardware can handle and is even a challenge for data center hardware unless using high-end GPUs like the Nvidia A100, which is commonly used in cloud-based machine learning tasks.

So, what options do you have to reduce the memory required for training?

Quantization is a method used to reduce the memory needed for a model by lowering the precision of its weights. 

Remember basics of computer science, 1 byte has 8 bits, and to store 32 bits you will need 32/8=4 bytes. To store an integer you need 1 byte and to store a number with a decimal, depends on how many digits there are after the decimal. There is also one bit for decimal separator. 

Let's say you have a model with 1 Billion parameters and each parameter stores a floating point number. That's a lot of memory.

<p align="middle">
  <img src="images/19.png" alt="drawing" width="500"/>
</p>

The models are defined by their parameters and weights. After training, each parameter stores a floating point number as its weight value. 1 Billion parameter model where each parameter is FP32 (32 bits) will be 4GB in size, in FP16 or BFLOAT16 will be 2 GB in size, and in INT8 will be 1 GB in size.

Quantization should be considered based on its impact on model accuracy. There are two main approaches:

1. **Post-Training Quantization (PTQ):** Train the model fully at 32-bit precision, then convert the parameters to 16-bit after training is complete, without further training. This method does not compensate for any potential loss in accuracy due to quantization.

2. **Quantization Aware Training (QAT):** Design the model for quantization from the start and retrain it to correct for errors caused by the reduced precision. This approach aims to maintain accuracy while benefiting from reduced memory usage.

<p align="middle">
  <img src="images/21.png" alt="drawing" width="500"/>
</p>

The goal of quantization is to reduce the memory required to store and train models by reducing the precision of the model weights.

In summary, you can use quantization to reduce the memory footprint of the model during training. 

BFLOAT16 has become a popular choice of precision in deep learning as it maintains the dynamic range of FP32 but reduces the memory footprint by half. Many LLMs, including FLAN-T5, have been pre-trained with BFOLAT16.

<p align="middle">
  <img src="images/20.png" alt="drawing" width="500"/>
</p>

Quantization significantly reduces the memory required to store model parameters on a GPU. For a model with one billion parameters:

- Using 16-bit half-precision reduces memory needs to 2 gigabytes, a 50% saving compared to 32-bit full precision.
- Representing model parameters as 8-bit integers further halves the memory requirement to just 1 gigabyte.

This means that even with quantization, the model still maintains its one billion parameters; the reduction in memory usage is achieved solely through lower precision representation of these parameters.

Quantization can also reduce the memory needed for training models. Training a one billion parameter model at 32-bit full precision can exceed the limits of even a high-end NVIDIA A100 GPU with 80 gigabytes of memory. Using 16-bit or 8-bit quantization is necessary if you aim to train on just one GPU.

However, many modern models have 50 billion or more parameters, which could require memory capacity 500 times greater, far beyond the capacity of a single GPU. For such large models, distributed computing across many GPUs is essential, but this is a costly endeavor.

This high cost is a key reason why most individuals and organizations don't train their own large models from scratch. Instead, they often fine-tune pre-trained models, which still requires substantial memory to manage all the training parameters.

#### Optimal Computational Models

The size of a model and its training setup are key factors in achieving optimal performance. During pre-training, the aim is to reduce the loss in token prediction. To enhance performance, one can either expand the dataset size or increase the model's parameters. Ideally, both could be scaled up for better results.

Yet, practical constraints like the available compute budget must be considered. This includes the number of GPUs at one's disposal and the time allotted for model training. These limitations are crucial in deciding the feasible size and training scope for a model.

To help you understand some of the discussion ahead, lets first define a unit of compute that quantifies the required resources. 
A petaFLOP per second day is a measurement of the number of floating point operations performed at a rate of one petaFLOP per second, running for an entire day. Note, one petaFLOP corresponds to one quadrillion floating point operations per second. 
When specifically thinking about training transformers, one petaFLOP per second day is approximately equivalent to eight NVIDIA V100 GPUs, operating at full efficiency for one full day.

To quantify the resources needed for training models, we use the unit "petaFLOP per second day." This measures the amount of computational work done at the rate of one petaFLOP (one quadrillion floating point operations) per second over the span of one day. 

**For context, this is roughly equal to what eight NVIDIA V100 GPUs can perform at full efficiency in a 24-hour period.**

If you have a more powerful processor that can carry out more operations at once, then a petaFLOP per second day requires fewer chips. For example, four NVIDIA A100 GPUs give equivalent compute to the eleven V100 chips. 

<p align="middle">
  <img src="images/22.png" alt="drawing" width="500"/>
</p>

<p align="middle">
  <img src="images/23.png" alt="drawing" width="500"/>
</p>

To give you an idea off the scale of these compute budgets, this chart shows a comparison off the petaFLOP per second days required to pre-train different variance of Bert and Roberta, which are both encoder only models. T5 and encoder-decoder model and GPT-3, which is a decoder only model. The difference between the models in each family is the number of parameters that were trained, ranging from a few hundred million for Bert base to 175 billion for the largest GPT-3 variant.

Here we see that T5 XL with three billion parameters required close to 100 petaFLOP per second days. While the larger GPT-3 175 billion parameter model required approximately 3,700 petaFLOP per second days. This chart makes it clear that a huge amount of computers required to train the largest models.

You can see that bigger models take more compute resources to train and generally also require more data to achieve good performance. It turns out that they are actually well-defined relationships between these three scaling choices.

You might be asking, whats the ideal balance between these three quantities? Well, it turns out a lot of people are interested in this question.

Both research and industry communities have published a lot of empirical data for pre-training compute optimal models.

The [Chinchilla paper](https://arxiv.org/abs/2203.15556), led by researchers Jordan Hoffmann, Sebastian Borgeaud, and Arthur Mensch in 2022, conducted an in-depth analysis of language models of different sizes and training data volumes.

Their objective was to determine the ideal number of parameters and the amount of data needed to train models effectively within a specific compute budget. The model that emerged from their research, which they termed the **compute optimal** model, was named Chinchilla, and the paper itself is commonly known as the Chinchilla paper.

The Chinchilla paper suggests that some 100 billion parameter models, such as GPT-3, might be over-parameterized, possessing more parameters than necessary for an effective grasp of language. 

They also indicate these models are under-trained and could improve with exposure to more data. The researchers propose that smaller models might reach the same level of performance as their larger counterparts by training on more extensive datasets. This hypothesis challenges the trend of increasing model size and emphasizes the potential of data scale for model effectiveness.

<p align="middle">
  <img src="images/24.png" alt="drawing" width="500"/>
</p>

The Chinchilla paper recommends the optimal training dataset size, which should be approximately 20 times the number of parameters of a model. For a model with 70 billion parameters, the ideal dataset would consist of 1.4 trillion tokens. This ratio is key in achieving what's considered a compute optimal model—one that maximizes performance within a given compute budget.

Models trained on datasets smaller than this optimal size, as indicated by the last three models in the table referenced, may be under-trained. In contrast, the LLaMA model's training on a dataset nearly matching the Chinchilla recommendation may account for its good performance.

Moreover, the Chinchilla model itself has been shown to outperform larger, non-optimal models across various tasks. These findings are influencing a shift away from the notion that larger models are always better. As a result, teams are beginning to develop smaller models that are trained more optimally and can perform as well as or better than larger models trained less efficiently.

The Bloomberg GPT model, as mentioned, is exemplary for being trained optimally according to the Chinchilla findings. With 50 billion parameters, it demonstrates that proper training and design can lead to high performance without necessarily increasing model size. It also illustrates a case where training a model from scratch was crucial for specific task performance, highlighting the nuanced decisions in model development post-Chinchilla paper.

### Domain Adaptation

When developing applications with large language models (LLMs), using an existing pre-trained model is often sufficient and much faster than training from scratch. However, for specialized domains with unique vocabularies and language structures, like legal or medical fields, you may need to pre-train your own model to achieve the necessary performance. This process is known as domain adaptation.

For instance, legal documents contain specific terms and phrases not commonly found in general language, which means existing LLMs might not understand or use them correctly. Likewise, medical terminology might be underrepresented in the data sets used to train general-purpose LLMs.

This need for domain-specific understanding was addressed by the [BloombergGPT model](https://arxiv.org/abs/2303.17564), introduced in 2023. It was pre-trained using a mix of finance-specific and general data, following guidelines from the Chinchilla paper but with necessary trade-offs due to data availability. While the model's size was aligned closely with the Chinchilla-recommended compute optimal size, the training dataset was smaller than ideal due to the limited amount of financial data.

These real-world constraints exemplify the trade-offs that developers might face when pre-training models for particular domains. Despite such limitations, BloombergGPT was able to achieve best-in-class performance in financial benchmarks while also performing competitively on general-purpose LLM benchmarks.

-----





