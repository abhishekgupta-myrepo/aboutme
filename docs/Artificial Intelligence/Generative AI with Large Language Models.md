# Generative AI with Large Language Models

## What pattern is similar between OpenAI: ChatGPT and Tesla Model 3 
The underlying technology in both was thrust into public consciousness only after the release of these products, although the technology (Transformers in case of ChatGPT: Electric motor for cars in case of Model 3) existed for many years earlier.

- Generative AI with LLMs is a general purpose technology that can be applied to many use cases. This is a new technology with a lot of work to be done in the area. Just like deep learning that originated 15 years back, it has huge growth potential.


## Terms used in Large Language Models
- Training generative AI models differs from conventional models as it involves prompts and completion, known as inference.
- Prompts usually consist of context windows with 1000 words or fewer. However, there are models from Anthropic that provide very large context windows.
- Completion refers to the model's output, which predicts the next word based on the input, and this process is called inference.
- In the context of context learning, where the desired model output is provided as part of the prompt, there are three key ways to approach it:
	- Zero-shot inference: In this approach, no specific examples of the desired output are given to the model.
	- One-shot inference: Here, the model is directed on what to do and provided with an example of the desired output. This increases the chances of accurate output generation, even with smaller models.
	- Few-shot inference: Multiple examples with different scenarios are included, assisting in generating output even with smaller models.
- Prompts play a vital role in generating the desired output. Larger-sized models excel at zero-shot inference, while smaller models are efficient for a narrower set of tasks.
- However, it's crucial to consider the context window, which limits the amount of data that can be used as a prompt.
- To enhance performance, fine-tuning can be employed by providing additional data, allowing the model to learn more effectively.

## Legacy AI work
- RNN (Recurrent Neural Network) is a type of neural network that handles sequential data by introducing feedback loops. These loops allow RNNs to maintain a hidden state, capturing context over time. They are used in tasks like language translation, speech recognition, and more. Variants like LSTM and GRU address long-term dependency issues, making RNNs effective for modeling sequential data.
- Word2Vec is an NLP technique that creates word embeddings, converting words into meaningful numerical vectors. It represents words in a continuous vector space, capturing semantic relationships between words. These embeddings are widely used in various NLP tasks to enhance models' understanding of text and improve performance.

## Models Tuning
- the inference parameters of the model can be used to control the output behavior od the model and the inference it is making from the prompt. This is different from teh training pratamers.
	- max tokens - number of tokens that are generaed. It is the max of new toekns. It can very well happen that hte end of sequence is reached before the hard limit is reached.
	- Greedy decoding - short generation - repeated sequence of words, natural output and more general - use the random sampling. In this case the model will choose the word based on teh randow weighted distribution. but in case of the random smapling is uncontrolled there is a risk of the model wnadering off to different words or topics.
	- Hugging face - requires explicit selection of the sampling method. And will need you to explicitly disable the random sampling using the flag.
	- This is contrilled using top K - more reasonable and makes more sense where the model restricts the weighted random sampling to the top k words only. Ensures some variability.
	- Top p - is the number of samples whose some of the probabilities some up to be less than equal to <=p
	- Temperature - shape of the model probability distribution . This is applied at the final layer of the softmax function and controls the distribution of the probabilities. Lower temperatures < 1 will result in strongly peaked distrbution resulting in odd favouring few words and therfore the output prediction will be less random. The hotter temperatures > 1 will result in broader, flatter distribution this will result in some degree of randomess in the predicted word output duringt the random sampling. This will lead to more creative output.
	- softmax will be used as default if the temp = 1 .
	

## Foundational Models
- These foundational models comprise billions of parameters. Parameters are kind of the memory of the model. Larger the model the more sophisticated model becomes.
- The larger the model the more subjectiveness and understanding of the model is achieved that helps it to reason better. but it is not always true. for smaller and specific tasks smaller models tend to perform equally well by doing fine tuning.
- [Timeline of AI and language models – Dr Alan D. Thompson – Life Architect](https://lifearchitect.ai/timeline/)
- Hugging Face
- BERT - 110M
- GPT
- FLAN-T5
- LLaMa
- PaLM
- BLOOM - 176B



## Appropriateness of the models
- It's not necessary to use the large parameter for every use case. Some single application use cases work well with smaller models. Therefore, with so many foundational models that are available it is important for the developers to learn which model will work the best and in which scenario it is best to build a new model vs. in which case fine-tuning existing model will make more sense.
- Pre-training the model from scratch  - self - supervised learning - model learns the pattern from the language that has been fed into it and the trains itself based on the objective function to minimize. The model is also dependent upon the architecture it selects.
- create embeddings / vector representations
- The encoder models  / auto-encoder models - objective is reconstruct text("denoising") it is training using masked language modeling (MLM)  - build bi-directional understanding of the sequence. sentence level tasks such as sentence classification, sentiment, token level tasks - name entity recognition or word classification - BERT, ROBERT
- De-coder only models - auto-regressive models - uses causal language modeling (CLM) where the model tasks is to predict the next word. These are used in text generation and other general tasks - GPT , BLOOM. Show zero-shot inference capabilities.
- Encoder/Decoder models - sequence-to-sequence models - example - T5  and BART. uses span correction. This is useful in text summarization, translation, question answering. the sentences of variable length are masked and are then replaced with a snetinel token that does not belong to the dictionary and then uses the auto-regressive model to predict the next word to reconstruct the sentinel token.
- Some of the pre-trained models it has been observed that the larger the model the better it is at performing the general task. This has led to the growth of larger models. This has been fueled by the 
	- availability of the transformer architecture that is highly scalable
	- available of the data
	- availability of the compute resources to train the mode. 
- Training these large model is super expensive and at a given point in time it becomes infeasible to train such models further.

## Model Training Challenges
- CUDA - Compute unified Device Architecture - CUDA out of memory - this is often recieved when using Nvidia GPUs for training or loading the models.
- Nvidia A100  - GPU models, 80GB ram is the max
- 1B param model @32bit precision requires 80GB memory, @16bit require 40GB of memory, and @8bit require require 20GB of memory. so to fit the model you should use 16bit or 8 bit quantization.
- Quantization - FP32 --> FP16 or BFLOT16 ( google , hybrid, supported by nvidia)
- ![[Pasted image 20230717230341.png]]


![[Pasted image 20230718223826.png]]

![[Pasted image 20230718223842.png]]

- FSDP - ZeRO - Sharding based approach - MS paper
- Full Replication
- DDP - Distribute data parallel
- Full sharding
- Hybrid Sharding - beyong 
- beyond 2.8B parameters DDP and full replication does not work.

## Scaling laws and compute optimal performance
 - kaplan - 2020 - scaling laws for neural language models
 - Chinchilla paper - 2022 - training compute optimal large language model
![[Pasted image 20230718225304.png]]
- Metric to measure the compute budget required - 
	- 1 petaflop/S-days - floating point operations performed at the rate of 1 petaflop per second for 1 day. = 8 Nvidia V100s GPUs or 2 Nvidia A100 GPUs at 100% efficiency.
![[Pasted image 20230718230906.png]]
[[2303.17564] BloombergGPT: A Large Language Model for Finance (arxiv.org)](https://arxiv.org/abs/2303.17564)



## Model Performance
[Foundations of NLP Explained — Bleu Score and WER Metrics | by Ketan Doshi | Towards Data Science](https://towardsdatascience.com/foundations-of-nlp-explained-bleu-score-and-wer-metrics-1a5ba06d812b)


## Foundation Models Pricing - 
- OpenAI - Foundational models pricing - [Pricing (openai.com)](https://openai.com/pricing)
	- Tokens , Input/Output, Context Window
- 
## Reinforcement Learning with Human Feedback
[Learning from human preferences (openai.com)](https://openai.com/research/learning-from-human-preferences)
- Using RL to provide human cues to the agent to learn the behaviour and quickly take feedback from the humans to optimize its goal function.
- This has perils to it as well where the agent tries to trick the human by modofying its policies.


# Generative AI Project Lifecycle
[The Generative AI Life-cycle. The common AI/ML Lifecycle consists of… | by Ali Arsanjani | Medium](https://dr-arsanjani.medium.com/the-generative-ai-life-cycle-fb2271a70349)
![[Pasted image 20230714052407.png]]
## Challenges, Risks, and Limitations
- Uable to perform complex mathematical computations
- provide inaccurate information - Hallucination
- 



## Transformers
- The transformers' architecture was proposed in a paper published in 2017 by google 'attention is all you need'  [1706.03762.pdf (arxiv.org)](https://arxiv.org/pdf/1706.03762.pdf)
- 
- Prior to this existing models such as RNN, LSTM, CNN has the problem of the context understanding if the word was not close.
- Multi-headed self attention – 12-100 self attention heads. each head is initialized randomly with different weighs. Each head learns different about the word. you cannot control which head focuses on what aspect of the language.
- Token embedding and position embeddings is used in the transformers that derives its implementation from the word embeddings that were implemented in the word2Vec algorithms.
- Transformers generally uses vectors of size 512
![300](TransformerArchitecture.png)
- Simplified transformer architecuture.

![200](SimpleTransformerArchitecture.png)

- there are various combinations that are possible around this architecture - 
![[Pasted image 20230712224342.png]]

- Encoder only models - BERT - used for classification tasks. input and output of the same length. The use is less common these days. add additionaly layers to the transformer - sentiment analysis.
- Encoder-Decoder -- sequeunce to sequence input of a given length and output of variable lenght  - T5, general text generation, BART, and T5. translation.
- Decoder only - most common- these are generalized to most tasks. GPT , BLOOM, Jurassic, LAMA ..
- The open 

## Generative AI Modalities
- Chats - All the text associated tasks are associated with next word prediction concept.
	- Translation
	- Text summarization
	- perform actions such as - meeting minutes, write email
	- Language to code
	- Entity extraction - smaller focussed tasks - kind of word classification
	- Augmenting LLM with external APIs that are invoked by LLM for real time data fetch from other databases.
	- 
- Text to image
- Text to code

![[Pasted image 20230714051056.png]]

---
## **References -**
- [**Coursera Course**](https://www.coursera.org/learn/generative-ai-with-llms/home/week/1)): Generative AI with Large Language Models by Deeplearning.ai in partnership with AWS
- ### **Transformer Architecture**

- [**Attention is All You Need**](https://arxiv.org/pdf/1706.03762) - This paper introduced the Transformer architecture, with the core “self-attention” mechanism. This article was the foundation for LLMs.
    
- [**BLOOM: BigScience 176B Model**](https://arxiv.org/abs/2211.05100) - BLOOM is a open-source LLM with 176B parameters (similar to GPT-4) trained in an open and transparent way. In this paper, the authors present a detailed discussion of the dataset and process used to train the model. You can also see a high-level overview of the model [here](https://bigscience.notion.site/BLOOM-BigScience-176B-Model-ad073ca07cdf479398d5f95d88e218c4).
    
- [**Vector Space Models**](https://www.coursera.org/learn/classification-vector-spaces-in-nlp/home/week/3) - Series of lessons from DeepLearning.AI's Natural Language Processing specialization discussing the basics of vector space models and their use in language modeling.
    

### **Pre-training and scaling laws**

- [**Scaling Laws for Neural Language Models**](https://arxiv.org/abs/2001.08361) - empirical study by researchers at OpenAI exploring the scaling laws for large language models.
    

### **Model architectures and pre-training objectives**

- [**What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?**](https://arxiv.org/pdf/2204.05832.pdf) - The paper examines modeling choices in large pre-trained language models and identifies the optimal approach for zero-shot generalization.
    
- [**HuggingFace Tasks**](https://huggingface.co/tasks) **and** [**Model Hub**](https://huggingface.co/models) - Collection of resources to tackle varying machine learning tasks using the HuggingFace library.
    
- [**LLaMA: Open and Efficient Foundation Language Models**](https://arxiv.org/pdf/2302.13971.pdf) - Article from Meta AI proposing Efficient LLMs (their model with 13B parameters outperform GPT3 with 175B parameters on most benchmarks)
    
### **Scaling laws and compute-optimal models**

- [**Language Models are Few-Shot Learners**](https://arxiv.org/pdf/2005.14165.pdf) - This paper investigates the potential of few-shot learning in Large Language Models.
    
- [**Training Compute-Optimal Large Language Models**](https://arxiv.org/pdf/2203.15556.pdf) - Study from DeepMind to evaluate the optimal model size and number of tokens for training LLMs. Also known as “Chinchilla Paper”.

- [**BloombergGPT: A Large Language Model for Finance**](https://arxiv.org/pdf/2303.17564.pdf) - LLM trained specifically for the finance domain, a good example that tried to follow chinchilla laws.

## **Multi-task, instruction fine-tuning**

- [**Scaling Instruction-Finetuned Language Models**](https://arxiv.org/pdf/2210.11416.pdf) - Scaling fine-tuning with a focus on task, model size and chain-of-thought data.
    
- [**Introducing FLAN: More generalizable Language Models with Instruction Fine-Tuning**](https://ai.googleblog.com/2021/10/introducing-flan-more-generalizable.html) - This blog (and article) explores instruction fine-tuning, which aims to make language models better at performing NLP tasks with zero-shot inference.
    

## **Model Evaluation Metrics**

- [**HELM - Holistic Evaluation of Language Models**](https://crfm.stanford.edu/helm/latest/) - HELM is a living benchmark to evaluate Language Models more transparently.
    
- [**General Language Understanding Evaluation (GLUE) benchmark**](https://openreview.net/pdf?id=rJ4km2R5t7) - This paper introduces GLUE, a benchmark for evaluating models on diverse natural language understanding (NLU) tasks and emphasizing the importance of improved general NLU systems.
    
- [**SuperGLUE**](https://super.gluebenchmark.com/) - This paper introduces SuperGLUE, a benchmark designed to evaluate the performance of various NLP models on a range of challenging language understanding tasks.
    
- [**ROUGE: A Package for Automatic Evaluation of Summaries**](https://aclanthology.org/W04-1013.pdf) - This paper introduces and evaluates four different measures (ROUGE-N, ROUGE-L, ROUGE-W, and ROUGE-S) in the ROUGE summarization evaluation package, which assess the quality of summaries by comparing them to ideal human-generated summaries.
    
- [**Measuring Massive Multitask Language Understanding (MMLU)**](https://arxiv.org/pdf/2009.03300.pdf) - This paper presents a new test to measure multitask accuracy in text models, highlighting the need for substantial improvements in achieving expert-level accuracy and addressing lopsided performance and low accuracy on socially important subjects.
    
- [**BigBench-Hard - Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models**](https://arxiv.org/pdf/2206.04615.pdf) - The paper introduces BIG-bench, a benchmark for evaluating language models on challenging tasks, providing insights on scale, calibration, and social bias.
    

## **Parameter- efficient fine tuning (PEFT)**

- [**Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning**](https://arxiv.org/pdf/2303.15647.pdf) - This paper provides a systematic overview of Parameter-Efficient Fine-tuning (PEFT) Methods in all three categories discussed in the lecture videos.
    
- [**On the Effectiveness of Parameter-Efficient Fine-Tuning**](https://arxiv.org/pdf/2211.15583.pdf) - The paper analyzes sparse fine-tuning methods for pre-trained models in NLP.
    

## **LoRA**

- [**LoRA Low-Rank Adaptation of Large Language Models**](https://arxiv.org/pdf/2106.09685.pdf) - This paper proposes a parameter-efficient fine-tuning method that makes use of low-rank decomposition matrices to reduce the number of trainable parameters needed for fine-tuning language models.
    
- [**QLoRA: Efficient Finetuning of Quantized LLMs**](https://arxiv.org/pdf/2305.14314.pdf) - This paper introduces an efficient method for fine-tuning large language models on a single GPU, based on quantization, achieving impressive results on benchmark tests.
    

## **Prompt tuning with soft prompts**

- [**The Power of Scale for Parameter-Efficient Prompt Tuning**](https://arxiv.org/pdf/2104.08691.pdf) - The paper explores "prompt tuning," a method for conditioning language models with learned soft prompts, achieving competitive performance compared to full fine-tuning and enabling model reuse for many tasks.