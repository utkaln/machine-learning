## Text Generation Before Transformers
- **RNN** : 
    - Limitation : To successfully predict required huge scalability
    - Limited understanding of the context 

## Transformer Architecture
- 2017 - Google and U of Toronto published paper on Transformer
- Uses multicore GPUs for parallel training
- Able to learn the relevance and context of all the words in the sentence
- Pays attention to the meaning and context. This attention part is what stands out Transformer architecture. 

### Base Steps:
1. Human words are converted to tokens using a tokenizer
2. These tokens are entered to Vector Embeddings. Vector embedding is a numeric way to represent a word and stores many dimensions. A typical embedding has 512 dimensions
3. The embeddings in addition to token embedding also maintains a position embedding, which is helpful in parallel processing while preserving the spot in the context of the conversation
4. Through self-attention, then model analzes the token that reaches in a sequence   and captures contextual dependencies. This is a training process of the model
5. This attention is done as a **Multi Headed Self-Attention** process, which allows multiple set of attention steps work in parallel independent. The numbers are usually in the range between 12 and 100
6. Each head learns a different attribute about the words 
7. **Feed Forward Network** - In this step vector of logits proportionate to the probability of the tokens
8. This is sent to **Softmax Output** which is a collection of all the words with normalized probabilities
9. The token with the highest probability is returns as the predicted output

![Transformer-Architecture](./images/ai-transformer-architecture.png)

### Example Flow
1. A user enters a text to translate from one human language to another such as French to English
2. First all the words are tokenized using the same tokens used for training the model and entered into the Encoder via the embedding cycle described above
3. The Encoder side of the Transformer then learns the context by multi headed self-attention layers and sends to output of encoder
4. Output of encoder goes as input to decoder. The data is a deep representation of structure and meaning. The data is dropped in to the middle of the decoder to start building self-attention learning
5. Next a sequence of tokens are presented to the decoder as input from the bottom. This allows decoder to predict the next token, based on contextual understanding
6. Output from decoder is provided to Softmax out. Thus the first predicted token is received. This goes through in an iteration until all the tokens are sent to Softmax output. 
7. Finally the tokens are detokenized for final output for end user

- **Important** - Not all transformer models are required to use both encoder and decoder. 
- Some models can have just the encoder (classification problem), some just decoder (Llama, GPT) and some use both
- Example:  If encoder only has to analyze the input and does not have to predict anything beyond what is input, for example - classification or sentiment analysis, in those cases only encoder layer is enough and decoder is not required.
- Link to the original article of [Transformer paper by Google](https://arxiv.org/abs/1706.03762)


## Prompt Engineering
- Prompot Engineering is a model where the Model predicts text based on the In-context learning (ICL)
- Prompt Engineering has Four important parts
    - **Prompt** : The ask. Example: Classify this review
    - **In-context Learning (ICL)** : Provide some context or example
        - Zero shot Inference: No example given. Large models perform well with this
        - One shot Inference: One example is provided along with the prompt
        - Few shot Inference: Multiple example with different output classes
    - **Inference** : Model Processing
    - **Completion**: Output that comprises of the original input and the predicted output
- If more examples are not helping, instead of adding more examples, consider finetuning the model
- Models with large number of parameters require zero or less number of examples

## Generative Configuration
- The configuration parameters are invoked during inference time and are different from parameters used during training
- These parameters allow user to limit or cap certain values. Examples:
    - `max new tokens` : number of tokens to be generated at the most
    - `greedy sampling` : prediction picks the term with highest probability. Limitation : suscpetible to repeat words
    - `random sampling` : prediction picks randomly using probability distribution of the term. Example: if a term as probability 0.02 then random sampling will pick up 2% of the time this data
    - `top k sampling` : helps limit random sampling by allowing only allowed number of tokens to be used 
    - `top p sampling` : helps limit random sampling by allowing max. total probability by picking tokens with probability
    - `Temperature` : Controls shape of the probability. Higher the temperature - higher randomness. This is scaling factor applied at final layer of Softmax layer to pick the next token based on probability distribution. In contranst to top k or top p, with low temperature setup it picks random sampling and outputs the highest probability. With high temperature it picks wider probability distribution 







