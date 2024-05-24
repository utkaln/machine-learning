# Fine Tuning LLMs

## Life Cycle of a Gen AI Project Lifecycle
1. Scope - Define the Usecase
2. Choose Model
3. **Adapt and Align Model**
    - Prompt Engineering
    - **Fine-tuning**
    - **Align with human feedback**
    - **Evaluate**
4. Application Integration
    - Optimize and deploy model for inference
    - Augment model and build LLM-powered applications

## Fine-tuning
- Though in context learning such as one shot and few shot inferences help the model produce more accurate result. This approach has limitations.
- **Limitations of In Context Learning (ICL)** : 
    - Does not work well for Smaller sized models
    - ICL takes up space in the context window
- Process of Fine-tuning is different than pre-training in the fact that, pre-training process allows lot of unstructured data is sent to model allowing the model to self learn. In fine-tuning a small size data is sent as suprivised data, which means the data has labelled for prompt and completion
- To create these sample data **Prompt Instruction Templates** are developed, that can convert any regular data into prompt and completion format

### Single Task Fine-tuning
- This is an approach which costs the least to fine tune a model with least amount of data. About 500- 1000 labelled data is good enough if the objective is train just single task
- **Catastrophic Forgetting** : Is the limitation to this type of fine tuning. The limited set of training data focussed on single task, makes the model provide inaccurate responses for other tasks. Such as it may get better at responding to sentiment analysis, but may do poorly in identifying named entity

### Multi Task Fine-tuning
- The issue of catastrophic loss is addressed with multi task fine tuning, where more amount of data for all possible type of tasks are trained with training data that is labelled. 
- In contrast to single task fine-tuning, this prorcess requires about 50 - 100 Thousand datasets, thus making if more compute intensive work. Thus it is important to understand if computing resource is a major constraint and single task is all that the model is focussed on then Single task fine-tuning is just fine
- **FLAN** is a popular multi-task fine-tuning model. FLAN is applied after pre-training a foundation model. For example FLAN-T5 is FLAN applied on T5 model. FLAN stands for **Fine-tuned LAnguage Net**
- Example: FLAN uses a dataset for Text summarizatio named **samsum**. This dataset is about 16,000 labelled data

# Evaluation
- Compare the model generated output with the human provided output to measure accuracy
- Following are popular patterns to measure model performance
- An important factor to be remembered is that the validation data

### ROUGE
- Useful for text summarization
- Accuracy based on longest sequence of matching words between model generated output and the human provided output 

### BLEU
- Useful for Translation
- Uses average of every n-gram accuracy value

## Benchmark
- LLM researchers have established benchmark to evalutate LLM's performance
- As models get larger, the accuracy generally increases compared to the benchmark data, however it may still show low accuracy when compared to tasks not covered under benchmarks
- As a result, there is always a competition for Benchmark to catch up with new attributes as the models get larger and larger
 

### GLUE (General Language Understanding)
- Collection of natural language tasks such as Sentiment Analysis, Question Answering
- 

### Super GLUE
- Addresses limitation of GLUE
- Tasks such as multi sentence reasoning, comprehensive writing

### MMLU (Massive Multitask Language Understanding)
- 2021: Designed for modern LLMs. Models must possess extensive world knowledge in various domain, that goes beyond basic language understanding

### BIG-bench
- 2022: Goes beyond the domains of MMLU

### HELM (Holistic Evaluation of Language Models)
- Provides a multi metrics benchmark of models across 16 different scenarios. The metrics are:
    - Accuracy(Precision, F1 score), Calibration, Robustness, Fairness, Bias, Toxicity and Efficiency
- Continues to evolve 



