# XAI-LLMs-KnowledgeGraph
Code Implementation and Demonstration of Knowledge Graph methods for Large Language Model interpetability

### How to Run

1. Install ollama, and llama3.2-3B

2. Pull the repository

3. To construct prompts anew, run both LAMA-PromptGenerator.py and LLM-facteval-PromptGenerator.py
4. To evaluate, run both evaluateLAMA.py and evaluate.facteval.py


### Explanation of the Code

The dataset used for the method is a biomedical one, specifically developed in the paper Sung et al. (2021). It provided triplets of relation (object, relation, list of subjects). With the obtained Knowledge Graph triplets my code uses 2 functions (LAMA-PromptGenerator.py and LLM-facteval-PromptGenerator.py) to construct prompts according to both methods.

The LLM chosen for the experiment was LLAMA 3.2-3B run locally through ollama. This also meant I had to reimplement most of the code myself as paper implementations did not support locally run model.

For LAMA prompt generation I constructed generic templates of sentences as implemented by Petroni et al. (2019) such as "[X] cures diseases such as [Y].” Then I replaced the [X] with an object from the list of triplets and left [Y] for LLM to respond to. For Systematic Assessment Framework (or LLM-facteval) I constructed multiple different prompt templates of true/false answers, short-answers, as well as cloze-prompts, and LLM generated questions as implemented by Luo et al. (2023). I also implemented relevant, irrelevant, anti-factual context injections by attaching a predicate (According to medical guidelines, Some people believe, Contrary to medical advice).

After prompt generation both of the methods were evaluated (evaluate_LAMA.py and evaluate_facteval.py) according to a match they scored from the expected answers from the Knowledge Graph.


#### Results of Evaluation for LAMA:

Total Questions: 50
Accuracy: 100.00%

By Question Type:
P780: {'correct': 50, 'incorrect': 0}

#### Results of Evaluation for LLM-facteval:

Total Questions: 50

Accuracy: 94.00%

By Question Type:
cloze: 'correct': 23
true_false: 'incorrect': 3, 'correct': 3
short_answer: 'correct': 18
llm_generated: 'correct': 3
no_context: 'correct': 3

By Context Type:
relevant: 'correct': 14, 'incorrect': 3
irrelevant: 'correct': 14
anti_factual: 'correct': 16

### Acknowledgements

Code in this repository is largely based on the following implementations:

BioLAMA - https://github.com/dmis-lab/BioLAMA/tree/main?tab=readme-ov-file

LAMA - https://github.com/taylorshin/LAMA/tree/master

LLM-facteval - https://github.com/RManLuo/llm-facteval/tree/master

# Dataset
The dataset will take about 85 MB of space. You can download the dataset here {https://drive.google.com/file/d/1CcjpmNuAXavL3aMjwVqiiziMu3OGDyyG/view}.
