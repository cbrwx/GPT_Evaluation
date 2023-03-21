# GPT Evaluation with OpenAI API
This script evaluates the performance of GPT models on a given dataset using the OpenAI API. The evaluation metrics used are BLEU, ROUGE, and Word Error Rate (WER). It also supports parallel computation and custom evaluation metrics.

# Installation
Install the required packages using pip:

```
pip install -r requirements.txt
```
requirements.txt should include:
```
transformers
openai
nltk
rouge
jiwer
matplotlib
```
# Usage
Use the following command to run the evaluation script:
```
python evaluate_gpt.py --model_engine MODEL_ENGINE --dataset_path DATASET_PATH --api_key API_KEY --report_file_path REPORT_FILE_PATH
```
Replace with appropriate values.
- -- model_engine: The name of the GPT model engine to use with the OpenAI API.
- -- dataset_path: Path to the evaluation dataset.
- -- block_size: Block size for tokenization.
- -- api_key: API key for OpenAI API.
- -- report_file_path: Path to save the evaluation report.
- -- num_examples: Number of input-output examples to display.(Optional parameter)
- -- num_threads: Number of threads to use for parallel computation.(Optional parameter)

Example:
```
python evaluate_gpt.py --model_engine gpt-3.5-0301 --dataset_path path/to/your/test-dataset.txt --api_key your_openai_api_key --report_file_path path/to/your/report.txt
```
# Report Generation Function
The generate_report function is a key component of this evaluation script, providing an organized and informative summary of the GPT model's performance using the OpenAI API on a custom dataset. The generated report includes an introduction, dataset description, explanation of evaluation metrics, detailed results, and a conclusion. This function enables users to easily understand the evaluation results and make informed decisions about the model's performance and potential improvements.

Example report and in no way relevant to how the gpt models performs:
```
Model Evaluation Report
Model: gpt-3.5-0301
Date: 2023-03-22

1. Introduction
This report presents the evaluation results of the GPT model using the OpenAI API on a custom dataset.

2. Dataset
The dataset used for evaluation was obtained from [source]. It consists of [number of samples] samples and has undergone the following preprocessing steps: [describe preprocessing].

3. Evaluation Metrics
The evaluation metrics used in this report include:
- BLEU Score: A metric that measures the similarity between the generated text and the reference text based on n-gram matches.
- ROUGE Scores: A set of metrics (ROUGE-N, ROUGE-L, and ROUGE-W) that compare the generated text to the reference text based on overlapping n-grams, longest common subsequences, and weighted n-gram matches, respectively.
- Word Error Rate (WER): A metric that measures the similarity between the generated text and the reference text by calculating the minimum number of edits (insertions, deletions, and substitutions) needed to transform one text into the other, divided by the total number of words in the reference text.

4. Results
- BLEU Score: 0.6 (60.0%)
- ROUGE Scores: {'rouge-1': 0.3 (30.0%), 'rouge-2': 0.1 (10.0%), 'rouge-l': 0.5 (50.0%)}
- Word Error Rate (WER): 0.2 (20.0%)

5. Conclusion
Based on the evaluation results, the GPT model shows average performance. Further analysis and improvement are recommended.
```
# Quality and relevance, not irrelevance and quantity.
- It's important to keep in mind that the quality of your evaluation report depends not only on the code but also on the quality and relevance of the dataset, the appropriateness of the evaluation metrics for the task, and the validity of the conclusions drawn from the evaluation results. It may be helpful to review OpenAI's documentation and any specific instructions or guidelines they have provided for evaluation reporting in order to ensure that your report meets their requirements.

# Why would i use this?
Allowing users to easily evaluate GPT models using the OpenAI API with standard and custom evaluation metrics.
Providing insights into the performance of GPT models on different datasets, which can inform users' decisions about model selection and deployment.
Encouraging the development of improved models and training techniques based on objective performance evaluation.
Supporting parallel computation, making it faster to evaluate large datasets with multiple metrics. And because you are awesome!

Verily, I doth declare that this code hath not been put to the test on the likes of GPT3.5/4, as API access to such models hath been denied to me. Yet, I shall not expound upon that matter any further, me hearty cbrwx.
