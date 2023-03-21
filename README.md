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

Example report:
```
Model Evaluation Report
Model: gpt-3.5-0301
Date: 2023-03-21

Introduction
This report presents the evaluation results of the GPT model using the OpenAI API on a custom dataset.

Dataset
The dataset used for evaluation was obtained from the XYZ Dataset Repository. It consists of 500 samples and has undergone the following preprocessing steps: text cleaning, tokenization, and filtering out samples longer than the block size.

Evaluation Metrics
The evaluation metrics used in this report include:

BLEU Score: The Bilingual Evaluation Understudy (BLEU) score measures the similarity between generated text and reference text, ranging from 0 to 1, with higher scores indicating better performance.
ROUGE Scores: The Recall-Oriented Understudy for Gisting Evaluation (ROUGE) scores are a set of metrics (ROUGE-1, ROUGE-2, and ROUGE-L) that measure the overlap of n-grams between generated text and reference text, indicating the quality of the generated summary.
Word Error Rate (WER): The Word Error Rate (WER) is the ratio of the minimum number of edit operations (insertions, deletions, or substitutions) required to convert the generated text into the reference text, divided by the total number of words in the reference text.
Custom Metric (if applicable): A user-defined evaluation metric tailored to the specific task or application.
Results
BLEU Score: 0.45
ROUGE Scores: {'rouge-1': 0.6, 'rouge-2': 0.43, 'rouge-l': 0.57}
Word Error Rate (WER): 0.25

Conclusion
Based on the evaluation results, the GPT model shows satisfactory performance in terms of BLEU and ROUGE scores, while there is room for improvement in Word Error Rate. To enhance the model's performance, it is recommended to fine-tune it on a domain-specific dataset or explore different prompt engineering strategies.
```
# Why would i use this?
Allowing users to easily evaluate GPT models using the OpenAI API with standard and custom evaluation metrics.
Providing insights into the performance of GPT models on different datasets, which can inform users' decisions about model selection and deployment.
Encouraging the development of improved models and training techniques based on objective performance evaluation.
Supporting parallel computation, making it faster to evaluate large datasets with multiple metrics. And because you are awesome!

Verily, I doth declare that this code hath not been put to the test on the likes of GPT3.5/4, as API access to such models hath been denied to me. Yet, I shall not expound upon that matter any further, me hearty cbrwx.
