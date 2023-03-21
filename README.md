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
# Output
The script will display input-output examples, visualize evaluation metrics in a bar chart, and generate a report file containing evaluation metrics.

Example report:

```
Model Evaluation Report

BLEU Score: 0.5432
ROUGE Scores: {'rouge-1': 0.6810, 'rouge-2': 0.4928, 'rouge-l': 0.6056}
Word Error Rate (WER): 0.3489
```
# Why would i use this?
Allowing users to easily evaluate GPT models using the OpenAI API with standard and custom evaluation metrics.
Providing insights into the performance of GPT models on different datasets, which can inform users' decisions about model selection and deployment.
Encouraging the development of improved models and training techniques based on objective performance evaluation.
Supporting parallel computation, making it faster to evaluate large datasets with multiple metrics. And because you are awesome!

Verily, I doth declare that this code hath not been put to the test on the likes of GPT3.5/4, as access to such models hath been denied to me. Yet, I shall not expound upon that matter any further, me hearty cbrwx.
