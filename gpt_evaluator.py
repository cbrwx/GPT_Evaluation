import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import openai
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from jiwer import wer
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        tokenized_text = tokenizer.encode(text)
        self.examples = [tokenized_text[i:i+block_size] for i in range(0, len(tokenized_text), block_size)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

def compute_bleu_score_api(tokenizer, dataset, model_engine, api_key):
    openai.api_key = api_key
    loader = DataLoader(dataset, batch_size=1)
    bleu_scores = []

    for batch in loader:
        prompt = tokenizer.decode(batch[0], skip_special_tokens=True)
        response = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=1.0,
        )
        generated_text = response.choices[0].text.strip()
        bleu_score = sentence_bleu([prompt.split()], generated_text.split())
        bleu_scores.append(bleu_score)

    return np.mean(bleu_scores)

def compute_rouge_scores_api(tokenizer, dataset, model_engine, api_key):
    openai.api_key = api_key
    loader = DataLoader(dataset, batch_size=1)
    rouge = Rouge()
    rouge_scores = []

    for batch in loader:
        prompt = tokenizer.decode(batch[0], skip_special_tokens=True)
        response = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=1.0,
        )
        generated_text = response.choices[0].text.strip()
        scores = rouge.get_scores(generated_text, prompt, avg=True)
        rouge_scores.append(scores)

    rouge_l = np.mean([score["rouge-l"]["f"] for score in rouge_scores])
    rouge_1 = np.mean([score["rouge-1"]["f"] for score in rouge_scores])
    rouge_2 = np.mean([score["rouge-2"]["f"] for score in rouge_scores])

    return {"rouge-1": rouge_1, "rouge-2": rouge_2, "rouge-l": rouge_l}

def display_examples(tokenizer, dataset, model_engine, api_key, num_examples=5):
    openai.api_key = api_key
    loader = DataLoader(dataset, batch_size=1)

    for i, batch in enumerate(loader):
        if i >= num_examples:
            break

        prompt = tokenizer.decode(batch[0], skip_special_tokens=True)
        response = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=1.0,
        )
        generated_text = response.choices[0].text.strip()
        print(f"Example {i + 1}:")
        print(f"Input: {prompt}")
        print(f"Output: {generated_text}\n")

def visualize_scores(scores, title):
    plt.figure()
    plt.bar(range(len(scores)), list(scores.values()), align='center')
    plt.xticks(range(len(scores)), list(scores.keys()))
    plt.title(title)
    plt.show()

def parallel_compute_metric(metric_fn, args):
    metric, dataset, model_engine, api_key = args
    score = metric_fn(dataset, model_engine, api_key)
    return metric, score

def compute_metrics_parallel(tokenizer, dataset, model_engine, api_key, metrics, num_threads=5):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(parallel_compute_metric, metric_fn, (metric, dataset, model_engine, api_key)) for metric, metric_fn in metrics.items()]
        
        results = {}
        for future in as_completed(futures):
            metric, score = future.result()
            results[metric] = score
    
    return results

def custom_metric_example(dataset, model_engine, api_key):
    # Implement your custom evaluation metric here
    return 0.0

def generate_report(model_engine, bleu_score, rouge_scores, wer_score, file_path):
    report = "Model Evaluation Report\n"
    report += f"Model: {model_engine}\n"
    report += f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\n"

    report += "1. Introduction\n"
    report += "This report presents the evaluation results of the GPT model using the OpenAI API on a custom dataset.\n\n"

    report += "2. Dataset\n"
    report += "The dataset used for evaluation was obtained from [source]. It consists of [number of samples] samples and has undergone the following preprocessing steps: [describe preprocessing].\n\n"

    report += "3. Evaluation Metrics\n"
    report += "The evaluation metrics used in this report include:\n"
    report += "- BLEU Score: A metric that measures the similarity between the generated text and the reference text based on n-gram matches.\n"
    report += "- ROUGE Scores: A set of metrics (ROUGE-N, ROUGE-L, and ROUGE-W) that compare the generated text to the reference text based on overlapping n-grams, longest common subsequences, and weighted n-gram matches, respectively.\n"
    report += "- Word Error Rate (WER): A metric that measures the similarity between the generated text and the reference text by calculating the minimum number of edits (insertions, deletions, and substitutions) needed to transform one text into the other, divided by the total number of words in the reference text.\n"
    report += "- Custom Metric (if applicable): [brief explanation]\n\n"

    report += "4. Results\n"
    report += f"- BLEU Score: {bleu_score:.2f} ({bleu_score*100:.2f}%)\n" # higher is better
    report += f"- ROUGE-1 Score: {rouge_scores['rouge-1']:.2f} ({rouge_scores['rouge-1']*100:.2f}%)\n" # higher is better
    report += f"- ROUGE-2 Score: {rouge_scores['rouge-2']:.2f} ({rouge_scores['rouge-2']*100:.2f}%)\n"
    report += f"- ROUGE-L Score: {rouge_scores['rouge-l']:.2f} ({rouge_scores['rouge-l']*100:.2f}%)\n"
    report += f"- Word Error Rate (WER): {wer_score:.2f} ({wer_score*100:.2f}%)\n\n" # lower is better

    report += "5. Conclusion\n"
    report += "Based on the evaluation results, the GPT model shows [summary of performance]. [Any additional insights, recommendations, or conclusions].\n"

    with open(file_path, "w") as f:
        f.write(report)

    return report

def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT model using OpenAI API")
    parser.add_argument("--model_engine", type=str, default="gpt-3.5-0301", help="The name of the GPT model engine to use with the OpenAI API")
    parser.add_argument("--dataset_path", type=str, default="path/to/your/test-dataset.txt", help="Path to the evaluation dataset")
    parser.add_argument("--block_size", type=int, default=128, help="Block size for tokenization")
    parser.add_argument("--api_key", type=str, required=True, help="API key for OpenAI API")
    parser.add_argument("--report_file_path", type=str, default="path/to/your/report.txt", help="Path to save the evaluation report")
    parser.add_argument("--num_examples", type=int, default=5, help="Number of input-output examples to display")
    parser.add_argument("--num_threads", type=int, default=5, help="Number of threads to use for parallel computation")
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    dataset = TextDataset(args.dataset_path, tokenizer, args.block_size)

    # Display input-output examples
    print("Input-Output Examples:")
    display_examples(tokenizer, dataset, args.model_engine, args.api_key, args.num_examples)

    # Parallel computation of evaluation metrics
    metrics = {
        "BLEU": compute_bleu_score_api,
        "ROUGE": compute_rouge_scores_api,
        "WER": compute_wer_api,
        "CUSTOM": custom_metric_example
    }
    scores = compute_metrics_parallel(tokenizer, dataset, args.model_engine, args.api_key, metrics, args.num_threads)
    
    # Averaging results
    avg_score = np.mean(list(scores.values()))
    scores["AVERAGE"] = avg_score

    # Visualize scores
    visualize_scores(scores, "Evaluation Metrics")

    # Generate and display report
    report = generate_report(args.model_engine, scores['BLEU'], scores['ROUGE'], scores['WER'], args.report_file_path)
    print("\nGenerated report:")
    print(report)

if __name__ == "__main__":
    main()
