from hashlib import md5
import tiktoken
import html
import re
import string
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
import numpy as np
from collections import Counter
from utils.logging import logger
import os
import json
import re

def encode_string_by_tiktoken(content: str, model_name: str = "cl100k_base"):
    ENCODER = tiktoken.get_encoding(model_name)
    tokens = ENCODER.encode(content)
    return tokens

def decode_string_by_tiktoken(tokens: list[int], model_name: str = "cl100k_base"):
    ENCODER = tiktoken.get_encoding(model_name)
    string = ENCODER.decode(tokens)
    return string

def mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()

def clean_str(input) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    result = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)

    # Remove non-alphanumeric characters and convert to lowercase
    return re.sub('[^A-Za-z0-9 ]', ' ', result.lower()).strip()

def normalize_name(name):
    suffixes = ["'s", "s'", "'"]
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[:-len(suffix)].strip()
    return name

def normalize_answer(answer: str) -> str:
    """
    Normalize a given string by applying the following transformations:
    1. Convert the string to lowercase.
    2. Remove punctuation characters.
    3. Remove the articles "a", "an", and "the".
    4. Normalize whitespace by collapsing multiple spaces into one.

    Args:
        answer (str): The input string to be normalized.

    Returns:
        str: The normalized string.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(answer))))

def calculate_metric_scores(global_config, results: List[Dict[str, str]], aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Calculates the F1 score and Accuracy

    Args:
        gold_answers (List[List[str]]): List of lists containing ground truth answers.
        predicted_answers (List[str]): List of predicted answers.
        aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

    Returns:
        Tuple[Dict[str, float], List[Dict[str, float]]]: 
            - A dictionary with the averaged F1 score.
            - A list of dictionaries with F1 scores for each example.
    """
    
    gold_answers: List[List[str]] = []
    predicted_answers: List[str] = []
    
    for item in results:
        if isinstance(item["answer"], str):
            gold_answers.append([item["answer"]])
        else:
            gold_answers.append(item["answer"])
        
        try:
            pre_answer = item["generation"][1] # qid is item["generation"][0], the generation is item["generation"][1]
            match = re.search(r'ANSWER:\s*(.*)', pre_answer) # When the model's capabilities are limited—such as with LLaMA3-8B—it inevitably generates outputs that are irrelevant to our requirements.
            if match:
                pre_answer = match.group(1)
            else:
                pre_answer = item["generation"][1]
        
            predicted_answers.append(pre_answer)
        except:
            breakpoint()
    
    assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."

    def compute(gold: str, predicted: str) -> float:
        normalize_gold = normalize_answer(gold)
        normalize_pred = normalize_answer(predicted)
        gold_tokens = normalize_gold.split()
        predicted_tokens = normalize_pred.split()
        common = Counter(predicted_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return {"F1":0.0, "ACC": 0.0}

        precision = 1.0 * num_same / len(predicted_tokens)
        recall = 1.0 * num_same / len(gold_tokens)
        acc = 1.0 if normalize_gold in normalize_pred else 0.0
        return {"F1" : 2 * (precision * recall) / (precision + recall), "ACC": acc}

    example_eval_results = []
    total_f1 = 0.0
    total_acc = 0.0
    lis = list(zip(gold_answers, predicted_answers))

    for idx, item in enumerate(lis):
        gold_list, predicted = item
        scores = [compute(gold, predicted) for gold in gold_list]
        aggregated_f1 = aggregation_fn([score["F1"] for score in scores])
        aggregated_Acc = aggregation_fn([score["ACC"] for score in scores])
        
        example_eval_results.append({"qid":results[idx]["qid"], "question":results[idx]["question"],
                                     "answer":results[idx]["answer"], "generation": predicted_answers[idx],
                                     "F1": aggregated_f1, "ACC": aggregated_Acc})
        total_f1 += aggregated_f1
        total_acc += aggregated_Acc

    avg_f1 = total_f1 / len(gold_answers) if gold_answers else 0.0
    avg_acc = total_acc / len(gold_answers) if gold_answers else 0.0
    pooled_eval_results = {"F1": float(avg_f1), "ACC": float(avg_acc)}
    
    # Save detailed results to JSON file
    save_dir = os.path.join(global_config.save_dir, f"{global_config.select_metric}_{global_config.alpha:.2f}")
    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, "results.json")
    try:
        with open(output_file, 'w') as f:
            json.dump({
                "pooled_results": pooled_eval_results,
                "specific_results": example_eval_results
            }, f, indent=2)
        if logger:
            logger.info(f"Successfully saved detailed evaluation results to {output_file}")
    except IOError as e:
        if logger:
            logger.error(f"Failed to save evaluation results: {str(e)}")
        raise
    
    logger.info(f"Final results: {pooled_eval_results}")
    
    return pooled_eval_results, example_eval_results

def clean_text(text):
    cleaned = re.sub(r'^```', '', text)
    cleaned = re.sub(r'```$', '', cleaned)

    return cleaned.strip()


def log_tokens(mode: str, res_metadata: Dict[str, Any]):
    """Log detailed generation metrics from the LLM responses"""
    if not res_metadata:
        logger.warning("No generation metadata available")
        return
    total_metadata = {"total_prompt_tokens": res_metadata["prompt_tokens"], 
                            "total_completion_tokens": res_metadata["completion_tokens"],
                            "total_tokens": res_metadata["total_tokens"]}
    logger.info(f"Total token usage during {mode}: {total_metadata}")