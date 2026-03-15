"""
LongBench v1 scoring library — official THUDM scoring functions and prompt templates.

Faithful to the official THUDM/LongBench v1 pipeline:
  - Official prompt templates per dataset (dataset2prompt)
  - Official max generation lengths per dataset (dataset2maxlen)
  - Official scoring functions: qa_f1_score, rouge_score, classification_score,
    code_sim_score, count_score, retrieval_score (from metrics.py)
  - Official post-processing: normalize_answer, first-line extraction for few-shot

Validated by 29 unit tests in tests/test_longbench_scoring.py.
Used by benchmarks/benchmark_mlx_longbench.py for paper evaluation.
"""

import re
import string
from collections import Counter
from typing import Optional


# =============================================================================
# Official LongBench v1 prompt templates (from config/dataset2prompt.json)
# =============================================================================

DATASET2PROMPT = {
    "narrativeqa": 'You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    "qasper": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": 'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: ',
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}

# Official max generation lengths per dataset (from config/dataset2maxlen.json)
DATASET2MAXLEN = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "lcc": 64,
    "repobench-p": 64,
}


# =============================================================================
# Official LongBench v1 scoring (from metrics.py)
# =============================================================================


def _normalize_answer(s: str) -> str:
    """Lower text, remove punctuation, articles and extra whitespace.

    Exact match of THUDM/LongBench/LongBench/metrics.py normalize_answer.
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

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _f1_score_tokens(prediction_tokens: list, ground_truth_tokens: list) -> float:
    """Token-level F1 using Counter intersection (handles duplicates).

    Exact match of THUDM/LongBench/LongBench/metrics.py f1_score.
    """
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """QA F1 with normalize_answer. Faithful to official metrics.py."""
    normalized_prediction = _normalize_answer(prediction)
    normalized_ground_truth = _normalize_answer(ground_truth)
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return _f1_score_tokens(prediction_tokens, ground_truth_tokens)


def rouge_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """ROUGE-L F1. Faithful to official metrics.py (uses 'rouge' package)."""
    try:
        from rouge import Rouge

        rouge = Rouge()
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
        return scores["rouge-l"]["f"]
    except Exception:
        return 0.0


def classification_score(
    prediction: str, ground_truth: str, all_classes: list = None, **kwargs
) -> float:
    """Classification accuracy. Faithful to official metrics.py."""
    if all_classes is None:
        all_classes = []
    em_match_list = []
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in list(em_match_list):
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def code_sim_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Code similarity via fuzz.ratio. Faithful to official metrics.py."""
    try:
        from fuzzywuzzy import fuzz
    except ImportError:
        import difflib

        return difflib.SequenceMatcher(None, prediction, ground_truth).ratio()

    all_lines = prediction.lstrip("\n").split("\n")
    prediction_line = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            prediction_line = line
            break
    return fuzz.ratio(prediction_line, ground_truth) / 100


def count_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Count matching score. Faithful to official metrics.py."""
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth))
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


def retrieval_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Retrieval score (paragraph number matching). Faithful to official metrics.py."""
    pattern = r"Paragraph (\d+)"
    matches = re.findall(pattern, ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth_id))
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


# Official dataset → metric mapping (from eval.py)
DATASET2METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "passage_count": count_score,
    "passage_retrieval_en": retrieval_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

# Tasks that need first-line extraction (official eval.py post-processing)
FIRST_LINE_TASKS = {"trec", "triviaqa", "samsum", "lsht"}


def score_task(
    task_name: str,
    predictions: list[str],
    references: list[list[str]],
    all_classes: Optional[list[str]] = None,
) -> float:
    """Score predictions using official LongBench v1 scoring.

    Faithful to THUDM/LongBench/LongBench/eval.py scorer().
    """
    if not predictions:
        return 0.0

    metric_fn = DATASET2METRIC.get(task_name, qa_f1_score)

    total_score = 0.0
    for prediction, ground_truths in zip(predictions, references):
        # Official post-processing: extract first line for few-shot tasks
        if task_name in FIRST_LINE_TASKS:
            prediction = prediction.lstrip("\n").split("\n")[0]

        score = 0.0
        for ground_truth in ground_truths:
            score = max(
                score,
                metric_fn(
                    prediction, ground_truth, all_classes=all_classes or []
                ),
            )
        total_score += score

    return round(100 * total_score / len(predictions), 2)


# -- LongBench task definitions -----------------------------------------------

TASK_CATEGORIES = {
    "single_doc_qa": ["narrativeqa", "qasper", "multifieldqa_en"],
    "multi_doc_qa": ["hotpotqa", "2wikimqa", "musique"],
    "summarization": ["gov_report", "multi_news", "qmsum"],
    "few_shot": ["trec", "triviaqa", "samsum"],
    "synthetic": ["passage_count", "passage_retrieval_en"],
    "code": ["lcc", "repobench-p"],
}

ALL_TASKS = [t for tasks in TASK_CATEGORIES.values() for t in tasks]

# TREC classification classes (official)
TREC_CLASSES = ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]


def build_prompt(task_name: str, sample: dict) -> str:
    """Format a LongBench prompt from template + sample."""
    template = DATASET2PROMPT[task_name]
    return template.format(
        context=sample.get("context", ""),
        input=sample.get("input", ""),
    )
