"""Tests verifying LongBench scoring matches the official THUDM implementation."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.benchmark_longbench import (
    _normalize_answer,
    _f1_score_tokens,
    qa_f1_score,
    classification_score,
    code_sim_score,
    count_score,
    retrieval_score,
    score_task,
    build_prompt,
    DATASET2PROMPT,
    DATASET2MAXLEN,
    DATASET2METRIC,
    ALL_TASKS,
)


class TestNormalizeAnswer:
    """Verify normalize_answer matches official metrics.py."""

    def test_lowercase(self):
        assert _normalize_answer("Hello World") == "hello world"

    def test_remove_articles(self):
        assert _normalize_answer("the cat and a dog") == "cat and dog"

    def test_remove_punctuation(self):
        assert _normalize_answer("hello, world!") == "hello world"

    def test_whitespace_fix(self):
        assert _normalize_answer("  hello   world  ") == "hello world"

    def test_combined(self):
        assert _normalize_answer("The quick, brown fox!") == "quick brown fox"


class TestQAF1:
    """Verify qa_f1_score uses Counter intersection, not set intersection."""

    def test_exact_match(self):
        assert qa_f1_score("Paris", "Paris") == 1.0

    def test_partial_match(self):
        score = qa_f1_score("the capital is Paris France", "Paris")
        assert 0.0 < score < 1.0

    def test_no_match(self):
        assert qa_f1_score("London", "Paris") == 0.0

    def test_counter_vs_set(self):
        """Counter handles duplicates correctly (set would not)."""
        # "cat cat cat" → normalized: "cat cat cat" (3 tokens)
        # "cat dog" → normalized: "cat dog" (2 tokens)
        # Counter intersection: {"cat": 1} → num_same=1
        # precision = 1/3, recall = 1/2, f1 = 2*(1/3)*(1/2)/((1/3)+(1/2)) = 0.4
        score = qa_f1_score("cat cat cat", "cat dog")
        assert abs(score - 0.4) < 0.01

    def test_normalization_applied(self):
        """Articles, punctuation removed before scoring."""
        # "The answer is: Paris!" → "answer is paris"
        # "paris" → "paris"
        score = qa_f1_score("The answer is: Paris!", "Paris")
        assert score > 0.0


class TestClassification:
    def test_exact_class_match(self):
        score = classification_score("NUM", "NUM", all_classes=["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"])
        assert score == 1.0

    def test_no_match(self):
        score = classification_score("something else", "NUM", all_classes=["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"])
        assert score == 0.0

    def test_multiple_classes_in_prediction(self):
        """If prediction contains multiple classes, score = 1/n_matches."""
        score = classification_score("NUM and HUM", "NUM", all_classes=["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"])
        assert score == 0.5


class TestCountScore:
    def test_correct_count(self):
        assert count_score("There are 5 unique paragraphs", "5") == 1.0

    def test_wrong_count(self):
        assert count_score("There are 3 unique paragraphs", "5") == 0.0

    def test_multiple_numbers(self):
        score = count_score("I found 5 and then 3 more, so 5", "5")
        assert abs(score - 2.0 / 3.0) < 0.01


class TestRetrievalScore:
    def test_correct_paragraph(self):
        assert retrieval_score("Paragraph 3", "Paragraph 3") == 1.0

    def test_wrong_paragraph(self):
        assert retrieval_score("Paragraph 5", "Paragraph 3") == 0.0


class TestScoreTask:
    """Verify score_task dispatches to the right metric per dataset."""

    def test_qa_task_uses_f1(self):
        score = score_task("hotpotqa", ["Paris"], [["Paris"]])
        assert score == 100.0

    def test_summarization_task(self):
        try:
            from rouge import Rouge
            has_rouge = True
        except ImportError:
            has_rouge = False
        score = score_task("gov_report", ["the quick brown fox"], [["the quick brown fox"]])
        if has_rouge:
            assert score > 0.0
        else:
            # Without rouge package, score falls back to 0.0 — expected
            assert score == 0.0

    def test_classification_task(self):
        score = score_task("trec", ["NUM"], [["NUM"]], all_classes=["NUM", "LOC"])
        assert score == 100.0

    def test_count_task(self):
        score = score_task("passage_count", ["5"], [["5"]])
        assert score == 100.0

    def test_retrieval_task(self):
        score = score_task("passage_retrieval_en", ["Paragraph 7"], [["Paragraph 7"]])
        assert score == 100.0

    def test_first_line_extraction(self):
        """Few-shot tasks extract first line from prediction."""
        # triviaqa is in FIRST_LINE_TASKS
        score = score_task("triviaqa", ["Paris\nsome other stuff"], [["Paris"]])
        assert score == 100.0


class TestPromptTemplates:
    """Verify all 16 tasks have prompt templates and max lengths."""

    def test_all_tasks_have_prompts(self):
        for task in ALL_TASKS:
            assert task in DATASET2PROMPT, f"Missing prompt template for {task}"

    def test_all_tasks_have_maxlen(self):
        for task in ALL_TASKS:
            assert task in DATASET2MAXLEN, f"Missing max gen length for {task}"

    def test_all_tasks_have_metrics(self):
        for task in ALL_TASKS:
            assert task in DATASET2METRIC, f"Missing metric for {task}"

    def test_prompt_uses_context_and_input(self):
        sample = {"context": "Some context here", "input": "What is the answer?"}
        prompt = build_prompt("hotpotqa", sample)
        assert "Some context here" in prompt
        assert "What is the answer?" in prompt

    def test_code_prompt_format(self):
        sample = {"context": "def foo():\n    return 1\n", "input": ""}
        prompt = build_prompt("lcc", sample)
        assert "def foo():" in prompt
        assert "Next line of code:" in prompt
