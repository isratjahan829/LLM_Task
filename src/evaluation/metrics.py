"""Metric computations for evaluating PDF-based RAG model outputs."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, Optional

import numpy as np


@dataclass(slots=True)
class MetricsConfig:
    """Configuration for text quality metrics."""

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    bert_score_model: str = "microsoft/deberta-base-mnli"
    bert_score_lang: str = "en"
    device: Optional[str] = None
    meteor_alpha: float = 0.9
    meteor_beta: float = 3.0
    meteor_gamma: float = 0.5
    lower_case: bool = True


class MetricsComputer:
    """Compute a set of evaluation metrics for RAG responses."""

    def __init__(self, config: MetricsConfig | None = None) -> None:
        self.config = config or MetricsConfig()
        self._ensure_nltk_resources()
        self._embedder = None
        self._bertscorer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_all(
        self,
        reference_answer: str,
        prediction: str,
        *,
        question: Optional[str] = None,
        reference_context: Optional[str] = None,
        retrieved_context: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute the full suite of metrics for a single sample."""

        normalized_ref = self._normalize(reference_answer)
        normalized_pred = self._normalize(prediction)
        normalized_question = self._normalize(question) if question else ""
        normalized_ref_context = self._normalize(reference_context) if reference_context else ""
        normalized_retrieved_context = (
            self._normalize(retrieved_context)
            if retrieved_context
            else normalized_ref_context
        )

        ref_tokens = self._tokenize(normalized_ref)
        pred_tokens = self._tokenize(normalized_pred)
        question_tokens = self._tokenize(normalized_question) if normalized_question else []
        reference_context_tokens = (
            self._tokenize(normalized_ref_context) if normalized_ref_context else []
        )
        retrieved_context_tokens = (
            self._tokenize(normalized_retrieved_context)
            if normalized_retrieved_context
            else reference_context_tokens
        )

        precision, recall, f1 = self._precision_recall_f1(ref_tokens, pred_tokens)
        cosine_similarity = self._cosine_similarity(normalized_ref, normalized_pred)
        bleu_score = self._bleu_score(ref_tokens, pred_tokens)
        meteor_score = self._meteor_score(normalized_ref, normalized_pred)
        bert_f1 = self._bert_f1(normalized_ref, normalized_pred)
        completeness = self._completeness(ref_tokens, pred_tokens)
        hallucination = self._hallucination(pred_tokens, retrieved_context_tokens)
        irrelevance = self._irrelevance(pred_tokens, question_tokens)

        response_length = len(pred_tokens)
        response_char_length = len(normalized_pred)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "cosine_similarity": cosine_similarity,
            "bleu": bleu_score,
            "meteor": meteor_score,
            "bert_score_f1": bert_f1,
            "completeness": completeness,
            "hallucination": hallucination,
            "irrelevance": irrelevance,
            "response_length": float(response_length),
            "response_char_length": float(response_char_length),
        }

    # ------------------------------------------------------------------
    # Individual metric helpers
    # ------------------------------------------------------------------
    def _cosine_similarity(self, reference: str, prediction: str) -> float:
        if not reference.strip() or not prediction.strip():
            return 0.0

        embedder = self._get_embedder()
        ref_vec, pred_vec = embedder.encode([reference, prediction], convert_to_numpy=True)
        numerator = float(np.dot(ref_vec, pred_vec))
        denominator = float(np.linalg.norm(ref_vec) * np.linalg.norm(pred_vec))
        if denominator == 0.0:
            return 0.0
        return numerator / denominator

    def _precision_recall_f1(self, reference_tokens: Iterable[str], prediction_tokens: Iterable[str]):
        ref_list = list(reference_tokens)
        pred_list = list(prediction_tokens)
        if not ref_list and not pred_list:
            return 1.0, 1.0, 1.0
        if not ref_list or not pred_list:
            return 0.0, 0.0, 0.0

        ref_counts = self._token_counts(ref_list)
        pred_counts = self._token_counts(pred_list)

        true_positive = sum(min(pred_counts[token], ref_counts.get(token, 0)) for token in pred_counts)
        precision = true_positive / sum(pred_counts.values()) if pred_counts else 0.0
        recall = true_positive / sum(ref_counts.values()) if ref_counts else 0.0

        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def _bleu_score(self, reference_tokens: Iterable[str], prediction_tokens: Iterable[str]) -> float:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        ref_list = list(reference_tokens)
        pred_list = list(prediction_tokens)

        if not ref_list or not pred_list:
            return 0.0

        smoothing_function = SmoothingFunction().method1
        try:
            return float(sentence_bleu([ref_list], pred_list, smoothing_function=smoothing_function))
        except ZeroDivisionError:
            return 0.0

    def _meteor_score(self, reference: str, prediction: str) -> float:
        from nltk.translate.meteor_score import meteor_score

        if not reference.strip() or not prediction.strip():
            return 0.0

        return float(
            meteor_score(
                [reference],
                prediction,
                alpha=self.config.meteor_alpha,
                beta=self.config.meteor_beta,
                gamma=self.config.meteor_gamma,
            )
        )

    def _bert_f1(self, reference: str, prediction: str) -> float:
        if not reference.strip() or not prediction.strip():
            return 0.0

        scorer = self._get_bertscorer()
        _, _, f1_scores = scorer.score([prediction], [reference])
        return float(f1_scores[0])

    def _completeness(self, reference_tokens: Iterable[str], prediction_tokens: Iterable[str]) -> float:
        ref_list = list(reference_tokens)
        pred_list = list(prediction_tokens)
        if not ref_list:
            return 0.0

        ref_counts = self._token_counts(ref_list)
        pred_counts = self._token_counts(pred_list)
        covered = sum(min(ref_counts[token], pred_counts.get(token, 0)) for token in ref_counts)
        return covered / sum(ref_counts.values())

    def _hallucination(self, prediction_tokens: Iterable[str], context_tokens: Iterable[str]) -> float:
        pred_list = list(prediction_tokens)
        if not pred_list:
            return 0.0

        context_counts = self._token_counts(context_tokens)
        pred_counts = self._token_counts(pred_list)
        supported = sum(min(count, context_counts.get(token, 0)) for token, count in pred_counts.items())
        hallucinated = sum(pred_counts.values()) - supported
        rate = hallucinated / sum(pred_counts.values())
        return max(0.0, min(1.0, rate))

    def _irrelevance(self, prediction_tokens: Iterable[str], question_tokens: Iterable[str]) -> float:
        pred_list = list(prediction_tokens)
        if not pred_list:
            return 0.0
        if not question_tokens:
            return 0.0

        question_counts = self._token_counts(question_tokens)
        pred_counts = self._token_counts(pred_list)
        relevant = sum(min(count, question_counts.get(token, 0)) for token, count in pred_counts.items())
        irrelevant = sum(pred_counts.values()) - relevant
        rate = irrelevant / sum(pred_counts.values())
        return max(0.0, min(1.0, rate))

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _normalize(self, text: Optional[str]) -> str:
        if not text:
            return ""
        normalized = text.strip()
        if self.config.lower_case:
            normalized = normalized.lower()
        return normalized

    def _tokenize(self, text: str) -> Iterable[str]:
        if not text:
            return []
        from nltk import word_tokenize

        return [token for token in word_tokenize(text) if token.strip()]

    @staticmethod
    def _token_counts(tokens: Iterable[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        return counts

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(
                self.config.embedding_model,
                device=self.config.device,
            )
        return self._embedder

    def _get_bertscorer(self):
        if self._bertscorer is None:
            from bert_score import BERTScorer

            self._bertscorer = BERTScorer(
                model_type=self.config.bert_score_model,
                lang=self.config.bert_score_lang,
                device=self.config.device,
                rescale_with_baseline=True,
            )
        return self._bertscorer

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------
    @staticmethod
    @lru_cache(maxsize=1)
    def _ensure_nltk_resources():
        import nltk

        resources = [
            ("tokenizers/punkt", "punkt"),
            ("corpora/wordnet", "wordnet"),
            ("corpora/omw-1.4", "omw-1.4"),
        ]

        for resource_path, resource_name in resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                nltk.download(resource_name, quiet=True)


def mean(values: Iterable[Optional[float]]) -> float:
    filtered = [v for v in values if v is not None and not math.isnan(v)]
    if not filtered:
        return 0.0
    return statistics.mean(filtered)
