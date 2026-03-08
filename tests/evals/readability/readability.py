from __future__ import annotations

from dataclasses import dataclass

import textstat


@dataclass(frozen=True)
class ReadabilityScores:
    """
    Class to hold readability scores
    """
    sentence_count: int
    word_count: int
    syllable_count: int
    avg_words_per_sentence: float
    avg_syllables_per_word: float
    flesch_reading_ease: float
    flesch_kincaid_grade: float


class ReadabilityCalculator:
    """
    Class to calculate readability scores for a given text.
    """
    @staticmethod
    def score_readability(text: str) -> ReadabilityScores:
        """Calculate readability scores for the input text."""
        sentence_count = max(1, int(textstat.sentence_count(text)))
        word_count = max(1, int(textstat.lexicon_count(text, removepunct=True)))
        syllable_count = max(1, int(textstat.syllable_count(text)))

        avg_words_per_sentence = word_count / sentence_count
        avg_syllables_per_word = syllable_count / word_count
        flesch_reading_ease = float(textstat.flesch_reading_ease(text))
        flesch_kincaid_grade = float(textstat.flesch_kincaid_grade(text))

        return ReadabilityScores(
            sentence_count=sentence_count,
            word_count=word_count,
            syllable_count=syllable_count,
            avg_words_per_sentence=avg_words_per_sentence,
            avg_syllables_per_word=avg_syllables_per_word,
            flesch_reading_ease=flesch_reading_ease,
            flesch_kincaid_grade=flesch_kincaid_grade,
        )
