"""Test text normalizer."""

import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))
from src.preprocess import text_normalizer


def test_text_normalizer():
    """Test text normalizer"""
    text_input = "We'll combine all functions into 1 SINGLE FUNCTION 🙂 & apply on @product #descriptions https://en.wikipedia.org/wiki/Text_normalization"
    text_output = "combine function function apply product description"

    assert text_normalizer(text_input) == text_output
