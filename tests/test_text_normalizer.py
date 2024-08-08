"""Test text normalizer Module"""

import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

#pylint: disable=wrong-import-position

from src.preprocess import text_normalizer

#pylint: enable=wrong-import-position


def test_text_normalizer():
    """Test text normalizer"""
    text_input = "We'll combine all functions into 1 SINGLE FUNCTION 🙂 & apply on @product #descriptions https://ept.sn/git/mlops"
    text_output = "combine function function apply product description"

    assert text_normalizer(text_input) == text_output
