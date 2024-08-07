"""Test dataset."""

import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))
from src.make_dataset import load_data

# load data
data = load_data("data/ecommerceDataset.csv")


def test_shape():
    """Test data shape."""
    nrows, ncols = data.shape

    assert nrows >= 27802
    assert ncols == 2


# def test_saleprice():
#     """Target feature."""
#     assert sum(data["SalePrice"].isnull()) == 0
#     assert sum(data["SalePrice"] <= 0) == 0
