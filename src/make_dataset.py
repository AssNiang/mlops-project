"""Module for loading data"""

from typing import Optional
import pandas as pd
from loguru import logger

# import sys
# from pathlib import Path
# sys.path.append(str(Path.cwd()))


def load_data(
    dataset_filepath: str, columns_to_lower: Optional[bool] = False
) -> pd.DataFrame:
    """Fetch dataset from a csv file by the given dataset_filepath

    Args:
        dataset_filepath (str): dataset path to load
        columns_to_lower (Optional[bool]): default is False
            flag to know if we should transform column names to lower

    Returns:
        pd.DataFrame: feature and target data

    """
    data = pd.DataFrame()
    logger.info(f"Dataset to load: {dataset_filepath}")
    if not dataset_filepath:
        raise ValueError("Dataset name, like ``dataset_name``, must be defined!")
    data = pd.read_csv(f"{dataset_filepath}", names=["label", "description"])
    data = data[["description", "label"]]

    # Calculate memory usage and dataset shape
    memory_usage = "{:.2f} MB".format(data.memory_usage().sum() / (1024 * 1024))
    # Log memory usage
    logger.info(f"Memory usage: {memory_usage}")
    # Log dataset shape
    logger.info(f"Dataset shape: {data.shape}")

    if columns_to_lower:
        logger.info("Columns will be transformed to lower!")
        data.columns = data.columns.str.lower()
    return data
