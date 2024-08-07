"""Utility functions"""

from pathlib import Path
from typing import Any, Union
import pickle
from loguru import logger

def save_object_with_pickle(
    object_to_save: Any,
    object_path: Union[str, Path],
):
    """Save an object with pickle.

    Args:
        object_to_save: Object to save.
        object_path: Path to save the object.

    Returns:
        None
    """
    logger.info(f"Starting object record in {object_path}")
    # Create parent directory if needed
    Path(object_path).parent.mkdir(parents=True, exist_ok=True)

    # Save the object with pickle using 'with' for resource management
    with open(object_path, "wb") as pickle_out:
        pickle.dump(object_to_save, pickle_out)

    logger.info("Done object record successfully")
