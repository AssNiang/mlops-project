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
        object_to_save:  object to save
        object_path: object handle path name

    Returns:
        None

    """

    logger.info(f"Starting object record in {object_path}")
    # create parent directory if needs
    Path(object_path).parent.mkdir(parents=True, exist_ok=True)

    pickle_out = open(object_path, "wb")
    pickle.dump(object_to_save, pickle_out)
    pickle_out.close()

    logger.info("Done object record successfully")
