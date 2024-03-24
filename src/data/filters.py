from typing import List, Optional, Union
from src.data.utils import batched


@batched
def keep_text(text: str, *args, min_length: int = -1) -> Union[bool, List[bool]]:
    """
    Keep text data

    Args:
        text (str): The text data
        min_length (int): The minimum length of the text data. Defaults to -1.

    Returns:
        bool: Whether to keep the text data
    """

    # Safeguard
    text = str(text or "")

    # Keep the text if it is not empty
    keep = bool(text)

    # Kepp the text if it has a minimum length
    keep = keep and (len(text) >= min_length)

    return keep


@batched
def keep_label(
    label: str,
    *args,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None
) -> Union[bool, List[bool]]:
    """
    Keep label data

    Args:
        label (str): The label data
        include (List[str]): A list of labels to include. Defaults to None.
        exclude (List[str]): A list of labels to exclude. Defaults to None.

    Returns:
        bool: Whether to keep the label data
    """

    # Safeguard
    label = str(label or "")

    # Keep label if it is not empty
    keep = bool(label)

    # Keep label if it is in include list
    if include:
        keep = keep and (label in include)

    # Keep label if it is not in exclude list
    if exclude:
        keep = keep and (label not in exclude)

    return keep
