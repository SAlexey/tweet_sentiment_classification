import re
from src.data.utils import dictwrap, batched

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_HTML_PATTERN = re.compile(r"<.*?>")
_USER_TAG_PATTERN = re.compile(r"@ ?[a-zA-Z0-9_]+")
_SEPARATOR_PATTERN = re.compile(r"\s+")


@dictwrap
@batched
def clean_text(text: str, *args, **kwargs) -> str:
    """
    Clean text data

    Args:
        text (str): The text data

    Returns:
        str: The cleaned text data
    """

    # Safeguard
    text = str(text or "")

    # To lower case
    text = text.lower()

    # Remove URLs
    text = _URL_PATTERN.sub("", text)

    # Remove HTML tags
    text = _HTML_PATTERN.sub("", text)

    # Remove user tags
    text = _USER_TAG_PATTERN.sub("", text)

    # Remove extra spaces
    text = _SEPARATOR_PATTERN.sub(" ", text)

    # Remove non-ASCII characters
    text = text.encode("ascii", "ignore").decode()

    return text.strip()
