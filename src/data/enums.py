import enum


class FileType(enum.Enum):
    """
    The file types
    """

    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"


class Split(enum.Enum):
    """
    The dataset splits
    """

    TRAIN = "train"
    EVAL = "eval"
    TEST = "test"


class Feature(enum.Enum):
    """
    The dataset features
    """

    TEXT = "text"
    LABEL = "label"
