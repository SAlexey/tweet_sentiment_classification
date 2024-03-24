import enum


class FileType(enum.StrEnum):
    """
    The file types
    """

    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"


class Split(enum.StrEnum):
    """
    The dataset splits
    """

    TRAIN = "train"
    EVAL = "eval"
    TEST = "test"


class Feature(enum.StrEnum):
    """
    The dataset features
    """

    TEXT = "text"
    LABEL = "label"
