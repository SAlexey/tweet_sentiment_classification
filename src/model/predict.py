import transformers


def load_model(model_name_or_path: str) -> transformers.TextClassificationPipeline:
    """
    Load the text classification pipeline

    Args:
        model_dir (str): The directory of the model

    Returns:
        pipeline: The text classification pipeline
    """

    return transformers.pipeline(
        "text-classification",
        model=model_name_or_path,
        tokenizer=model_name_or_path,
    )
