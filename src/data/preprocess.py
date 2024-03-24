import os
from pathlib import Path
import datasets
import hydra
from omegaconf import DictConfig

from src.data import filters, transforms
from src.data.enums import Feature, Split
import textwrap
import random


@hydra.main(config_path="config", config_name="preprocess", version_base="1.3")
def main(cfg: DictConfig) -> None:

    # Instantiate the dataset
    print("Loading Dataset")
    dataset = hydra.utils.instantiate(cfg.dataset, _convert_="all")
    print(dataset)

    # Clean the text column
    dataset = dataset.map(
        transforms.clean_text,
        desc="Cleaning Text",
        input_columns=Feature.TEXT.value,
        fn_kwargs={"key": Feature.TEXT.value},
        batched=True,
    )

    # Clean the label column
    dataset = dataset.map(
        transforms.clean_text,
        desc="Cleaning Label",
        input_columns=Feature.LABEL.value,
        fn_kwargs={"key": Feature.LABEL.value},
        batched=True,
    )

    # Filter the dataset by text length
    dataset = dataset.filter(
        filters.keep_text,
        desc="Filtering Text",
        input_columns=Feature.TEXT.value,
        fn_kwargs={
            "min_length": cfg.get("min_text_length"),
        },
        batched=True,
    )

    # Filter the dataset by label
    dataset = dataset.filter(
        filters.keep_label,
        desc="Filtering Label",
        input_columns=Feature.LABEL.value,
        fn_kwargs={
            "include": cfg.get("include_labels"),
            "exclude": cfg.get("exclude_labels"),
        },
        batched=True,
    )

    # Remove Duplicates
    for split, data in dataset.items():
        print(f"Removing duplicates in {split=}")
        data = data.to_pandas()
        print("- Shape before:", data.shape)
        data = data.drop_duplicates(subset=[Feature.LABEL.value, Feature.TEXT.value])
        print("- Shape after:", data.shape)
        dataset[split] = datasets.Dataset.from_pandas(data)

    # Encode the labels
    dataset = dataset.class_encode_column(Feature.LABEL.value)

    # If eval split not in dataset, create it from train split
    if Split.EVAL.value not in dataset:
        print(
            f"Splitting '{Split.TRAIN}' split into '{Split.TRAIN}' (0.85) and '{Split.EVAL}' (0.15)"
        )
        splits = dataset[Split.TRAIN.value].train_test_split(
            0.15, stratify_by_column=Feature.LABEL.value
        )
        dataset[Split.TRAIN.value] = splits["train"]
        dataset[Split.EVAL.value] = splits["test"]

    # Save the cleaned dataset
    if not os.path.exists(cfg.output_path):
        os.makedirs(cfg.output_path, parents=True, exist_ok=True)

    print(f"Saving the dataset to {cfg.output_path}")
    dataset.save_to_disk(cfg.output_path)


if __name__ == "__main__":
    main()
