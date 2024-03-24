import logging
import os

import datasets
import hydra
import pandas as pd
import torch
import transformers as hf
from datasets.features import ClassLabel
from omegaconf import DictConfig
from sklearn.metrics import precision_recall_fscore_support

from src.data.enums import Split

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    """
    Flatten a nested dictionary.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The parent key.
        sep (str): The separator.

    Returns:
        dict: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def preprocess_logits_for_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """
    Preprocess the logits for computing metrics.

    Args:
        logits (torch.Tensor): The model logits.
        labels (torch.Tensor): The labels.

    Returns:
        torch.Tensor: The predicted class labels.
    """
    class_labels = logits.argmax(axis=1)
    return class_labels


def compute_metrics(
    eval_predictions: hf.trainer_utils.EvalPrediction,
) -> pd.DataFrame:
    """
    Compute metrics for the given predictions and labels.

    Args:
        eval_predictions (hf.trainer_utils.EvalPrediction): The evaluation predictions.
    Returns:
        pd.DataFrame: Metrics for each category and averages.
    """

    names = ("precision", "recall", "f1_score", "support")
    averages = ("micro", "macro", "weighted")

    y_pred = eval_predictions.predictions
    y_true = eval_predictions.label_ids

    # compute metrics for each category
    metrics = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    metrics = pd.DataFrame(metrics, index=names)

    # safe guard agains error in Transformrs Trainer
    # the Trainer expects dictionary keys to be strings
    # but the class labels have been encoded as integers
    metrics.columns = [f"label_{col}" for col in metrics.columns]

    # compute average metrics
    average_metrics = pd.concat(
        [
            pd.Series(
                precision_recall_fscore_support(
                    y_true, y_pred, average=average, zero_division=0
                ),
                index=names,
            )
            for average in averages
        ],
        axis=1,
    )
    average_metrics.columns = [f"{name}_avg" for name in averages]

    metrics = pd.concat([metrics, average_metrics], axis=1)

    return metrics.to_dict(orient="index")


def prepare_dataset(
    dataset: datasets.Dataset,
    tokenizer: hf.PreTrainedTokenizer,
    padding: bool | str | None = None,
    truncation: bool | None = True,
    max_length: int | None = None,
    **kwargs,
):
    """
    Prepare a dataset for training.

    Args:
        dataset (datasets.Dataset): The dataset to prepare.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        padding (bool | str | None): The padding strategy.
        truncation (bool | None): The truncation strategy.
        max_length (int | None): The maximum length of the input sequence.

    Returns:
        datasets.Dataset: The prepared dataset.
    """

    if padding == "max_length" and max_length is None:
        max_length = tokenizer.model_max_length

    fn_kwargs = {"max_length": max_length, "padding": padding, "truncation": truncation}
    dataset = dataset.map(tokenizer, fn_kwargs=fn_kwargs, **kwargs)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return dataset


@hydra.main(config_path="config", config_name="train", version_base="1.3")
def main(config: DictConfig) -> None:

    # Load the dataset
    dataset = datasets.load_from_disk(config.dataset_path)

    if isinstance(dataset, datasets.DatasetDict):
        train_dataset = dataset[Split.TRAIN.value]
        eval_dataset = dataset.get(Split.EVAL.value)
        test_dataset = dataset.get(Split.TEST.value)
    else:
        train_dataset = dataset
        eval_dataset = None
        test_dataset = None

    class_label = train_dataset.features[config.label_name]

    if not isinstance(class_label, ClassLabel):
        raise ValueError(
            f"Expected `{config.label_name}` for be of type `datasets.features.ClassLabel`, got {type(class_label)}"
        )

    num_labels = len(class_label.names)
    label_counts = pd.Series(train_dataset[config.label_name])
    label_counts = label_counts.value_counts().sort_index()
    class_weights = label_counts.sum() / label_counts
    class_weights = class_weights.sort_index().tolist()
    id2label = {i: label for i, label in enumerate(class_label.names)}
    label2id = {label: i for i, label in enumerate(class_label.names)}

    logger.info("Label description:")
    logger.info(f"Number of labels: {num_labels}")
    logger.info(f"Label counts: {label_counts}")
    logger.info(f"Class weights: {class_weights}")
    logger.info(f"Label to id: {label2id}")
    logger.info(f"Id to label: {id2label}")

    model = hf.AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    tokenizer = hf.AutoTokenizer.from_pretrained(config.tokenizer_name)

    train_dataset = prepare_dataset(
        train_dataset,
        tokenizer=tokenizer,
        padding=config.padding,
        truncation=config.truncation,
        max_length=config.max_length,
        batched=True,
        input_columns=config.text_name,
        desc="Preparing train dataset for training",
    )

    if eval_dataset:
        eval_dataset = prepare_dataset(
            eval_dataset,
            tokenizer=tokenizer,
            padding=config.padding,
            truncation=config.truncation,
            max_length=config.max_length,
            batched=True,
            input_columns=config.text_name,
            desc="Preparing eval dataset for training",
        )

    if limit := config.get("limit_eval", None):
        if isinstance(limit, int):
            limit = round(limit / len(eval_dataset), 3)

        eval_dataset = eval_dataset.train_test_split(test_size=limit)
        eval_dataset = eval_dataset["test"]

    training_args = hf.TrainingArguments(**config.training_args)

    trainer = hf.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
    )

    if config.early_stopping:
        trainer.add_callback(
            hf.EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold,
            )
        )

    logger.info("Training model ...")
    train_results = trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint
    )
    logger.info("Training finished.")
    logger.info("Saving train metrics ...")
    trainer.save_metrics(Split.TRAIN.value, train_results.metrics)
    trainer.log_metrics(Split.TRAIN.value, flatten_dict(train_results.metrics))

    # Save the best model
    if training_args.load_best_model_at_end:
        logger.info("Loaded the best model at the end of training.")
        logger.info(f"Saving model to {config.model_save_path} ... ")
        trainer.save_model(config.model_save_path)

    if training_args.do_eval:
        # Evaluate the model
        logger.info("Evaluating model ...")
        eval_metrics = trainer.evaluate()
        print(eval_metrics)
        logger.info("Evaluation finished.")
        # Save the evaluation metrics
        trainer.save_metrics(Split.EVAL.value, eval_metrics)
        # Log the evaluation metrics
        trainer.log_metrics(Split.EVAL.value, flatten_dict(eval_metrics))

    if training_args.do_predict:

        if test_dataset:
            test_dataset = prepare_dataset(
                test_dataset,
                tokenizer=tokenizer,
                padding=config.padding,
                truncation=config.truncation,
                max_length=config.max_length,
                input_columns=config.text_name,
                desc="Preparing test dataset for predictions",
                batched=True,
            )

        else:
            test_dataset = eval_dataset

        if not test_dataset:
            logger.error(
                "No test/eval dataset is provided while `args.do_predict` is set to `True`."
                "Please provide a test/eval dataset to make predictions."
                "Exiting ..."
            )
            quit()

        # Make predictions
        logger.info("Making predictions ...")
        test_results = trainer.predict(test_dataset)
        logger.info("Predictions finished.")
        trainer.save_metrics(Split.TEST.value, test_results.metrics)
        trainer.log_metrics(Split.TEST.value, flatten_dict(test_results.metrics))

        predictions = pd.DataFrame(
            {
                "prediction": class_label.int2str(test_results.predictions.tolist()),
                "label": class_label.int2str(test_results.label_ids.tolist()),
            }
        )

        # Save predictions
        logger.info(f"Saving predictions to {training_args.output_dir} ...")
        predictions.to_csv(
            os.path.join(training_args.output_dir, f"{Split.TEST}_predictions.csv"),
            index=False,
        )

    logger.info("Finished.")


if __name__ == "__main__":
    main()
