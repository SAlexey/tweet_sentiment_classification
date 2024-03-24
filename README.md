# Tweet Sentiment Classification

Tweet sentiment classification using transformers

## Project Description

This project aims to classify the sentiment of tweets using transformers. The project uses the Hugging Face Transformers library to train a transformer model. The model is trained using the `transformers` library and the `Trainer` class. The model is then saved and deployed using Docker.
Use the `Makefile` to run the project.

## Makefile Commands

**IMPORTANT**:

- Before running the commands, make sure to set the PYTHON variabe in the Makefile to the path of the Python executable on your system. The project was developed and tested using Python 3.11.8.
- In oder to run `make up` and `make down` commands, you need to have Docker installed on your system.
- In order to run the `make up` command, you need to have the model saved in the `models` directory. You can train the model using the `make train` command. Or
  you can run the `notebooks/02_model_training.ipynb` notebook to mock train the model.

Run the following commands in order to get started with the project:

- `make environment`: Create a Python virtual environment
- `make install`: Install the required packages
- `make test`: Run the tests
- `make data`: Prepare the data
- `make train[-dev]`: Train the model (the `train-dev` command is used to train the model on a smaller dataset for development purposes)
- `make up`: Compose and run the docker containers that deploy the model behind an HTTP server (localhost:1234)
- `make down`: Stop and remove the docker containers

Run `make help` to see all available commands.

## Project Organization

    ├── LICENSE
    ├── Makefile                    <- Makefile with commands like `make data` or `make train`
    ├── README.md                   <- The top-level README for developers using this project.
    ├── compose.yaml                <- Docker Compose File
    ├── data
    │   ├── processed               <- The final, canonical data sets for modeling.
    │   └── raw                     <- The original, immutable data dump.
    │
    ├── docker                      <- Docker Images
    │   ├── Dockerfile.serve        <- Server Image (Deploying model behind an HTTP Server)
    │ 
    │
    ├── models                      <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                   <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                  the creator's initials, and a short `-` delimited description, e.g.
    │                                  `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt            <- The requirements file for reproducing the analysis environment, e.g.
    │                                  generated with `pip freeze > requirements.txt`
    │
    ├── setup.py                    <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                         <- Source code for use in this project.
    │   ├── __init__.py             <- Makes src a Python module
    │   │
    │   ├── data                    <- Scripts to download or generate data
    │   │   └── preprocess.py
    │   |
    │   ├── model                   <- Scripts to train models and then use trained models to make
    │       │                          predictions
    │       ├── predict.py
    │       └── train.py
    │
    ├── tests                       <- Pytest scripts.
    |   |–– test_data_filters.py
    |   |–– test_data_transforms.py
    |   |–– test_data_utils.py
    |
    └── pytest.ini                  <- Pytest configuration

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
