from flask import Flask, jsonify, request
from model.predict import load_model
import hydra

CONFIG_PATH = "model/config"
CONFIG_NAME = "predict"

with hydra.initialize(config_path=CONFIG_PATH, version_base="1.3"):
    cfg = hydra.compose(config_name=CONFIG_NAME)

app = Flask(__name__)
model = load_model(cfg.model_dir)


@app.route("/", methods=["POST"])
def classify_text():
    data = request.get_json()
    result = model(data)
    return jsonify(result)


if __name__ == "__main__":
    app.run()
