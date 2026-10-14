from flask import Flask
import joblib
from pathlib import Path
from flask import Flask, request

# API  application created
app = Flask(__name__)

#find prject folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Locaterhe saaved model and vectoriser
MODEL_PATH = BASE_DIR / 'model' / 'model.joblib'
VECTORISER_PATH = BASE_DIR / 'model' / 'vectoriser.joblib'

# Loads model and vectoriser when api starts
model = joblib.load(MODEL_PATH)
vectoriser = joblib.load(VECTORISER_PATH)

# returns the categorys recognised by the trained model
@app.route('/categories', methods=['GET'])
def categories():
    return {'categories': model.classes_.tolist()}

# recieve and validatre service request
@ app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)

    # check if input is json object
    if not isinstance(data, dict):
        return {'error': 'Send valid JSON object.'}, 400

    text = data.get('request_text')

    # reject missing, or non string or blank text
    if not isinstance(text, str) or not text.strip():
        return {'error': 'request_text must not be an empty string.'}, 400

# Check if the API is running
@app.route('/health', methods=['GET'])
def health():
    return {'status': 'ok'}