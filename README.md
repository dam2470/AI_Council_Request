# Council Service Request Classifier

**Kai Morgan | BCIT601 - Assignment 2 | Final Submission**

## Overview

I am building a machine learning system to classify council service requests as **road, waste, water or safety**. In Iteration 3, I compared the baseline with a version that removes selected stop words and replaced Flask with FastAPI. The API now returns a predicted category and confidence score and validates incoming requests.

Project repository: [AI_Council_Request](https://github.com/dam2470/AI_Council_Request).

## Tools Used

| Tool | Purpose |
| --- | --- |
| Python 3.13 | Programming language used during development. |
| pandas | Loading, inspecting and cleaning the dataset. |
| scikit-learn | TF-IDF features, Logistic Regression, training/test splitting and evaluation. |
| joblib | Saving and loading the model and fitted vectoriser. |
| FastAPI | API endpoints and interactive documentation. |
| Pydantic | Validating request text. |
| Uvicorn | Running the API locally. |
| pathlib and contextlib | File paths and API startup/shutdown; included with Python. |

## Main Files

| File or folder | Purpose |
| --- | --- |
| `data/raw/` and `data/processed/` | Original and cleaned starter datasets. |
| `notebooks_or_scripts/data_cleaning.py` | Checks the data, handles blank values and standardises text formatting. |
| `notebooks_or_scripts/baseline_model.py` | Original Iteration 2 training and evaluation script. |
| `notebooks_or_scripts/model_improvement.py` | Removes repeated request texts, compares two models and saves the selected model pair. |
| `model/model.joblib` and `model/vectoriser.joblib` | Original Iteration 2 model pair. |
| `model/stopword_model.joblib` and `model/stopword_vectoriser.joblib` | Selected model pair used by the API. |
| `outputs/baseline_*` | Historical baseline report, example predictions and incorrect predictions. |
| `outputs/model_improvement_evaluation.txt` | Saved Iteration 3 comparison results. |
| `api/app.py` | FastAPI application and prediction code. |
| `requirements.txt` | External Python dependencies and their recorded versions. |

## Running the Project

Open PowerShell in the project root, where this README and `requirements.txt` are stored. Run all commands below from this folder.

### 1. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

The versions in `requirements.txt` match my development environment. The saved model and vectoriser were created using scikit-learn 1.9.0.

### 2. Recreate the saved model (optional)

The saved stop-word model pair is already included. To recreate it and the evaluation output, run:

```powershell
python notebooks_or_scripts/data_cleaning.py
python notebooks_or_scripts/model_improvement.py | Tee-Object -FilePath outputs/model_improvement_evaluation.txt
```

The improvement script prints both evaluation reports and saves the selected model and vectoriser in `model/`. `Tee-Object` also saves the printed results in `outputs/model_improvement_evaluation.txt`.

To reproduce the historical Iteration 2 baseline separately, run:

```powershell
python notebooks_or_scripts/baseline_model.py
```

This writes the original model pair and the `baseline_*` outputs.

### 3. Start the API

```powershell
python -m uvicorn api.app:app --reload
```

Leave the terminal running and open:

- [Interactive API documentation](http://127.0.0.1:8000/docs)
- [Health check](http://127.0.0.1:8000/health)
- [Category list](http://127.0.0.1:8000/categories)

The API loads `stopword_model.joblib` and `stopword_vectoriser.joblib` once at startup using `joblib.load`. It does not retrain inside the API. Press **Ctrl+C** to stop the server.

## Data and Model Results

The provided synthetic dataset contains **80 records**, with 20 per category. Cleaning applies lowercase and whitespace normalisation, removes blank text or labels and checks exact duplicate rows. The current processed CSV still contains 80 records.

The improvement script removes duplicate `request_text` values before splitting, leaving **60 unique requests**, with 15 per category. This happens in memory and does not overwrite the processed CSV. It prevents identical request texts from appearing in both training and testing.

Both models use the same stratified **48/12 training/test split**, `random_state=42` and `LogisticRegression(max_iter=1000)`. Each TF-IDF vectoriser is fitted on training text only.

The improvement tested was removing the fixed stop-word list **the, was, is, a and an**.

| Model version | Accuracy | Correct predictions | Macro F1 |
| --- | ---: | ---: | ---: |
| Single-word TF-IDF baseline | 75.00% | 9/12 | 0.73 |
| Single-word TF-IDF with stop-word removal | 83.33% | 10/12 | 0.83 |

Stop-word removal improved accuracy by **8.33 percentage points** on this split, equivalent to one additional correct prediction. Road recall increased from 1/3 to 2/3. The full classification reports are saved in `outputs/model_improvement_evaluation.txt`.

The original Iteration 2 baseline scored 87.5% on a different 64/16 split. Eight test texts also appeared in training. That result is retained as historical evidence and is not directly comparable with the Iteration 3 scores.

## API Endpoints

| Endpoint | Method | Response |
| --- | --- | --- |
| `/health` | GET | `{"status": "ok"}` |
| `/categories` | GET | `{"categories": ["road", "safety", "waste", "water"]}` |
| `/predict` | POST | Original request text, predicted category and confidence score. |

In `/docs`, expand **POST /predict** and select **Try it out**. Swagger fills in the example below. Select **Execute** to run it, or edit `request_text` to try another council request.

```json
{
  "request_text": "Large pothole"
}
```

Recorded response:

```json
{
  "request_text": "Large pothole",
  "predicted_category": "road",
  "confidence": 0.35
}
```

Confidence is the highest category probability from `predict_proba()`, rounded to three decimal places. It is not a guarantee of correctness.

The request must contain a string between 1 and 2,000 characters. Spaces-only text is rejected. Valid text is converted to lowercase and its whitespace is normalised before vectorisation.

## Testing and Evidence

I tested the endpoints manually through the interactive documentation. `/health` and `/categories` returned HTTP 200. The following prediction requests also returned HTTP 200:

| Request text | Predicted category | Confidence |
| --- | --- | ---: |
| Large pothole | road | 0.350 |
| Water leaking from a pipe | water | 0.504 |
| Broken playground equipment | safety | 0.385 |
| Rubbish dumped beside bins | waste | 0.445 |
| Broken glass on the sidewalk | safety | 0.296 |

Invalid requests were also tested:

| Input | Recorded response |
| --- | --- |
| `{"request_text": ""}` | HTTP 422: text must contain at least one character. |
| `{"request_text": "   "}` | HTTP 422: `request_text must not be blank.` |
| `{}` | HTTP 422: required field missing. |
| `{"request_text": 123}` | HTTP 422: input must be a string. |

These tests check API behaviour. The model's 83.33% accuracy comes from the separate 12-request evaluation.

Screenshots are stored in the **API Testing Evidence** folder supplied separately with the assessment, outside the code repository. They include prediction tests, endpoint tests, invalid-input tests and the model comparison.

The final Swagger example and its successful response are shown in `API Testing Evidence/Final Iteration/Swagger Example Prediction.png`.
