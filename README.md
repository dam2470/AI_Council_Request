# Council Service Request Classifier

**Kai Morgan | BCIT601 – Assignment 2 | Iteration 2**

## Overview

I am building a machine learning system to classify council service requests as **road, waste, water or safety**. So far, I have cleaned the provided dataset, trained and evaluated a baseline model, saved the results and started a Flask API.

## Tools Used

| Tool | Purpose |
| --- | --- |
| Python | Programming language for the project. |
| pandas | Loading, inspecting, cleaning and saving CSV data. |
| scikit-learn | Splitting data, TF-IDF text features, Logistic Regression and evaluation. |
| joblib | Saving and loading the trained model and fitted vectoriser. |
| Flask | Creating API routes and handling JSON requests and responses. |
| pathlib | Managing file paths and creating folders; included with Python. |

I use Visual Studio Code to edit and run the code, and Git/GitHub for version control.

## Main Files

| File or folder | Purpose |
| --- | --- |
| `data/raw/` and `data/processed/` | Original and cleaned datasets. |
| `data_cleaning.py` | Checks the data and standardises text formatting. |
| `baseline_model.py` | Trains, evaluates and saves the model and vectoriser. |
| `model/` | Generated `model.joblib` and `vectoriser.joblib` files. |
| `outputs/` | Evaluation report, five example predictions and incorrect predictions. |
| `api/app.py` | Flask API code. |

The Word report contains my progress summary and ethical reflection.

## Running the Project

Open a terminal in the main project folder and use the project's Python environment. Install the external packages:

```bash
python -m pip install pandas scikit-learn joblib Flask
```

Run the cleaning script, then the baseline script. If they are directly inside `notebooks_or_scripts/`, use:

```bash
python notebooks_or_scripts/data_cleaning.py
python notebooks_or_scripts/baseline_model.py
```

Adjust these script paths if needed, keeping the terminal in the main project folder. Training creates the two saved files in `model/`.

Start the API:

```bash
python -m flask --app api.app run
```

Leave the terminal running and open the [health check](http://127.0.0.1:5000/health) or [category list](http://127.0.0.1:5000/categories).

## Baseline Results

The dataset contains **80 records**, with 20 per category. Cleaning applies lowercase and whitespace normalisation; no rows were removed.

I used a stratified **64/16 training/test split**, `random_state=42`, TF-IDF and Logistic Regression. The saved evaluation reports **87.5% accuracy — 14 of 16 predictions correct**.

One waste request and one road request were incorrectly predicted as safety. The dataset has only 60 distinct request texts, and eight test texts also appear in training. This limits conclusions about performance on new requests.

## API Status

| Endpoint | Method | Current implementation |
| --- | --- | --- |
| `/health` | GET | Returns `{"status": "ok"}`. |
| `/categories` | GET | Returns the model's category labels. |
| `/predict` | POST | Validates input; prediction is unfinished. |

The API loads the saved model and vectoriser at startup. Currently, `/predict` has no response for valid input. My next step is to add preprocessing, prediction and confidence, then test valid and invalid requests.
