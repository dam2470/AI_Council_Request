from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Create the API application
# Find the project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Locate the saved model and vectoriser
MODEL_PATH = BASE_DIR / "model" / "stopword_model.joblib"
VECTORISER_PATH = BASE_DIR / "model" / "stopword_vectoriser.joblib"

# Store the loaded model and vectoriser
artefacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load both files when the API starts
    artefacts["model"] = joblib.load(MODEL_PATH)
    artefacts["vectoriser"] = joblib.load(VECTORISER_PATH)

    yield

    # Clear the stored objects when the API stops
    artefacts.clear()

# Create the API application with startup and shutdown handling
app = FastAPI(lifespan=lifespan)

# Define the expected input for a prediction
class PredictRequest(BaseModel):
    request_text: str = Field(min_length=1, max_length=2000)


# Check that the API is running
@app.get("/health")
def health():
    return {"status": "ok"}


# Return the categories recognised by the model
@app.get("/categories")
def get_categories():
    categories = artefacts["model"].classes_.tolist()
    return {"categories": categories}


# Receive and validate a service request
@app.post("/predict")
def predict(request: PredictRequest):
    text = request.request_text.strip()

    if text == "":
        raise HTTPException(
            status_code=422,
            detail="request_text must not be blank."
        )

        # Match the text cleaning used during training
    cleaned_text = text.lower()
    cleaned_text = " ".join(cleaned_text.split())

    model = artefacts["model"]
    vectoriser = artefacts["vectoriser"]

    # Convert the cleaned text into numerical features
    features = vectoriser.transform([cleaned_text])

    # Calculate the category probabilities
    probabilities = model.predict_proba(features)[0]
    best_index = probabilities.argmax()

    predicted_category = str(model.classes_[best_index])
    confidence = round(float(probabilities[best_index]), 3)

    return {
        "request_text": request.request_text,
        "predicted_category": predicted_category,
        "confidence": confidence
    }