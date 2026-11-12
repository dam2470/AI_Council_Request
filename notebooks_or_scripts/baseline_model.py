"""Historical Iteration 2 baseline.

This script uses the dataset before repeated request texts are removed.
The final model comparison is in model_improvement.py.
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

PROCESSED_PATH = "data/processed/BCIT601_starter_service_requests_dataset_clean.csv"
PREDICTIONS_PATH = "outputs/baseline_example_predictions.csv"
MODEL_PATH = "model/model.joblib"
VECTORISER_PATH = "model/vectoriser.joblib"
REPORT_PATH = "outputs/baseline_evaluation.txt"
ERRORS_PATH = "outputs/baseline_incorrect_predictions.csv"

if __name__ == "__main__":
    #Load the clean dataset
    df = pd.read_csv(PROCESSED_PATH)
    # Separates the request text from the correct categories
    x = df["request_text"]
    y = df["category"]

    # Split the data  into training and testing sets
    x_train, x_test, y_train,  y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42, 
        stratify=y
    )

    print("Training requests:"  , len(x_train))
    print("Testing requests:", len(x_test))

    print("\nTraining categories:", y_train.value_counts(), sep="\n")
    print("\nTesting categories:", y_test.value_counts(), sep="\n")

    # converts the requests text into numerics
    vectoriser = TfidfVectorizer()
    x_train_tfidf = vectoriser.fit_transform(x_train)
    x_test_tfidf = vectoriser.transform(x_test)

    print("\nTraining feature shape:", x_train_tfidf.shape)
    print("\nTesting feature shape:", x_test_tfidf.shape)

    # Create and train  the baseline classifier
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train_tfidf, y_train)

    print("\nBaseline model trained.")
    print("categories:", model.classes_)

    # Predict categories for test requests
    y_pred = model.predict(x_test_tfidf)

    # Compare the predictions with the correct categories
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\nTest accuracy:", round(accuracy * 100, 2), "%")
    print("\nClassification report:", report, sep="\n")

    # This saves the accuracy and classification report
    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)

    report_text  = f"Test accuracy: {accuracy * 100:.2f}%\n\n{report}"
    Path(REPORT_PATH).write_text(report_text, encoding="utf-8")

    print("\nEvaluation report saved to:", REPORT_PATH)

    # Combines the text requests , correct categories and predictions
    results = pd.DataFrame({
        "request_text": x_test,
        "true_category": y_test,
        "predicted_category": y_pred
    })

    # Selects the first 5 test results
    examples =  results.head(5)

    # Creates the output folder and it save the examples
    Path(PREDICTIONS_PATH).parent.mkdir(parents=True, exist_ok=True)
    examples.to_csv(PREDICTIONS_PATH, index=False)

    print("\nFive example predictions:")

    # I did this to make the example table easier to read
    for _, row in examples.iterrows():
        print("\nRequest:", row["request_text"])
        print("Correct category:", row["true_category"])
        print("Predicted category:", row["predicted_category"])
    
    print("\nExample predictions saved to:", PREDICTIONS_PATH)

    # This finds test requests that were predicted incorrectly
    mistakes = results[
        results["true_category"] !=  results["predicted_category"]
    ]

    # Saves the incorrect predictions so they can be reviewed
    mistakes.to_csv(ERRORS_PATH, index=False)

    print("\nIncorrect predictions:", len(mistakes))

    for _, row in mistakes.iterrows():
        print("\nRequest:", row["request_text"])
        print("Correct category:", row["true_category"])
        print("Predicted category:", row["predicted_category"])

    print("\nIncorrect predictions saved to:", ERRORS_PATH)  
  
    # This saves the trained model and the fitted vectoriser
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectoriser, VECTORISER_PATH)

    print("\nModel saved to:", MODEL_PATH)
    print("Vectoriser saved to:", VECTORISER_PATH)

