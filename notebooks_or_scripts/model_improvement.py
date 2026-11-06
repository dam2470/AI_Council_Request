import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

PROCESSED_PATH = "data/processed/BCIT601_starter_service_requests_dataset_clean.csv"
MODEL_PATH = "model/stopword_model.joblib"
VECTORISER_PATH = "model/stopword_vectoriser.joblib"


if __name__ == "__main__":
    # Load the cleaned dataset
    df = pd.read_csv(PROCESSED_PATH)

    print("Requests before removing repeated text:", len(df))

    # Remove repeated text so it cannot appear in both training and testing
    df = df.drop_duplicates(subset=["request_text"])

    print("Requests after removing repeated text:", len(df))

    # Separate the input text from the correct category labels
    x = df["request_text"]
    y = df["category"]

    # Use 80% for training and 20% for testing
    x_train, x_test, y_train, y_test = train_test_split(
        x, # Requests model reads
        y, # Correct labels for categories
        test_size=0.2, # Sets 20% of the data for testing
        random_state=42, # Makes split repeatable when data and settings are the same
        stratify=y # Keeps same proportion across both groups
    )
    print("Training requests:", len(x_train))
    print("Testing requests:", len(x_test))


    # Create single word TF-IDF features
    baseline_vectoriser = TfidfVectorizer()
    x_train_baseline = baseline_vectoriser.fit_transform(x_train)
    x_test_baseline = baseline_vectoriser.transform(x_test)

    # Train the reference model using the original classifier settings
    baseline_model = LogisticRegression(max_iter=1000)
    baseline_model.fit(x_train_baseline, y_train)

    print("Single-word baseline trained.")


    # Evaluate the single-word baseline
    baseline_predictions = baseline_model.predict(x_test_baseline)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)
    baseline_report = classification_report(y_test, baseline_predictions)

    print("\nBaseline accuracy:", round(baseline_accuracy * 100, 2), "%")
    print("\nBaseline classification report:\n", baseline_report)


    # Test removing the common words
    stopword_vectoriser = TfidfVectorizer(
        stop_words=["the", "was", "is", "a", "an"]
    )
    x_train_stopwords = stopword_vectoriser.fit_transform(x_train)
    x_test_stopwords = stopword_vectoriser.transform(x_test)

    # Keep the classifier settings the same for a fair comparison
    stopword_model = LogisticRegression(max_iter=1000)
    stopword_model.fit(x_train_stopwords, y_train)

    # Evaluate using the same test requests
    stopword_predictions = stopword_model.predict(x_test_stopwords)
    stopword_accuracy = accuracy_score(y_test, stopword_predictions)

    print("\nStop-word accuracy:", round(stopword_accuracy * 100, 2), "%")
    print(classification_report(y_test, stopword_predictions))


    # Saves the selected model and its matching fitted vectoriser
    joblib.dump(stopword_model, MODEL_PATH)
    joblib.dump(stopword_vectoriser, VECTORISER_PATH)

    print("\nImproved model saved to:", MODEL_PATH)
    print("Improved vectoriser saved to:", VECTORISER_PATH)
