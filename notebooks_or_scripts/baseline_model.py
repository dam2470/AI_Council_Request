import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_PATH = "data/processed/BCIT601_starter_service_requests_dataset_clean.csv"

if __name__ == "__main__":
    #Load the clean dataset
    df = pd.read_csv(PROCESSED_PATH)
# Seperate the request text from the correct categorys
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

print("\nTraining categorys:", y_train.value_counts(), sep="\n")
print("\nTesting categorys:", y_test.value_counts(), sep="\n")