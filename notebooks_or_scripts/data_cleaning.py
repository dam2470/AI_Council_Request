import pandas as pd
from pathlib import Path

RAW_PATH = "data/raw/BCIT601_starter_service_requests_dataset.csv"
PROCESSED_PATH = "data/processed/BCIT601_starter_service_requests_dataset_clean.csv"

# load the data into the pandas DataFrame
def load_data(path):
    return pd.read_csv(path)

# Check missing values, duplicate records and inconsistant text formatting
def inspect_data(df):
    print("\nRows and Columns:", df.shape)

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isna().sum())

    print("\nExact duplicate rows:",  df.duplicated().sum())


    print("\nRequests per category:")
    print(df["category"].value_counts(dropna=False))

    print("\nFirst five request texts:")
    for text in df["request_text"].head():
        print(text)

#  convert to lower and remove white spaces

def clean_text(text): 
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()
    text = " ".join(text.split())

    return text

# Apply cleaning steps  to the dataset copy

def  clean_data(df):
    df = df.copy()

    df["request_text"] = df["request_text"].apply(clean_text)
    df["category"] = df["category"].apply(clean_text)

    before = len(df)
    df = df[df["request_text"] != ""]
    print("Rows removed for missing or blank request text:", before - len(df))

    before = len(df)
    df = df [df["category"] != ""]
    print("Rows removed for missing or blank categorys:", before - len(df))

    before = len(df)
    df = df.drop_duplicates()
    print("Exact duplicate rows removed:", before - len(df))

    return df

if __name__ == "__main__":
    raw_df = load_data(RAW_PATH)

    print("BEFORE CLEANING")
    inspect_data(raw_df)

    print("\nCLEANING")
    clean_df = clean_data(raw_df)

    print("\nAFTER CLEANING")
    inspect_data(clean_df)

    Path(PROCESSED_PATH).parent.mkdir(parents=True,  exist_ok=True)

    clean_df.to_csv(PROCESSED_PATH, index=False)

    print("\nCleaned dataset saved to:", PROCESSED_PATH)
