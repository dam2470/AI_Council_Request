import pandas as pd

RAW_PATH = "data/raw/BCIT601_starter_service_requests_dataset.csv"
PROCESSED_PATH = "data/processed/CIT601_starter_service_requests_dataset_clean.csv"

def load_data(RAW_PATH):
    return pd.read_csv(RAW_PATH)