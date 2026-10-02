"""Preprocessing for the CatBoost model (kept for reference, not used by the service)."""
import logging
import os
import pickle

import numpy as np
import pandas as pd
from geopy.distance import great_circle

logger = logging.getLogger(__name__)

ARTIFACTS_PATH = os.getenv('PREPROC_ARTIFACTS_PATH', './artifacts/preproc_artifacts.pkl')

CATEGORICAL_COLS = ['gender', 'merch', 'cat_id', 'one_city', 'us_state', 'jobs']
TIME_COLS = ['hour', 'year', 'month', 'day_of_month', 'day_of_week']
CONTINUOUS_COLS = ['amount', 'population_city', 'distance']
DROP_COLS = ['name_1', 'name_2', 'street', 'post_code']
INPUT_COLS = ['transaction_time', 'merch', 'cat_id', 'amount', 'name_1', 'name_2', 'gender', 'street', 'one_city',
              'us_state', 'post_code', 'lat', 'lon', 'population_city', 'jobs', 'merchant_lat', 'merchant_lon']


def load_artifacts(path=ARTIFACTS_PATH):
    """Load precomputed encoders (built offline by scripts/build_artifacts.py)."""
    logger.info('Loading preprocessing artifacts from %s', path)
    with open(path, 'rb') as f:
        return pickle.load(f)


def add_time_features(df):
    t = pd.to_datetime(df['transaction_time']).dt
    df['hour'] = t.hour
    df['year'] = t.year
    df['month'] = t.month
    df['day_of_month'] = t.day
    df['day_of_week'] = t.dayofweek
    return df.drop(columns='transaction_time')


def add_distance(df):
    df['distance'] = [
        great_circle((a, b), (c, d)).km
        for a, b, c, d in zip(df['lat'], df['lon'], df['merchant_lat'], df['merchant_lon'])
    ]
    return df.drop(columns=['lat', 'lon', 'merchant_lat', 'merchant_lon'])


def run_preproc(artifacts, input_df):
    """Pure function: (artifacts, raw rows in test.csv format) -> model features."""
    # Keep only the expected test.csv columns (extras are ignored, missing ones raise KeyError)
    df = input_df[INPUT_COLS].drop(columns=DROP_COLS).reset_index(drop=True)

    # Frequency-rank category encoding (unseen / missing values -> cat_NAN)
    for col in CATEGORICAL_COLS:
        df[col + '_cat'] = df[col].map(artifacts['cat_maps'][col]).fillna('cat_NAN')
        df = df.drop(columns=col)

    df = add_time_features(df)

    # Target mean encoding
    for col in [c + '_cat' for c in CATEGORICAL_COLS] + TIME_COLS:
        df[col + '_mean_enc'] = df[col].map(artifacts['mean_enc'][col])

    df = add_distance(df)

    # Mean imputation + log transform
    for col in CONTINUOUS_COLS:
        df[col + '_log'] = np.log(df[col].fillna(artifacts['impute_means'][col]) + 1)
        df = df.drop(columns=col)

    return df
