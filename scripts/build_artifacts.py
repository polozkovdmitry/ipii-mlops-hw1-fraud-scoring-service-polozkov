"""Offline step: compute preprocessing statistics from train.csv and save them
as a small artifact file used by the scoring service (no train.csv needed at runtime).

Usage:
    python scripts/build_artifacts.py path/to/train.csv
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from geopy.distance import great_circle

TARGET_COL = 'target'
CATEGORICAL_COLS = ['gender', 'merch', 'cat_id', 'one_city', 'us_state', 'jobs']
TIME_COLS = ['hour', 'year', 'month', 'day_of_month', 'day_of_week']
CONTINUOUS_COLS = ['amount', 'population_city', 'distance']
DROP_COLS = ['name_1', 'name_2', 'street', 'post_code']
N_CATS = 50

OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'fraud_detector', 'artifacts', 'preproc_artifacts.pkl')


def main(train_path):
    train = pd.read_csv(train_path).drop(columns=DROP_COLS)

    t = pd.to_datetime(train['transaction_time']).dt
    train['hour'], train['year'], train['month'] = t.hour, t.year, t.month
    train['day_of_month'], train['day_of_week'] = t.day, t.dayofweek
    train = train.drop(columns='transaction_time')

    # Frequency-rank encoding: cat_0 .. cat_49 for the most frequent values, cat_50+ for the rest
    cat_maps = {}
    for col in CATEGORICAL_COLS:
        # Same operations as the original pipeline (incl. default sort) so that ties rank identically
        counts = (
            train.groupby(col, dropna=False)[[TARGET_COL]].count()
            .sort_values(TARGET_COL, ascending=False)
            .reset_index()
        )
        cat_maps[col] = {
            value: (f'cat_{rank}' if rank < N_CATS else f'cat_{N_CATS}+')
            for rank, value in enumerate(counts[col])
            if not pd.isna(value)
        }
        train[col + '_cat'] = train[col].map(cat_maps[col]).fillna('cat_NAN')

    # Target mean encoding for all categorical + time columns
    mean_enc = {}
    for col in [c + '_cat' for c in CATEGORICAL_COLS] + TIME_COLS:
        mean_enc[col] = train.groupby(col)[TARGET_COL].mean().to_dict()

    # Distance between client and merchant (needed for the imputer mean)
    train['distance'] = [
        great_circle((a, b), (c, d)).km
        for a, b, c, d in zip(train['lat'], train['lon'], train['merchant_lat'], train['merchant_lon'])
    ]
    impute_means = {col: float(train[col].mean()) for col in CONTINUOUS_COLS}

    artifacts = {'cat_maps': cat_maps, 'mean_enc': mean_enc, 'impute_means': impute_means}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'wb') as f:
        pickle.dump(artifacts, f)
    print('Saved', os.path.abspath(OUT_PATH), os.path.getsize(OUT_PATH) // 1024, 'KB')


if __name__ == '__main__':
    main(sys.argv[1])
