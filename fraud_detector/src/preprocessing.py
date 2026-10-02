"""Feature building for the logistic regression model.

Stateless: raw rows in test.csv format -> model features. Everything that has to be fitted
(one-hot categories, scaling, imputation) lives in the sklearn pipeline built in trainer.py,
so the saved model file is all that is needed for scoring. The CatBoost version of this
step is in preprocessing_catboost.py.
"""
import numpy as np
import pandas as pd

INPUT_COLS = ['transaction_time', 'merch', 'cat_id', 'amount', 'name_1', 'name_2', 'gender', 'street', 'one_city',
              'us_state', 'post_code', 'lat', 'lon', 'population_city', 'jobs', 'merchant_lat', 'merchant_lon']

CATEGORICAL_COLS = ['gender', 'merch', 'cat_id', 'one_city', 'us_state', 'jobs']
NUMERIC_COLS = ['amount_log', 'population_log', 'distance_km',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _cyclic(out, name, values, period):
    angle = 2 * np.pi * values / period
    out[name + '_sin'] = np.sin(angle)
    out[name + '_cos'] = np.cos(angle)


def make_features(df):
    """Raw rows (test.csv columns; extras are ignored, missing ones raise KeyError) -> feature frame."""
    df = df[INPUT_COLS]
    out = pd.DataFrame(index=range(len(df)))

    # Categories stay raw: the pipeline one-hot encodes them (unseen / rare values are grouped)
    for col in CATEGORICAL_COLS:
        out[col] = df[col].reset_index(drop=True).fillna('missing').astype(str)

    num = lambda col: pd.to_numeric(df[col], errors='coerce').reset_index(drop=True)

    out['amount_log'] = np.log1p(num('amount'))
    out['population_log'] = np.log1p(num('population_city'))
    out['distance_km'] = haversine_km(num('lat'), num('lon'), num('merchant_lat'), num('merchant_lon'))

    t = pd.to_datetime(df['transaction_time'].reset_index(drop=True), errors='coerce').dt
    _cyclic(out, 'hour', t.hour + t.minute / 60, 24)
    _cyclic(out, 'dow', t.dayofweek, 7)
    _cyclic(out, 'month', t.month - 1, 12)

    return out
