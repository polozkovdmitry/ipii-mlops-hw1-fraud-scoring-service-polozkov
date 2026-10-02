"""Training of the logistic regression model.

As a library: train(params) -> bundle used by the service (scorer.retrain).
As a script (builds the committed baseline):
    python src/trainer.py --data /path/to/train.csv --out models/baseline.joblib
"""
import argparse
import logging
import os
import sys
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import CATEGORICAL_COLS, INPUT_COLS, NUMERIC_COLS, make_features  # noqa: E402

logger = logging.getLogger(__name__)

# Full train.csv if it is there, otherwise the small sample committed to the repo (misc/train_sample.csv.gz)
TRAIN_DATA_PATH = os.getenv('TRAIN_DATA_PATH', '/data/train.csv')
TRAIN_SAMPLE_PATH = os.getenv('TRAIN_SAMPLE_PATH', '/misc/train_sample.csv.gz')
TARGET_COL = 'target'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Parameters of the baseline model and the initial values of the UI form
DEFAULT_PARAMS = {'C': 1.0, 'class_weight': 'balanced', 'max_iter': 1000}


def build_pipeline(params):
    features = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=20), CATEGORICAL_COLS),
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), NUMERIC_COLS),
    ])
    clf = LogisticRegression(
        C=float(params['C']),
        class_weight=None if params['class_weight'] in (None, 'none') else params['class_weight'],
        max_iter=int(params['max_iter']),
    )
    return Pipeline([('make_features', FunctionTransformer(make_features)), ('features', features), ('clf', clf)])


def train(params, data_path=None):
    """Fit on a train split of the data, score on the holdout. Returns {'pipeline', 'params', 'metrics'}."""
    data_path = data_path or (TRAIN_DATA_PATH if os.path.exists(TRAIN_DATA_PATH) else TRAIN_SAMPLE_PATH)
    started = time.time()
    df = pd.read_csv(data_path, usecols=INPUT_COLS + [TARGET_COL])
    X, y = df[INPUT_COLS], df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    pipeline = build_pipeline(params)
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best = int(f1.argmax())
    metrics = {
        'roc_auc': float(roc_auc_score(y_test, proba)),
        'pr_auc': float(average_precision_score(y_test, proba)),
        'best_f1': float(f1[best]),
        'best_f1_threshold': float(thresholds[best]),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'fraud_rate': float(y.mean()),
        'data_source': os.path.basename(data_path),
        'fit_seconds': round(time.time() - started, 1),
    }
    logger.info('Trained logreg %s: %s', params, metrics)
    return {'pipeline': pipeline, 'params': dict(params), 'metrics': metrics}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=TRAIN_DATA_PATH)
    parser.add_argument('--out', default='models/baseline.joblib')
    args = parser.parse_args()
    joblib.dump(train(DEFAULT_PARAMS, args.data), args.out, compress=3)
    print('Saved', args.out, os.path.getsize(args.out) // 1024, 'KB')
