"""Scoring with the active logistic regression model.

The active model (pipeline + threshold) is kept in memory in one tuple. The Kafka loop reads it
once per batch, and the HTTP API replaces it with a single assignment, so a model switch
never lands in the middle of a batch. The model history is in Postgres (registry.py).
The CatBoost version of this module is in scorer_catboost.py.
"""
import logging
import os
import threading
import uuid
from typing import Any, NamedTuple

import joblib
import pandas as pd

import registry
import trainer

logger = logging.getLogger(__name__)

BASELINE_PATH = os.getenv('BASELINE_MODEL_PATH', './models/baseline.joblib')
TRAINED_DIR = os.getenv('TRAINED_MODELS_DIR', './models/trained')

# Decision threshold of the baseline model (env FRAUD_THRESHOLD); changeable later from the UI
DEFAULT_THRESHOLD = float(os.getenv('FRAUD_THRESHOLD', '0.5'))


class ActiveModel(NamedTuple):
    id: int
    pipeline: Any
    threshold: float


class TrainingBusy(Exception):
    pass


_active = None
_train_lock = threading.Lock()


def _load(row):
    bundle = joblib.load(row['path'])
    return ActiveModel(row['id'], bundle['pipeline'], row['threshold'])


def init():
    """Create the table, register the committed baseline on the very first start, load the active model."""
    global _active
    registry.init_db()
    if registry.count() == 0:
        bundle = joblib.load(BASELINE_PATH)
        model_id = registry.add_model('baseline', bundle['params'], bundle['metrics'],
                                      DEFAULT_THRESHOLD, BASELINE_PATH, is_default=True)
        registry.activate(model_id)
        logger.info('Registered baseline model #%s', model_id)

    row = registry.get_active()
    try:
        _active = _load(row)
    except Exception as e:  # e.g. the volume with trained models was wiped
        logger.error('Cannot load active model #%s (%s), falling back to the default one', row['id'], e)
        row = registry.activate(registry.get_default()['id'])
        _active = _load(row)
    logger.info('Active model: #%s, threshold %s', _active.id, _active.threshold)


def active_id():
    return _active.id


def make_pred(raw_df, source_info="kafka"):
    """raw rows in test.csv format -> DataFrame(score, fraud_flag)"""
    model = _active  # one read: the model can't change under us mid-batch
    proba = model.pipeline.predict_proba(raw_df)[:, 1]
    submission = pd.DataFrame({
        'score': proba,
        'fraud_flag': (proba > model.threshold) * 1
    })
    logger.info(f'Prediction complete for data from {source_info} (model #{model.id})')
    return submission


def retrain(params):
    """Train on train.csv with the given hyperparameters, register the model and make it active."""
    global _active
    if not _train_lock.acquire(blocking=False):
        raise TrainingBusy('Training is already running')
    try:
        bundle = trainer.train(params)
        os.makedirs(TRAINED_DIR, exist_ok=True)
        path = os.path.join(TRAINED_DIR, f'model_{uuid.uuid4().hex[:8]}.joblib')
        joblib.dump(bundle, path, compress=3)
        model_id = registry.add_model('retrained', bundle['params'], bundle['metrics'],
                                      _active.threshold, path)
        row = registry.activate(model_id)
        _active = _load(row)
        return row
    finally:
        _train_lock.release()


def activate(model_id):
    """Switch to a model from the history. Returns its row or None if there is no such model."""
    global _active
    row = registry.get_model(model_id)
    if row is None:
        return None
    new = _load(row)  # load first: if the file is gone, nothing changes
    row = registry.activate(model_id)
    _active = new
    return row


def reset_to_default():
    return activate(registry.get_default()['id'])


def set_threshold(value):
    global _active
    registry.set_threshold(_active.id, value)
    _active = _active._replace(threshold=value)
