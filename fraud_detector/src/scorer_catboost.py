"""CatBoost scoring (kept for reference, not used by the service; needs `catboost` installed)."""
import os
import pandas as pd
import logging
from catboost import CatBoostClassifier

# Настройка логгера
logger = logging.getLogger(__name__)

logger.info('Importing pretrained model...')

# Import model
model = CatBoostClassifier()
model.load_model('./models/my_catboost.cbm')

# Decision threshold (env FRAUD_THRESHOLD). Competition-tuned value was 0.98; the default 0.05 is
# the mean score on test.csv, chosen so that demo runs produce a visible share of fraud flags
model_th = float(os.getenv('FRAUD_THRESHOLD', '0.05'))
logger.info('Pretrained model imported successfully. Fraud threshold: %s', model_th)


def make_pred(dt, source_info="kafka"):

    # Меняем формат категориальных фичей на string перед скорингом
    expected_categorical = ['hour',
                            'year',
                            'month',
                            'day_of_month',
                            'day_of_week',
                            'gender_cat',
                            'merch_cat',
                            'cat_id_cat',
                            'one_city_cat',
                            'us_state_cat',
                            'jobs_cat']
    for col in expected_categorical:
        if col in dt.columns:
            dt[col] = dt[col].astype(str)

    # Calculate score (probability of the positive class)
    proba = model.predict_proba(dt)[:, 1]
    submission = pd.DataFrame({
        'score': proba,
        'fraud_flag': (proba > model_th) * 1
    })
    logger.info(f'Prediction complete for data from {source_info}')

    return submission