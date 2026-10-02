"""HTTP API used by the UI to retrain / switch the model and change the threshold.
Interactive docs: http://localhost:8000/docs"""
import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import registry
import scorer

logger = logging.getLogger(__name__)

app = FastAPI(title='fraud_detector API')


class TrainRequest(BaseModel):
    C: float = Field(gt=0, le=10000)
    class_weight: Literal['balanced', 'none']
    max_iter: int = Field(ge=10, le=5000)


class ThresholdRequest(BaseModel):
    value: float = Field(ge=0, le=1)


@app.get('/models')
def list_models():
    return {'active_id': scorer.active_id(), 'models': registry.list_models()}


# A plain `def` endpoint runs in a worker thread, so a long fit does not stop the Kafka loop
@app.post('/train')
def train(req: TrainRequest):
    try:
        return scorer.retrain(req.model_dump())
    except scorer.TrainingBusy as e:
        raise HTTPException(409, str(e))
    except FileNotFoundError as e:
        raise HTTPException(400, f'Train data not found: {e.filename}. See data/README.md')
    except Exception as e:
        logger.exception('Training failed')
        raise HTTPException(500, f'Training failed: {e}')


@app.post('/models/{model_id}/activate')
def activate(model_id: int):
    row = scorer.activate(model_id)
    if row is None:
        raise HTTPException(404, f'No model #{model_id}')
    return row


@app.post('/models/reset-default')
def reset_default():
    return scorer.reset_to_default()


@app.post('/threshold')
def set_threshold(req: ThresholdRequest):
    scorer.set_threshold(req.value)
    return {'active_id': scorer.active_id(), 'threshold': req.value}
