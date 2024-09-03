from typing import Any, List, Optional
import datetime

from pydantic import BaseModel
from heartdisease_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                     "age": 60,
                     "sex": 1,
                     "cp": 1,
                     "trestbps": 145,
                     "chol": 233,
                     "fbs": 1,
                     "restecg": 2,
                     "thalach": 150,
                     "exang": 0,
                     "oldpeak": 2.3,
                     "slope": 3,
                     "ca": 0,
                     "thal": 2
                    }
                ]
            }
        }
