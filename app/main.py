from fastapi import FastAPI
from pydantic import BaseModel

from app.model.model import modelPrediction
from app.model.model import __version__ as model_version


app = FastAPI()

        
class PhotosIn(BaseModel):
    photos: list[dict]
 

class PredictionOut(BaseModel):
    predictions: list[str] = None


@app.get('/')
def home() -> dict:
    """Fucnction return status
    and model version
    
    Returns:
        dict: Status and model version
    """
    return {'health_check': 'ok', 'model_version': model_version} 


@app.post('/predict', response_model=PredictionOut)
def predict(payload: PhotosIn) -> dict:
    """Function return predict by data contains
    urls of woman photos

    Args:
        payload (PhotosIn): List with dicts contains urls

    Returns:
        dict: predictions for each photo
    """
    predictions = []
    
    # Get predictions list
    for photoIdx in range(len(payload.photos)):
        url = payload.photos[photoIdx]['photo'][1]['url']
    
        predictions.append(modelPrediction(url=url))
    
    return {'predictions': predictions}


