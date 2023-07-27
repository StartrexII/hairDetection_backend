import pickle
from pathlib import Path

import pandas as pd
import cv2

from app.model.hairColorDetection import hairColorDetector
from app.model.parsing import Downloader

# Uploading model with current version

__version__ = '0.1.0'

baseDir = Path(__file__).resolve(strict=True).parent

with open(f'{baseDir}/trained_model-{__version__}.pkl', '+rb') as f:
    model = pickle.load(f)
    
# Dict to interpret. classes
colorClasses = {
    0: 'light brown',
    1: 'blond',
    2: 'brown-haired',
    3: 'red hair',
    4: 'black'
}

# Colors we get from script
colorNames = ['brown-haired', 'light brown', 'blond', 'red hair', 'black']

def modelPrediction(url: str) -> str:
    """Function predict hair color by url of photo with woman

    Args:
        url (str): Link to the photo

    Returns:
        str: Hair color
    """
    # Get image
    img = Downloader(url=url).getPhoto()
    
    # Check acess to photo
    if img is None:
        return 'acess_denied'
    
    detector = hairColorDetector(img)
    rectangle = detector.findFace()
    
    # Check face
    if rectangle == ():
        return 'None'
    
    try:    
        result = detector.getColorRatio(detector.getFaceAndHair(rectangle))
        result = pd.DataFrame(result, columns=colorNames)
        result.fillna(0, inplace=True)
    except cv2.error:
        return 'None'
    else:
        return colorClasses[int(model.predict(result))]
