from pathlib import Path

import cv2
import pandas as pd
import numpy as np

from app.model.parsing import Downloader

class hairColorDetector():
    

    def __init__(self, img: np.ndarray) -> None:
        """
        Args:
            img (np.ndarray): Photo
        """
        # Base dir
        baseDir = Path(__file__).resolve(strict=True).parent
        
        # Data
        index = ["color_name", "R", "G", "B", "type"]
        self.colorNamesDf = pd.read_csv(f'{baseDir}/data/hairColors.csv', names=index, header=None)
        
        # Img
        self.img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Classifier
        self.classifierPath = f'{baseDir}/facesClassifier.xml'


    def findFace(self) -> tuple:
        """Function find face in img and return rectangle coordinates 
        of the face box or None if the face is not found

        Returns:
            tuple: A tuple containing the coordinates of the face box in the format (startingPoint_x, startingPoint_y, width, height)
        """
        
        # Find face
        grayImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(self.classifierPath)
        coords = faces.detectMultiScale(grayImg, scaleFactor=1.2, minNeighbors=4)
        
        
        # Check
        if len(coords) != 1:
            return ()
        
        # Get box coordinates
        for (x, y, w, h) in coords:
            rect_coords = (
                int(x - 0.25*w),
                int(y - 0.25*h),
                int(1.75 * w),
                int(1.75 * h)
            )
            
        return rect_coords


    def getFaceAndHair(self, rectangle: tuple) -> np.ndarray:
        """The function returns an image with hair and face, the remaining pixels are black

        Args:
            rectangle (tuple): Coordinates for background selection

        Returns:
            np.ndarray: Image with face and hair
        """
        
        # GrabCut foreground
        
        # Create mask for hairs
        hairMask = self.img.copy()
        
        # Set params for extraction
        black_mask = np.zeros(self.img.shape[:2], np.uint8)
        background = np.zeros((1, 65), np.float64)
        foreground = np.zeros((1, 65), np.float64)

        # Apply the grabcut algorithm
        cv2.grabCut(hairMask, black_mask, rectangle, background, foreground, 5, cv2.GC_INIT_WITH_RECT)
        
        # In the new mask image, pixels will 
        # be marked with four flags 
        # four flags denote the background / foreground 
        # mask is changed, all the 0 and 2 pixels 
        # are converted to the background
        # mask is changed, all the 1 and 3 pixels
        # are now the part of the foreground
        # the return type is also mentioned,
        # this gives us the final mask
        mask2 = np.where((black_mask == 2) | (black_mask == 0), 0, 1).astype('uint8')
        
        # Final mask
        hairMask = hairMask * mask2[:, :, np.newaxis]

        # Convert color
        hairMask = cv2.cvtColor(hairMask, cv2.COLOR_BGR2GRAY)

        # Final img
        img = cv2.bitwise_and(self.img, self.img, mask=hairMask)
        
        # Resize to minimize pixels count
        img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
        
        return img


    def get_color_name(self, BGRlist: list) -> str:
        """Function return color name with the help of 
        list of a colors numbers(BGR)

        Args:
            BGRlist (list): List of numbers denoting color in BGR format

        Returns:
            str: Color name
        """
        # The difference between a pixel in the image 
        # and a pixel of the corresponding color
        distance = abs(self.colorNamesDf['R'] - BGRlist[-1])\
            +abs(self.colorNamesDf['G'] - BGRlist[1])\
                +abs(self.colorNamesDf['B'] - BGRlist[0])
        
        # Choose a color for which the difference
        # between rgb values is minimal
        colorIdx = distance.idxmin()
        
        return self.colorNamesDf.loc[colorIdx, 'color_name']


    def getColorRatio(self, img: np.ndarray) -> dict:
        """The function considers the color pixels in the image 
        and returns the ratio of these pixels by hair color type(5 types)

        Args:
            img (np.ndarray): Prepared image

        Returns:
            dict: Color ratio
        """

        # Array with lists of rgb values of each pixel
        colorsArray = []
        for pixelsRow in iter(img):
            for pixel in iter(pixelsRow):
                colorsArray.append(list(pixel))
                
        # Create a dataframe and remove all black pixels (which were cut off by the mask)
        colorsDf = pd.DataFrame({'bgr': colorsArray})
        
        # Removing black pixels from DF
        colorsDf['notBlack'] = colorsDf['bgr'].apply(lambda x: True if sum(x) > 0 else False)
        colorsDf = colorsDf[colorsDf['notBlack'] == True]
        colorsDf = colorsDf.drop('notBlack', axis=1)

        # Get a color for each pixel
        colorsDf['ColorName'] = colorsDf['bgr'].apply(self.get_color_name)
        
        return colorsDf['ColorName'].value_counts(normalize=True).to_dict()


if __name__ == '__main__':
    url = 'https://images-ssl.gotinder.com/u/5tLAiGvmHR5xuHMwetRpAg/aqis4VLtmP5fc9nHaG59i9.jpg?Policy=eyJTdGF0ZW1lbnQiOiBbeyJSZXNvdXJjZSI6IiovdS81dExBaUd2bUhSNXh1SE13ZXRScEFnLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE2ODk3NDUyMzR9fX1dfQ__&Signature=zYMXx4M8aHa0oNP8-YONM9vNhR1Vrfd-imWzI2t4ykUKxKQiuzmxC7o2teUiKeHxUcfk44N~-B5~6ZCmdlBHkD0ASNnUrG9PSMdDMIjEEjt7hbrPb-FUs~GxF3zKfYdyLI091tWVJ7xGAuZ80TAecjrmYHLqf0CL-Qpj2G55B52K5EpXDEu2i-mLiW9q~0wJdoIZpBRGBnZbmUVMlzkd2FyJ9l7-xpxkby0VC2hK8mA7cMFifnklMWcJEzGa8vgRzEpwngGpVOlRkwQBFsqt4YoeqFpbhndwkNuXLRE5X4gi3Xt53la~XCGsFdMIUkqk6C2dbwrB~LF19H8kNtEzdg__&Key-Pair-Id=K368TLDEUPA6OI'
    
    img = Downloader(url=url).getPhoto()
    if img is None:
        print('kek')
    else:
        detector = hairColorDetector(img)
        rectangle = detector.findFace()
        prepImg = detector.getFaceAndHair(rectangle)
    
        cv2.imshow('image',prepImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    #result = detector.getColorRatio(prepImg)
    #print(result)











