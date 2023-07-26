import httplib2
import cv2
import numpy as np

class Downloader():
    """The class provides photo upload and saves it in accordance with the transmitted name
    or return how numpy array
    """
    
    
    def __init__(self, url: str) -> None:
        self.url = url
    
    def photoDownload(self, name: str, path: str) -> None:
        """Function save an image in folder with transmitted name

        Args:
            name (str): File name
            path (str): Path to image
        """
        h = httplib2.Http('.cache')
        response, content = h.request(self.url)
        out = open(path + str(name) + '.jpg', 'wb')
        out.write(content)
        out.close()
        
        
    def getPhoto(self) -> np.ndarray:
        """Function return cv2 image by url

        Returns:
            np.ndarray: Image
        """
        
        h = httplib2.Http('.cache')
        response, content = h.request(self.url)
        
        # Image building from request content
        img = np.asarray(bytearray(content), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
        return img

if __name__ == '__main__':
    img = Downloader('https://images-ssl.gotinder.com/u/k1W3ubArqUCHDTGttXTpRL/5Wypd6ddGXYfaBbpDGb1D9.jpg?Policy=eyJTdGF0ZW1lbnQiOiBbeyJSZXNvdXJjZSI6IiovdS9rMVczdWJBcnFVQ0hEVEd0dFhUcFJMLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE2ODk3OTQ1NTF9fX1dfQ__&Signature=wbfunY8GSBZWdhqpflbbTwhXJAiw1T8xSt2VpN4sS5TSHm4H8U01ycPyNRlL2w83AXcL9bRe8HXMAeRM1attCDq-6z6NUBOD000CoepYAkB55VBf7Nw4~Fkd4yamHYiwke76gbMrXwB6zI27A7r3-OW1tTCAh4vCtKm6fAj15VcsgPmm~6~1iPxA2EyDtsJuz0Oxy81te0ozn~kPUDddUSsc4jpVDdgf74LpH-~vp5eQk43kE7ORO1DPylFhuyxIBGLeIMg1ziZ1DlxnKFKGrcuoyYOzjqdZgjlaz0t43yVq1cmrFGTlrcODY9e2Q1jaE7uG7b81d58ik6RTyg2b8A__&Key-Pair-Id=K368TLDEUPA6OI')\
        .getPhoto()        

    print(img)
    #cv2.imshow('image', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()