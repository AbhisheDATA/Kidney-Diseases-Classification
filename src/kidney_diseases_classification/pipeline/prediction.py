import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import cv2



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts","training", "model.h5"))

        imagename = self.filename
        
        test_image = cv2.imread(str(imagename))
        test_image = cv2.resize(test_image, (224, 224))
        test_image = test_image / 255.0
        test_image = np.array(test_image)
        
        test_image = np.expand_dims(test_image, axis=0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 0:
            prediction = 'Cyst'
            return [{ "image" : prediction}]
        elif result[0] ==1:
            prediction = 'Normal'
            return [{ "image" : prediction}]
        elif result[0] ==2:
            prediction = 'Stone'
            return [{ "image" : prediction}]
        else:
            prediction = 'Tumor'
            return [{ "image" : prediction}]