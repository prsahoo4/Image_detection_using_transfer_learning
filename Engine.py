from .foodDetection.foodCategories.predict import predictFoodCategory
from .foodDetection.foodDetect import foodDetection
from .hotelObjectDetection.hotelObjectDetection import hotelObject
from .landmarkDetection.landmarkDetection import landmark
from .foodDetection.foodCategories.keras_model import build_model
import os

class Engine:
    def ImagePath(self,imageName):
        image_path = r"C:\Users\nabaj\source\repos\final_module_iDIL_django\image_detection_algorithms\images"
        full_pathImage = os.path.join(image_path,imageName)
        return full_pathImage

    def StartEngine(self,full_pathImage):
        foodArray = foodDetection().detectFood(full_pathImage)
        landmarkArray = landmark().DefineLandmark(full_pathImage)
        objectArray = hotelObject().ObjectDetect(full_pathImage)
        bagOfObjects = []
        for food in foodArray:
            bagOfObjects.append(food)
        for land in landmarkArray:
            bagOfObjects.append(land)
        for object in objectArray:
            bagOfObjects.append(object)
        print(bagOfObjects)
        return bagOfObjects

# obj = Engine().StartEngine(Engine().ImagePath("conf_room1.jpeg"))

