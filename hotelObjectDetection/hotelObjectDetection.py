from imageai.Prediction.Custom import CustomImagePrediction
from imageai.Detection import ObjectDetection
import os
import json


class hotelObject:
    def ObjectDetect(self,input_imagePath):
        execution_path = r"C:\Users\nabaj\source\repos\final_module_iDIL_django\image_detection_algorithms\images"
        output_path = r"C:\Users\nabaj\source\repos\final_module_iDIL_django\image_detection_algorithms\images\outputImages"
        print(execution_path)
        detector = ObjectDetection()
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath(r"C:\Users\nabaj\source\repos\final_module_iDIL_django\image_detection_algorithms\resnet50_coco_best_v2.0.1.h5")
        detector.loadModel()
        detections1 = detector.detectCustomObjectsFromImage(input_image=input_imagePath,
                                                            output_image_path=os.path.join(output_path,
                                                                                           "example3.jpg"))

        """detector2 = ObjectDetection()
        detector2.setModelTypeAsYOLOv3()
        detector2.setModelPath(os.path.join(execution_path, "yolo.h5"))
        detector2.loadModel()
        detections2 = detector2.detectCustomObjectsFromImage(input_image=os.path.join(execution_path, "trail1.jpg"),
                                                             output_image_path=os.path.join(execution_path,
                                                                                            "example4.jpg"))"""




        List = []
        for i in detections1:
            List.append(i["name"])

        return List

