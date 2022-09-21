import collections
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
from keras.models import load_model
# from keras.engine.saving import load_model
import matplotlib.pyplot as plt

class foodDetection:

    def detectFood(self,image_path):
        food_list = ["apple_pie",
                     "baby_back_ribs",
                     "baklava",
                     "beef_carpaccio",
                     "beef_tartare",
                     "beet_salad",
                     "beignets",
                     "bibimbap",
                     "bread_pudding",
                     "breakfast_burrito",
                     "bruschetta",
                     "caesar_salad",
                     "cannoli",
                     "caprese_salad",
                     "carrot_cake",
                     "ceviche",
                     "cheesecake",
                     "cheese_plate",
                     "chicken_curry",
                     "chicken_quesadilla",
                     "chicken_wings",
                     "chocolate_cake",
                     "chocolate_mousse",
                     "churros",
                     "clam_chowder",
                     "club_sandwich",
                     "crab_cakes",
                     "creme_brulee",
                     "croque_madame",
                     "cup_cakes",
                     "deviled_eggs",
                     "donuts",
                     "dumplings",
                     "edamame",
                     "eggs_benedict",
                     "escargots",
                     "falafel",
                     "filet_mignon",
                     "fish_and_chips",
                     "foie_gras",
                     "french_fries",
                     "french_onion_soup",
                     "french_toast",
                     "fried_calamari",
                     "fried_rice",
                     "frozen_yogurt",
                     "garlic_bread",
                     "gnocchi",
                     "greek_salad",
                     "grilled_cheese_sandwich",
                     "grilled_salmon",
                     "guacamole",
                     "gyoza",
                     "hamburger",
                     "hot_and_sour_soup",
                     "hot_dog",
                     "huevos_rancheros",
                     "hummus",
                     "ice_cream",
                     "lasagna",
                     "lobster_bisque",
                     "lobster_roll_sandwich",
                     "macaroni_and_cheese",
                     "macarons",
                     "miso_soup",
                     "mussels",
                     "nachos",
                     "omelette",
                     "onion_rings",
                     "oysters",
                     "pad_thai",
                     "paella",
                     "pancakes",
                     "panna_cotta",
                     "peking_duck",
                     "pho",
                     "pizza",
                     "pork_chop",
                     "poutine",
                     "prime_rib",
                     "pulled_pork_sandwich",
                     "ramen",
                     "ravioli",
                     "red_velvet_cake",
                     "risotto",
                     "samosa",
                     "sashimi",
                     "scallops",
                     "seaweed_salad",
                     "shrimp_and_grits",
                     "spaghetti_bolognese",
                     "spaghetti_carbonara",
                     "spring_rolls",
                     "steak",
                     "strawberry_shortcake",
                     "sushi",
                     "tacos",
                     "takoyaki",
                     "tiramisu",
                     "tuna_tartare",
                     "waffles"]

        model = load_model(r'C:\Users\nabaj\source\repos\final_module_iDIL_django\image_detection_algorithms\InceptionNet.h5', compile=False)

        def predict_class(model, images, show=True):
            result = []
            for img in images:
                img = load_img(img, target_size=(299, 299))
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img /= 255.

                pred = model.predict(img)
                index = np.argmax(pred)
                food_list.sort()
                pred_value = food_list[index]
                result.append(pred_value)
                """if show:
                    plt.imshow(img[0])
                    plt.axis('off')
                    plt.title(pred_value)
                    print(pred_value)"""
            return result

        images = []
        images.append(image_path)
        return predict_class(model, images, True)

