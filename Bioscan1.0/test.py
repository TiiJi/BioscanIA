import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Chemin vers le modèle sauvegardé
model_path = 'plant_recognition_model.keras'

# Charger le modèle
model = tf.keras.models.load_model(model_path)

# Classes de plantes (assurez-vous que l'ordre des classes correspond à l'ordre utilisé lors de l'entraînement)
class_names = [
    'aloevera', 
    'banane', 
    'bilimbi',
    'cantaloupe',
    'cassava',
    'coconut',
    'corn',
    'cucumber',
    'curcuma',
    'eggplant',
    'galangal',
    'ginger',
    'guava',
    'kale',
    'longbeans',
    'mango',
    'melon',
    'orange',
    'paddy',
    'papaya',
    'peperchili',
    'pineapple',
    'pomelo',
    'shalot',
    'soybeans',
    'spinach',
    'sweetpotatoes',
    'tabaco',
    'waterapple',
    'watermelon'
]

def prepare_image(img_path, img_height=180, img_width=180):
    """
    Prépare une image pour la prédiction par le modèle.

    Args:
        img_path (str): Chemin vers l'image.
        img_height (int): Hauteur de l'image attendue par le modèle.
        img_width (int): Largeur de l'image attendue par le modèle.

    Returns:
        np.array: Image prétraitée.
    """
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalisation
    return img_array

def predict_plant(img_path):
    """
    Prédit le nom de la plante dans une image.

    Args:
        img_path (str): Chemin vers l'image.

    Returns:
        str: Nom de la plante prédite.
    """
    img_array = prepare_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return class_names[predicted_class[0]]

# Chemin vers l'image à prédire
img_path = 'C:\\Big_Projects\\test6.png'

# Prédire le nom de la plante
predicted_plant = predict_plant(img_path)
print(f'La plante dans l\'image est: {predicted_plant}')
