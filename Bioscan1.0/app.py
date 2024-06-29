from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import io

app = Flask(__name__)

# Charger le modèle
model = tf.keras.models.load_model('plant_recognition_model.keras')

# Classes de plantes
class_names = [
    'aloevera', 'banane', 'bilimbi', 'cantaloupe', 'cassava', 'coconut',
    'corn', 'cucumber', 'curcuma', 'eggplant', 'galangal', 'ginger',
    'guava', 'kale', 'longbeans', 'mango', 'melon', 'orange', 'paddy',
    'papaya', 'peperchili', 'pineapple', 'pomelo', 'shalot', 'soybeans',
    'spinach', 'sweetpotatoes', 'tabaco', 'waterapple', 'watermelon'
]

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = tf.keras.preprocessing.image.load_img(io.BytesIO(file.read()), target_size=(180, 180))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)  # Create batch axis
    img_array /= 255.0  # Normalisation

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]

    print(f'tonga tato')
    
    return jsonify({'prediction': predicted_class})

@app.route('/', methods=['GET'])
def index():
    return "Flask server is running!"

if __name__ == '__main__':
    app.run(debug=True)
