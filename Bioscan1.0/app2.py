import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os

# Paramètres de base
batch_size = 32
img_height = 180
img_width = 180

# Chemins vers les fichiers CSV
train_csv = 'C:\\Big_Projects\\archive\\train.csv'
val_csv = 'C:\\Big_Projects\\archive\\val.csv'
test_csv = 'C:\\Big_Projects\\archive\\test.csv'

# Répertoires contenant les images
base_dir = 'C:\\Big_Projects\\archive'

# Fonction pour charger les données à partir des CSVs
def load_data(csv_file, base_dir):
    """
    Charge les chemins de fichiers et les labels à partir d'un fichier CSV.

    Args:
        csv_file (str): Chemin vers le fichier CSV.
        base_dir (str): Répertoire de base contenant les images.

    Returns:
        (pd.Series, pd.Series): Séries contenant les chemins de fichiers et les labels.
    """
    df = pd.read_csv(csv_file)
    filepaths = df['image:FILE'].apply(lambda x: os.path.join(base_dir, x))
    labels = df['category']
    return filepaths, labels

# Charger les données
train_filepaths, train_labels = load_data(train_csv, base_dir)
val_filepaths, val_labels = load_data(val_csv, base_dir)
test_filepaths, test_labels = load_data(test_csv, base_dir)

# Création des DataGenerators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Création des générateurs de données à partir des DataFrames
train_generator = train_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': train_filepaths, 'class': train_labels}),
    x_col='filename',
    y_col='class',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='raw'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': val_filepaths, 'class': val_labels}),
    x_col='filename',
    y_col='class',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='raw'
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': test_filepaths, 'class': test_labels}),
    x_col='filename',
    y_col='class',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='raw',
    shuffle=False
)

# Définir le modèle CNN
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(train_labels.unique()), activation='softmax')
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle
epochs = 10
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# Évaluer le modèle
loss, accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {accuracy}')

# Sauvegarder le modèle au format recommandé par Keras
model.save('plant_recognition_model.tflite')
