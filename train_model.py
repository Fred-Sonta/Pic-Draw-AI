# train_model.py

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --- Paramètres ---
DATA_DIR = "dataset"
IMG_WIDTH, IMG_HEIGHT = 28, 28
BATCH_SIZE = 32
EPOCHS = 20 # 20 passages devraient être largement suffisants

# 1. Préparation des données (avec Data Augmentation)
# C'est la magie : on crée plus de données à partir de nos 20-30 images
datagen = ImageDataGenerator(
    rescale=1./255,             # Normalise (déjà fait, mais bonne pratique)
    validation_split=0.2,       # Garde 20% des images pour tester
    rotation_range=10,          # Tourne les images un peu
    width_shift_range=0.1,      # Décale un peu horizontalement
    height_shift_range=0.1,     # Décale un peu verticalement
    zoom_range=0.1              # Zoome un peu
)

# Générateur pour l'entraînement
train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',     # TRÈS IMPORTANT : nos images sont en N&B
    class_mode='categorical',
    subset='training'           # Indique que c'est le set d'entraînement
)

# Générateur pour la validation (test)
validation_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',     # TRÈS IMPORTANT
    class_mode='categorical',
    subset='validation'         # Indique que c'est le set de validation
)

# 2. Définition du modèle (CNN simple mais efficace)
num_classes = len(train_generator.class_indices)

model = Sequential([
    # Couche 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
    MaxPooling2D((2, 2)),
    
    # Couche 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Aplatir et connecter
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Prévient le sur-apprentissage
    Dense(num_classes, activation='softmax') # Couche de sortie
])

model.summary()

# 3. Compilation et Entraînement
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\n--- Début de l'entraînement ---")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# 4. Sauvegarde du modèle et des classes
print("Entraînement terminé. Sauvegarde du modèle...")
model.save("my_demo_model.h5")

# Sauvegarde de la liste des classes
class_names = list(train_generator.class_indices.keys())
with open("my_demo_classes.txt", "w") as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")

print(f"Modèle sauvegardé sous 'my_demo_model.h5'")
print(f"Classes sauvegardées sous 'my_demo_classes.txt'")
print("Vous êtes prêt pour la démonstration !")

