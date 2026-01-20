# setup_tools.py

import os
import tensorflow as tf

# On définit les noms de nos fichiers locaux
MODEL_FILENAME = "my_demo_model.h5"
CLASSES_FILENAME = "my_demo_classes.txt"

def initialize_model_and_classes():
    """
    Charge le modèle et les classes depuis les fichiers locaux.
    """
    
    # Étape 1: Charger le modèle Keras
    if not os.path.exists(MODEL_FILENAME):
        print(f"ERREUR: Le fichier modèle '{MODEL_FILENAME}' est introuvable.")
        print("Veuillez d'abord lancer 'python train_model.py' pour le créer.")
        return None, None
        
    try:
        model = tf.keras.models.load_model(MODEL_FILENAME, compile=False)
        print("Modèle local 'my_demo_model.h5' chargé avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle {MODEL_FILENAME}: {e}")
        return None, None

    # Étape 2: Charger la liste des classes
    if not os.path.exists(CLASSES_FILENAME):
        print(f"ERREUR: Le fichier des classes '{CLASSES_FILENAME}' est introuvable.")
        print("Veuillez d'abord lancer 'python train_model.py' pour le créer.")
        return None, None

    try:
        with open(CLASSES_FILENAME, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"Liste des {len(class_names)} classes locales chargée avec succès.")
        
        # Vérification de correspondance
        model_output_shape = model.output_shape[1]
        if model_output_shape != len(class_names):
            print(f"ERREUR CRITIQUE : Incompatibilité entre le modèle ({model_output_shape}) et les classes ({len(class_names)}).")
            return None, None
            
        print("Le modèle et la liste des classes correspondent parfaitement.")

    except Exception as e:
        print(f"Erreur lors de la lecture du fichier {CLASSES_FILENAME}: {e}")
        return model, None
        
    return model, class_names

if __name__ == '__main__':
    initialize_model_and_classes()


    