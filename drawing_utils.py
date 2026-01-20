# drawing_utils.py

import cv2
import numpy as np

IMG_SIZE = 28 # Taille d'image attendue par le modèle
PADDING = 5   # Marge ajoutée autour du dessin

def preprocess_drawing_for_model(canvas):
    """
    Isole, centre, redimensionne, et formate le dessin pour le modèle.
    """
    
    # 1. Convertir en niveaux de gris et trouver le dessin
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    
    # Si le canevas est vide, retourner None
    if coords is None:
        return None

    # 2. Trouver la "bounding box" (boîte englobante) du dessin
    x, y, w, h = cv2.boundingRect(coords)

    # 3. Recadrer (Crop) l'image sur le dessin (en gardant le blanc sur noir)
    cropped_image = gray[y:y+h, x:x+w]

    # 4. Créer un canevas carré pour éviter la déformation
    size = max(w, h) + PADDING * 2

    # 5. Créer un canevas carré NOIR
    square_canvas = np.zeros((size, size), dtype=np.uint8)

    # 6. Centrer le dessin (blanc sur noir) dans le canevas carré noir
    start_x = (size - w) // 2
    start_y = (size - h) // 2
    square_canvas[start_y:start_y+h, start_x:start_x+w] = cropped_image

    # 7. Redimensionner à la taille attendue par le modèle (28x28)
    resized_image = cv2.resize(square_canvas, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # 8. INVERSER L'IMAGE 28x28 FINALE pour passer en "noir sur blanc"
    final_image = cv2.bitwise_not(resized_image)

    # 9. Normaliser les valeurs (entre 0 et 1) et reformater pour Keras
    final_image_normalized = final_image.astype('float32') / 255.0
    final_image_reshaped = np.reshape(final_image_normalized, (1, IMG_SIZE, IMG_SIZE, 1))

    # Retourner l'image pour l'IA, ET l'image 28x28 brute pour la sauvegarde
    return final_image_reshaped, final_image

