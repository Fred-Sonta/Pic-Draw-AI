# main_app.py

import cv2
import numpy as np
import mediapipe as mp
import math
import os 

# Importer nos modules personnalisés
import train_model
import drawing_utils
import setup_tools

# --- Initialisation ---
print("Initialisation du modèle et des classes...")
model, class_names = setup_tools.initialize_model_and_classes()

if model is None or class_names is None:
    print("Échec de l'initialisation. L'application va se fermer.")
    exit()

# Initialisation de MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Capture vidéo
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erreur: Impossible d'ouvrir la webcam.")
    exit()

# Création du canevas pour le dessin
ret, frame = cap.read()
if not ret:
    print("Erreur: Impossible de lire une image de la webcam.")
    exit()
h, w, _ = frame.shape
canvas = np.zeros((h, w, 3), dtype=np.uint8)

# Variables pour le dessin
prev_point = None
PINCH_THRESHOLD = 0.05 # Seuil pour détecter le pincement

# Variables pour l'affichage du résultat
prediction_text = ""
predicted_image = None
display_result = False

print("\n--- Commandes ---")
print("'P' : Lancer la prédiction du dessin")
print("'C' : Effacer le canevas")
print("'Q' : Quitter l'application")
print("Pincez le pouce et l'index pour dessiner.")

# --- Boucle principale de l'application ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inverser l'image horizontalement (effet miroir)
    frame = cv2.flip(frame, 1)
    
    # Convertir l'image en RGB pour MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Logique de dessin (avec pincement)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Coordonnées de l'index (landmark 8)
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        ix, iy = int(index_tip.x * w), int(index_tip.y * h)
        
        # Coordonnées du pouce (landmark 4)
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        
        # Calculer la distance
        distance = math.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)

        # Vérifier le geste de pincement
        if distance < PINCH_THRESHOLD:
            if prev_point is None: 
                prev_point = (ix, iy)
            
            # Réduite à 5 pour une meilleure précision
            cv2.line(canvas, prev_point, (ix, iy), (255, 255, 255), 5)
            prev_point = (ix, iy)
        else:
            prev_point = None # Réinitialiser quand on arrête de pincer
    else:
        # Si aucune main n'est détectée, on arrête de dessiner
        prev_point = None

    # Combinaison du frame de la webcam et du canevas
    combined_display = cv2.add(frame, canvas)
    
    # Afficher le texte de la prédiction
    if display_result:
        cv2.putText(combined_display, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('EmoDraw - Dessinez avec votre doigt', combined_display)

    # Afficher l'image prédite
    if display_result and predicted_image is not None:
        cv2.imshow('Image Predite', predicted_image)

    # Gestion des touches du clavier
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'): # Quitter
        break
    elif key == ord('c'): # Effacer (Clear)
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        display_result = False
        prediction_text = ""
        # Fermer les fenêtres supplémentaires
        if cv2.getWindowProperty('Image Predite', cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow('Image Predite')
        if cv2.getWindowProperty('Debug - Ce que voit IA (28x28)', cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow('Debug - Ce que voit IA (28x28)')
            
    # --- BLOC DE PRÉDICTION ENTIÈREMENT CORRIGÉ ---
    elif key == ord('p'): # Prédire
        
        # 1. Appeler la fonction de prétraitement
        processed_data = drawing_utils.preprocess_drawing_for_model(canvas)
        
        if processed_data is not None:
            # 2. Déballer les DEUX valeurs retournées
            processed_img_for_model, debug_image_28x28 = processed_data
            
            # 3. Afficher la fenêtre de debug (redimensionnée pour être visible)
            debug_visible = cv2.resize(debug_image_28x28, (400, 400), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('Debug - Ce que voit IA (28x28)', debug_visible)

            # 4. Donner la BONNE image (formatée) au modèle
            predictions = model.predict(processed_img_for_model)
            
            # 5. Traiter les résultats
            top_prediction_index = np.argmax(predictions[0])
            confidence = predictions[0][top_prediction_index]
            predicted_class_name = class_names[top_prediction_index]
            
            prediction_text = f"Prediction: {predicted_class_name} ({confidence:.2%})"
            print(prediction_text)
            display_result = True
            
            # 6. Charger l'image correspondante
            img_path = os.path.join("images_pred", f"{predicted_class_name}.png")
            if os.path.exists(img_path):
                predicted_image = cv2.imread(img_path)
            else:
                predicted_image = np.zeros((200, 400, 3), dtype=np.uint8)
                cv2.putText(predicted_image, "Image non trouvee", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                print(f"Avertissement : L'image {img_path} n'a pas été trouvée.")

        else:
            prediction_text = "Canevas vide. Impossible de predire."
            print(prediction_text)
            display_result = True
            predicted_image = None


# Libérer les ressources
cap.release()
hands.close()
cv2.destroyAllWindows()
