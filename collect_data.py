# collect_data.py

import cv2
import numpy as np
import mediapipe as mp 
import math
import os
import drawing_utils # On importe notre prétraitement

# --- Configuration ---
DATA_DIR = "dataset"
CLASSES_LIST = ["cercle", "carre", "etoile", "maison", "triangle", "coeur", "bonhomme", "eclair", "arbre", "voiture"]
# (Modifiez cette liste si vous avez choisi d'autres noms)

# --- AJOUT (Recommandation 1 & 2) ---
LINE_THICKNESS = 7  # Épaisseur du trait plus visible
DEAD_ZONE_RADIUS = 3 # Rayon de la zone morte pour ignorer les tremblements

# --- Initialisation OpenCV/MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if not ret:
    print("Erreur webcam")
    exit()
h, w, _ = frame.shape
canvas = np.zeros((h, w, 3), dtype=np.uint8)

prev_point = None
PINCH_THRESHOLD = 0.05

# --- Logique de Collecte ---
master_collecting_flag = True 

for class_name in CLASSES_LIST:
    if not master_collecting_flag: 
        break

    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    
    count = len(os.listdir(class_path))
    
    print(f"\n--- Prêt à dessiner pour la classe : '{class_name}' ---")
    print("Commandes :")
    print("Pincez pour dessiner.")
    print("'C' : Effacer le dessin actuel.")
    print("'S' : Sauvegarder le dessin (faites-en 20-30).")
    print("'Q' : Passer à la classe suivante.")
    print("ESPACE : QUITTER TOUTE LA COLLECTE.")
    
    collecting = True
    while collecting and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Logique de dessin
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            current_point = (ix, iy)
            
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            distance = math.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)

            if distance < PINCH_THRESHOLD:
                if prev_point is None: 
                    prev_point = current_point
                
                # --- AJOUT: Zone Morte (Recommandation 2) ---
                # Calcule la distance du mouvement depuis le dernier point dessiné
                dist_moved = math.sqrt((current_point[0] - prev_point[0])**2 + (current_point[1] - prev_point[1])**2)
                
                # Ne dessine que si le mouvement est significatif
                if dist_moved > DEAD_ZONE_RADIUS:
                    cv2.line(canvas, prev_point, current_point, (255, 255, 255), LINE_THICKNESS)
                    prev_point = current_point
                # --- Fin Ajout ---
            
            else:
                prev_point = None
        else:
            prev_point = None

        # Affichage
        combined_display = cv2.add(frame, canvas)
        cv2.putText(combined_display, f"Classe: {class_name} | Images: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Collecte de Donnees - EmoDraw', combined_display)
        
        cv2.imshow('Canevas (Dessin Seul)', canvas)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            collecting = False 
        
        elif key == 32: # Touche Espace
            print("Arrêt de la collecte demandé.")
            collecting = False
            master_collecting_flag = False 
        
        elif key == ord('c'):
            canvas = np.zeros((h, w, 3), dtype=np.uint8) 
        
        elif key == ord('s'):
            # --- AJOUT: Filtre des dessins vides (Recommandation 3) ---
            # Compte le nombre de pixels blancs (le dessin)
            drawn_pixels = np.sum(canvas > 0)
            
            if drawn_pixels < 200: # Seuil (à ajuster si besoin)
                print("Dessin trop vide ou vide! Sauvegarde annulée.")
                continue # On ne sauvegarde pas, on n'efface pas
            # --- Fin Ajout ---
            
            # Sauvegarde de l'image prétraitée
            processed_data = drawing_utils.preprocess_drawing_for_model(canvas)
            
            if processed_data is not None:
                _, debug_image_28x28 = processed_data 
                
                save_preview = cv2.resize(debug_image_28x28, (280, 280), interpolation=cv2.INTER_NEAREST)
                cv2.imshow('Sauvegarde (28x28)', save_preview)
                
                filename = os.path.join(class_path, f"{class_name}_{count}.png")
                cv2.imwrite(filename, debug_image_28x28)
                print(f"Sauvegarde de : {filename}")
                
                count += 1
                canvas = np.zeros((h, w, 3), dtype=np.uint8) 
            else:
                print("Canevas vide, rien a sauvegarder.")

cap.release()
cv2.destroyAllWindows()
print("Collecte de données terminée.")

