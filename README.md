# 🎨 PicDraw AI : Reconnaissance de Dessin Gestuel en Temps Réel

EmoDraw est une application Python qui permet de dessiner "en l'air" devant sa webcam en utilisant le suivi de la pointe du doigt. Grâce à l'intelligence artificielle (CNN), l'application prédit en temps réel la forme dessinée et affiche une image correspondante.

## 🚀 Fonctionnalités Clés

- **Dessin Gestuel (Air Drawing)** : Utilise MediaPipe pour suivre l'index. Le dessin s'active par un geste de "pincement" (pouce + index).
- **IA Personnalisée** : Inclut un pipeline complet pour collecter vos propres données, entraîner un modèle de réseau de neurones convolutifs (CNN) et l'utiliser.
- **Prétraitement Avancé** : Isolation automatique du dessin, centrage, redimensionnement en 28x28 pixels et inversion des couleurs pour une précision maximale.

## 🛠️ Technologies Utilisées

- **Langage** : Python 3.11
- **Vision par Ordinateur** : OpenCV
- **Suivi de Main** : MediaPipe
- **Intelligence Artificielle** : TensorFlow / Keras
- **Traitement de Données** : NumPy

## 📦 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-compte/emodraw.git
cd emodraw
```

### 2. Créer un environnement virtuel (Recommandé)

**Windows :**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**Linux/Mac :**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install tensorflow==2.15.0 opencv-python mediapipe numpy requests tqdm
```

## 🎮 Utilisation : Mode d'Emploi

Le projet est divisé en trois phases pour garantir une précision optimale lors des démonstrations.

### 1. Collecte de Données

Lancez le script de collecte pour créer votre propre base d'images (20 à 30 par forme).

```bash
python collect_data.py
```

**Commandes :**
- **Pincement** : Dessiner
- **S** : Sauvegarder le dessin actuel
- **C** : Effacer le canevas
- **Q** : Passer à la forme suivante
- **Espace** : Quitter la collecte

### 2. Entraînement de l'IA

Une fois vos dossiers d'images remplis dans `/dataset`, entraînez votre modèle personnalisé.

```bash
python train_model.py
```

**Fichiers générés :**
- `my_demo_model.h5` : Modèle entraîné
- `my_demo_classes.txt` : Liste des classes

### 3. Application Principale

Lancez la démonstration finale.

```bash
python main_app.py
```

**Commandes :**
- **P** : Lancer la prédiction
- **C** : Effacer pour un nouveau dessin
- **Q** : Quitter

## 📂 Structure du Projet

| Fichier | Rôle |
|---------|------|
| `main_app.py` | Point d'entrée de l'application (Boucle principale) |
| `collect_data.py` | Outil de création du jeu de données via webcam |
| `train_model.py` | Architecture du CNN et script d'entraînement |
| `drawing_utils.py` | Algorithmes de prétraitement d'image (Crop, Resize, Invert) |
| `setup_tools.py` | Gestionnaire de chargement du modèle et des classes locales |
| `images_pred/` | Dossier contenant les PNG à afficher après prédiction |

## 🧠 Algorithme de Prétraitement

Pour garantir que l'IA comprenne le dessin, chaque tracé subit la transformation suivante avant d'être injecté dans le modèle :

1. **Bounding Box** : Détection des limites du dessin
2. **Square Padding** : Ajout de marges pour éviter la déformation
3. **Resizing** : Réduction à 28x28 pixels (format MNIST)
4. **Bitwise Not** : Inversion pour obtenir un trait noir sur fond blanc (conforme à l'entraînement standard)

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## 👋 Contact

**Njomani Sonta Yan Fred*
- LinkedIn : https://linkedin.com/in/Fred-Njomani
- Email : njomanifred@gmail.com
