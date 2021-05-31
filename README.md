# Rendu pour l'exercice de Exxact Robotics

>**but :**<br>
>utiliser le transfert learning sur le dataset de deepFruits pour apprendre YOLOv3 à détecter différentes sortes de fruits.

Le dataset deepFruits peut être téléchargé [ici](https://drive.google.com/drive/folders/1CmsZb1caggLRN7ANfika8WuPiywo4mBb).

# Etapes nécessaire pour entraîner un modèle YOLOv3 à détecter de nouvelles classes

## modifier le dataset de deepFruits pour qu'il puisse être utilisé comme input par les fichier train.py, test.py et detect.py du [github officiel de YOLOv3](https://github.com/ultralytics/yolov3).

Pour que le dataset soit compréhensible par YOLOv3, il faut que les images et les labels soient contenus dans une arborescence de dossier spécifique:
```
deepFruits_for_training/
    images/
        train/
            <all training images in jpg format>
        val/
            <all validation images in jpg format>
        test/
            <all testing images in jpg format>
    labels/
        train/
            <all labels of training images in txt format>
        val/
            <all labels of validation images in txt format>
        test/
            <all labels of testing images in txt format>
```

chaque fichier image du dossier images est annoté par un fichier texte du même nom dans le dossier labels. Comme les annotations fournies par le dataset de deepFruits ne sont pas celles que YOLO attend, il faut les réécrire. Il faut également convertir toutes les images en format JPEG, supprimer les images sans leurs annotations et les annotations sans leurs images, et redonner des bonnes nombres au labels (presque tous les fruits sont annotés avec la classe "rockmelon").

## cloner le contenu du github yolov3 dans le dossier actuel

## créer un fichier deepFruits.yaml

ce fichier contiendra:
- le chemin vers chaque dossier images/train, val et test
- le nombre de classes (ici 7)
- le nom de chaque classe

## utiliser train.py avec comme argumet les poids du modèle préentraîné yolov3.pt, et le chemin vers le fichier .yaml créé.

On pourra alors utiliser detect.py pour effectuer de nouvelles inférences, en utilisant les nouveaux poids généré par l'entraînement.

# Ce que j'ai fait

Je n'ai pu que commencer le fichier qui permet de générer un nouveau dataset pour YOLO.

Il permet de rassembler toutes les images de fruits aux bons endroits (train et test), il les converti en jpeg, et extracte les annotations des fichiers test_RGB.txt et train_RGB.txt pour créer un nouveau fichier texte par ligne.

Il me reste à convertir **x1, y1, x2, y2** en **center_x, center_y, width, height**, normalisé de 0 à 1, pour chaque box annotée de chaque image, et à changer le numéro de classe pour certains fruits.

## Comment l'utiliser

1. cloner le répo github sur votre machine:<br>```git clone https://github.com/Gwizdo51/yolov3_deepFruits.git```
1. créer un nouvel environnement virtuel, avec **Python 3.8**.
2. ```pip install -r requirements.txt``` 
3. copier le contenu du drive de [deepFruits_dataset](https://drive.google.com/drive/folders/1CmsZb1caggLRN7ANfika8WuPiywo4mBb) dans le même dossier.
4. ```python prepare_deepfruits_for_training.py --input_dir deepFruits_dataset```

le script va créer un nouveau dossier rassemblant les informations pour l'entraînement de YOLO.
