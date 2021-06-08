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

chaque fichier image du dossier images est annoté par un fichier texte du même nom dans le dossier labels. Comme les annotations fournies par le dataset de deepFruits ne sont pas celles que YOLO attend, il faut les réécrire. Il faut également convertir toutes les images en format JPEG, supprimer les images sans leurs annotations et les annotations sans leurs images, et donner des bons nombres au labels (presque tous les fruits sont annotés avec la classe "rockmelon").

## créer un fichier deepFruits.yaml

ce fichier contiendra:
- le chemin vers chaque dossier images/train, val et test
- le nombre de classes (ici 7)
- le nom de chaque classe

## cloner le contenu du github yolov3 dans le dossier actuel

## utiliser train.py avec comme arguments les poids du modèle préentraîné yolov3.pt, et le chemin vers le fichier .yaml créé.

On pourra alors utiliser detect.py pour effectuer de nouvelles inférences, en utilisant les nouveaux poids généré par l'entraînement.

# Ce que j'ai fait

J'ai entraîné YOLOv3 sur la totalité du dataset, sur 100 epochs. Les nouveaux poids peuvent être téléchargé [ici](https://drive.google.com/drive/folders/1BdsYVVx7YsC8sNsFYzwCFAv82btq0ZHG) (yolov3_deepFruits_100epochs.pt), ou en utilsant "download_deepfruits_dataset.py":
```python
from download_deepfruits_dataset import download_file_from_google_drive

download_file_from_google_drive("1ij4hEXrvv9158-qxJxXhxl0URHKDnalo", "yolov3_deepFruits_100epochs.pt")
```

## Comment l'utiliser

Entraînement avec deepFruits : [1_yolov3_training_on_deepFruits.ipynb](https://github.com/Gwizdo51/yolov3_deepFruits/blob/dev/1_yolov3_training_on_deepFruits.ipynb)

Tests sur les nouveaux poids : [2_yolov3_testing_new_weights.ipynb](https://github.com/Gwizdo51/yolov3_deepFruits/blob/dev/2_yolov3_testing_new_weights.ipynb)