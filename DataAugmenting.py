import os
import torchvision
from torchvision import datasets, transforms
from PIL import Image

# Définir le chemin vers le dossier de données
data_dir = 'Adataset/train'

# Définir les transformations pour l'augmentation des données
transform = transforms.Compose([
    transforms.RandomChoice([
        transforms.RandomRotation(3),
        transforms.RandomRotation(1),
        transforms.RandomRotation(2)
    ]),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mean(dim=0))
])

# Charger les données
dataset = datasets.ImageFolder(data_dir, transform=transform)
BaseDataset = datasets.ImageFolder(data_dir)
# Créer un nouveau dossier pour les images transformées
os.makedirs('Adataset_transformed/train', exist_ok=True)

# Parcourir les images du dataset et les sauvegarder dans le nouveau dossier
for i, (image, label) in enumerate(dataset):
    # Récupérer le nom du sous-dossier (classe) de l'image actuelle
    class_name = dataset.classes[label]

    # Créer le sous-dossier correspondant dans le dossier des images transformées
    os.makedirs(f'Adataset_transformed/train/{class_name}', exist_ok=True)

    # Convertir le tensor en une image PIL pour pouvoir la sauvegarder
    image = transforms.ToPILImage()(image)

    # Sauvegarder l'image dans le sous-dossier correspondant
    image.save(f'Adataset_transformed/train/{class_name}/image_{i}.png')
