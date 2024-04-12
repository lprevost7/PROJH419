import torch
import torchvision
import torchvision.transforms as transforms

from UpgradeNet import Net


# Définir le chemin vers le modèle entraîné
PATH = './model/model_20240410_130807_30'

# Définir les classes
classes = {0: 'NonDemented', 1: 'VeryMildDemented'}

# Définir la transformation
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Grayscale(num_output_channels=1)
])

# Charger le jeu de données de test
data_dir = 'Dataset'
dataset = torchvision.datasets.ImageFolder(data_dir+'/test', transform=transform)
testloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# Charger le modèle
net = Net()
net.load_state_dict(torch.load(PATH))

# Vérifier si CUDA est disponible et définir l'appareil PyTorch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)  # Déplacer le modèle vers GPU

# Tester le modèle
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)  # Déplacer les images et les étiquettes vers GPU
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')
