import torch
import torchvision
import torchvision.transforms as transforms
from cnn import Net

#Script to test 1 model

# Define the path to the model and the classes
PATH = './model/model_20240410_130807_30'
classes = {0: 'NonDemented', 1: 'VeryMildDemented'}

# Définir la transformation
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Grayscale(num_output_channels=1)
])

# Load the test dataset and the model and GPU
data_dir = 'Dataset'
dataset = torchvision.datasets.ImageFolder(data_dir+'/test', transform=transform)
testloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
net = Net()
net.load_state_dict(torch.load(PATH))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

# Test the model
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)  # Move the images and labels to the GPU
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')