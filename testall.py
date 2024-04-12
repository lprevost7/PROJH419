import os
import torch
import torchvision
import torchvision.transforms as transforms
from cnn import Net 

# Script to test all models

# Define the path to the directory containing the trained models
MODELS_DIR = './model/'

# Define the classes
classes = {0: 'NonDemented', 1: 'VeryMildDemented'}

# Define the transformation
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Grayscale(num_output_channels=1)
])

# Load the test dataset
data_dir = 'Dataset'
dataset = torchvision.datasets.ImageFolder(data_dir + '/test', transform=transform)
testloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# Get a list of all model files in the directory
model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('model_')]

# Initialize an empty dictionary to store test results
results = {}

# Iterate over each model file
for model_file in model_files:
    # Load the model
    net = Net()
    net.load_state_dict(torch.load(os.path.join(MODELS_DIR, model_file))) 

    # Check if CUDA is available and set the PyTorch device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)  # Move the model to GPU if available

    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)  # Move images and labels to GPU
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct // total
    results[model_file] = accuracy

# Print the results
for model_file, accuracy in results.items():
    print(f"Model {model_file}: Accuracy = {accuracy}%")
