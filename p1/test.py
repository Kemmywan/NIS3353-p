import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.lenet import LeNet

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # 标准化，mean = (0.1307,) std = (0.3081,) x=(x-mean)/std
])

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device)
model.load_state_dict(torch.load('lenet.pth', map_location=device))
model.eval()

correct, total = 0, 0

with torch.no_grad(): # banning gradient calculation
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")


