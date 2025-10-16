import torch
from utils import fgsm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.lenet import LeNet
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

epsilon_values = [0.00, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # 标准化，mean = (0.1307,) std = (0.3081,) x=(x-mean)/std
])

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device)
model.load_state_dict(torch.load('lenet_fgsm.pth', map_location=device))
model.eval()

loss_fn = torch.nn.CrossEntropyLoss()

accuracies = []

for epsilon in epsilon_values:

    correct, total = 0, 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        atk_images = fgsm(model, loss_fn, images, labels, epsilon)

        outputs = model(atk_images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracies.append(100 * correct / total)
    print(f"Epsilon: {epsilon:.2f} Test Accuracy: {100 * correct / total:.2f}%")

plt.figure()
plt.plot(epsilon_values, accuracies, marker='o')
plt.xlabel('Epsilon')
plt.ylabel('Accuracy(%)')
plt.title('FGSM Attack Effect on Model Accuracy')
plt.grid(True)
plt.savefig('fgsm_attack_after_train.png')
plt.show()