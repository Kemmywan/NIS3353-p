import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.lenet import LeNet
import matplotlib.pyplot as plt
from utils import fgsm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

batch_size = 64
lr = 0.01 #learning rate
epochs = 10

# Preprocess

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # 标准化，mean = (0.1307,) std = (0.3081,) x=(x-mean)/std
])

train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9)

losses = []

for epoch in range(epochs):
    model.train() # Switch to training mode
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        atk_images = fgsm(model, criterion, images, labels, epsilon=0.3)

        mixed_images = torch.cat((images, atk_images), 0)
        mixed_labels = torch.cat((labels, labels), 0)
        optimizer.zero_grad() # Zero the parameter gradients
        outputs = model(mixed_images)
        loss = criterion(outputs, mixed_labels)
        loss.backward() # Backward pass
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), 'lenet_fgsm.pth')
print("Training complete and model saved as lenet_fgsm.pth")

# draw loss curve

plt.figure()
plt.plot(range(1, epochs + 1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.savefig('loss_curve_fgsm.png')
plt.show()
