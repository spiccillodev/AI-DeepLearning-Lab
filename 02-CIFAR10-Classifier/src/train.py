import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
import shutil
import time
from tqdm import tqdm

# --- CONFIGURAZIONE PERCORSI ---
script_dir = os.path.dirname(os.path.abspath(__file__))
local_data_path = os.path.join(script_dir, "data")
model_dir = os.path.join(script_dir, "models")
model_file = os.path.join(model_dir, "modello_cifar10.pt")

os.makedirs(model_dir, exist_ok=True)

# --- ARCHITETTURA RETE (Elite) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_stage = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2_final = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.final_stage = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512), nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv1_stage(x)
        x = self.conv2_final(x)
        x = self.final_stage(x)
        return x

# --- IL CANCELLO DI SICUREZZA PER WINDOWS ---
if __name__ == '__main__':
    # Spostiamo qui tutto ciò che avvia calcoli o processi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"🚀 Hardware pronto: {gpu_name}")

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root=local_data_path, train=True, download=True, transform=transform)
    # Su Windows, num_workers > 0 richiede il blocco if __name__ == '__main__'
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"🔥 Inizio sessione di Deep Learning (5 Epoche)")
    inizio_totale = time.time()

    for epoch in range(5):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(enumerate(trainloader), total=len(trainloader), leave=True)
        
        for i, (inputs, labels) in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            accuracy = 100. * correct / total
            loop.set_description(f"Epoca [{epoch+1}/5]")
            loop.set_postfix(loss=running_loss/(i+1), acc=f"{accuracy:.2f}%")

    fine_totale = time.time()
    print(f"\n✅ Training completato in {fine_totale - inizio_totale:.2f}s")

    torch.save(model.state_dict(), model_file)
    print(f"💾 Modello salvato in: {model_file}")