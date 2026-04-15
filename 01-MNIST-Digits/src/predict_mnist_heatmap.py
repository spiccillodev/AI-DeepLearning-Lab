import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np

# --- CONFIGURAZIONE PERCORSI ESATTI (Z: Drive) ---
# Usiamo le stringhe raw (r"") per evitare problemi con i backslash di Windows
base_path = r"G:\Il mio Drive\CODING\Python\AI-DeepLearning-Lab\01-MNIST-Digits"
model_path = os.path.join(base_path, "models", "cervello_numeri.pt")
test_folder = os.path.join(base_path, "test_images")

# --- AUTO-CREAZIONE CARTELLA ---
if not os.path.exists(test_folder):
    os.makedirs(test_folder)
    print(f"📂 Cartella creata: {test_folder}")
    print("👉 Ora incolla una o più immagini (es. un numero scritto da te) in quella cartella!")

# --- ARCHITETTURA MNIST (Sincronizzata con il tuo training) ---
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),    # Livello 0: 28x28 -> 26x26
            nn.ReLU(),                         # Livello 1
            nn.MaxPool2d(2, 2),                # Livello 2: 26x26 -> 13x13
            nn.Conv2d(32, 64, kernel_size=3),   # Livello 3: 13x13 -> 11x11
            nn.ReLU(),                         # Livello 4
            nn.Flatten(),                      # Livello 5: 11x11x64 = 7744
            nn.Linear(7744, 128),               # Livello 6 (Quello dell'errore!)
            nn.ReLU(),                         # Livello 7
            nn.Linear(128, 10)                  # Livello 8
        )

    def forward(self, x):
        return self.conv(x)

def scan_neurons(image_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Caricamento Modello
    if not os.path.exists(model_path):
        print(f"❌ Errore: Non trovo il file del modello in {model_path}")
        return

    model = MNIST_CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Pre-elaborazione
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    img_full_path = os.path.join(test_folder, image_name)
    img = Image.open(img_full_path)
    img_tensor = transform(img).unsqueeze(0).to(device)

    # --- ESTRAZIONE "PENSIERO" ---
    with torch.no_grad():
        x = model.conv[0](img_tensor) # Conv1
        activations = model.conv[1](x) # ReLU
    
    activations = activations.squeeze().cpu().numpy()

    # --- DASHBOARD ---
    fig, axes = plt.subplots(4, 8, figsize=(12, 7))
    fig.canvas.manager.set_window_title(f'Brain Scan - {image_name}')
    fig.suptitle(f"Neuroni del 1° Strato: Risposta a '{image_name}'\nModello: cervello_numeri.pt", color='white', fontsize=14)
    fig.set_facecolor('#121212')

    for i, ax in enumerate(axes.flat):
        if i < 32:
            ax.imshow(activations[i], cmap='magma')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Cerchiamo file immagine nella cartella
    immagini = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if immagini:
        scan_neurons(immagini[0]) # Analizza la prima foto trovata
    else:
        print(f"\n⚠️  La cartella è pronta ma è VUOTA.")
        print(f"Percorso: {test_folder}")
        print("Metti un'immagine nera con un numero bianco e riavvia!")