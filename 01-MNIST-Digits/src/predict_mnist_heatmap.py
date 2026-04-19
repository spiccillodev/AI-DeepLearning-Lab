"""
Visualizzazione delle Attivazioni (Heatmap) per MNIST
-----------------------------------------------------
Questo script carica un'immagine di test e mostra come i primi
strati convoluzionali del modello "vedono" l'immagine (Feature Maps).
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# 1. Importiamo il nostro pannello di controllo e l'architettura
from config import BASE_DIR, MODEL_SAVE_PATH, DEVICE, NORM_MEAN, NORM_STD
from mnist_ai import DigitNet

# --- SETUP PERCORSO TEST ---
TEST_FOLDER = BASE_DIR / "test"
TEST_FOLDER.mkdir(parents=True, exist_ok=True)

def scan_neurons(image_name: str) -> None:
    """
    Carica un'immagine, la processa e visualizza le mappe di attivazione
    del primo strato convoluzionale del modello addestrato.
    """
    if not MODEL_SAVE_PATH.exists():
        print(f"❌ Errore: Non trovo il file del modello in {MODEL_SAVE_PATH}")
        print("💡 Suggerimento: Hai eseguito mnist_ai.py per addestrare il modello prima?")
        return

    model = DigitNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    # Pre-elaborazione
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((NORM_MEAN,), (NORM_STD,))
    ])

    # Lettura immagine
    img_full_path = TEST_FOLDER / image_name
    img = Image.open(img_full_path)
    
    # FIX PYLANCE 1: Dichiariamo esplicitamente che il risultato è un Tensore
    img_tensor: torch.Tensor = transform(img) # type: ignore
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    # --- ESTRAZIONE "PENSIERO" (Feature Maps) ---
    with torch.no_grad():
        x = model.conv[0](img_tensor)
        activations = model.conv[1](x)
    
    activations = activations.squeeze().cpu().numpy()

    # --- DASHBOARD MATPLOTLIB ---
    fig, axes = plt.subplots(4, 8, figsize=(12, 7))
    
    # FIX PYLANCE 2: Controllo di sicurezza prima di impostare il titolo della finestra
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(f'Brain Scan - {image_name}')
        
    fig.suptitle(f"Neuroni del 1° Strato: Risposta a '{image_name}'\nModello: {MODEL_SAVE_PATH.name}", color='white', fontsize=14)
    fig.set_facecolor('#121212')

    for i, ax in enumerate(axes.flat):
        if i < 32:
            ax.imshow(activations[i], cmap='magma')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# --- ESECUZIONE PRINCIPALE ---
if __name__ == "__main__":
    immagini = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if immagini:
        print(f"🔍 Trovata immagine: {immagini[0]}. Avvio scansione...")
        scan_neurons(immagini[0]) 
    else:
        print(f"\n⚠️ La cartella di test è pronta ma è VUOTA.")
        print(f"📂 Percorso: {TEST_FOLDER}")
        print("👉 Inserisci un'immagine (preferibilmente di dimensioni quadrate, sfondo nero e numero bianco) e riavvia lo script!")