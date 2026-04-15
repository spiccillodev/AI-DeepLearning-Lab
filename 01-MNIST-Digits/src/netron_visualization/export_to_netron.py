import torch
import torch.nn as nn
import os

# --- ARCHITETTURA SINCRONIZZATA (7744 Input Features) ---
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),    # Strato 0: 28x28 -> 26x26
            nn.ReLU(),                         # Strato 1
            nn.MaxPool2d(2, 2),                # Strato 2: 26x26 -> 13x13
            nn.Conv2d(32, 64, kernel_size=3),   # Strato 3: 13x13 -> 11x11
            nn.ReLU(),                         # Strato 4
            nn.Flatten(),                      # Strato 5: 11x11x64 = 7744
            nn.Linear(7744, 128),               # Strato 6
            nn.ReLU(),                         # Strato 7
            nn.Linear(128, 10)                  # Strato 8
        )
    def forward(self, x): return self.conv(x)

# --- CONFIGURAZIONE PERCORSI (Z: Drive) ---
# Usiamo r"" per gestire correttamente gli spazi e i backslash di Windows
base_path = r"Z:\Il mio Drive\CODING\Python\AI-DeepLearning-Lab\01-MNIST-Digits"
model_path = os.path.join(base_path, "models", "cervello_numeri.pt")
# Salviamo l'output nella cartella dedicata alla visualizzazione
output_folder = os.path.join(base_path, "netron_visualization")
output_onnx = os.path.join(output_folder, "cervello_visibile.onnx")

def export():
    # 1. Controllo esistenza modello
    if not os.path.exists(model_path):
        print(f"❌ Errore: Non trovo il file .pt in {model_path}")
        return

    # 2. Preparazione modello
    device = torch.device("cpu")
    model = MNIST_CNN()
    
    try:
        # Carichiamo i pesi con la protezione safe_globals attiva
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print("✅ Pesi caricati correttamente.")
    except Exception as e:
        print(f"⚠️ Errore durante il caricamento: {e}")
        return

    # 3. Creazione Dummy Input (1 immagine, 1 canale, 28x28 pixel)
    dummy_input = torch.randn(1, 1, 28, 28)

    # 4. Esportazione ONNX
    print(f"🚀 Generazione grafico per Netron...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_onnx, 
        export_params=True, 
        opset_version=11, 
        do_constant_folding=True, 
        input_names=['Input_Immagine'], 
        output_names=['Output_Digit_Class']
    )

    print("-" * 50)
    print(f"✨ OPERAZIONE COMPLETATA ✨")
    print(f"📍 File creato: {output_onnx}")
    print(f"👉 Ora trascina questo file su https://netron.app")
    print("-" * 50)

if __name__ == "__main__":
    export()