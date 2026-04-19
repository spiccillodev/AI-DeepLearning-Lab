"""
Esportatore ONNX per Netron
--------------------------
Converte il modello PyTorch (.pt) nel formato universale ONNX.
ONNX permette di visualizzare l'architettura della rete su https://netron.app
"""

import torch
import os
from pathlib import Path

# 1. Importiamo la nostra configurazione e l'architettura
from config import BASE_DIR, MODEL_SAVE_PATH
from mnist_ai import DigitNet

# --- SETUP PERCORSI ---
# Creiamo una cartella dedicata per le visualizzazioni
OUTPUT_FOLDER = BASE_DIR / "netron_visualization"
OUTPUT_ONNX = OUTPUT_FOLDER / "mnist_model_graph.onnx"

def export_to_onnx() -> None:
    """
    Carica i pesi del modello e genera un file .onnx 
    rappresentativo del grafo computazionale.
    """
    # 1. Controllo esistenza modello
    if not MODEL_SAVE_PATH.exists():
        print(f"❌ Errore: Non trovo il file .pt in {MODEL_SAVE_PATH}")
        print("💡 Suggerimento: Addestra il modello con mnist_ai.py prima di esportarlo.")
        return

    # 2. Creazione cartella di output se non esiste
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # 3. Preparazione modello (usiamo la CPU per l'esportazione)
    device = torch.device("cpu")
    model = DigitNet().to(device)
    
    try:
        # Caricamento pesi
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
        model.eval()
        print("✅ Pesi caricati correttamente.")
    except Exception as e:
        print(f"⚠️ Errore durante il caricamento: {e}")
        return

    # 4. Creazione Dummy Input
    # ONNX ha bisogno di un "esempio" per capire le dimensioni (1 immagine, 1 canale, 28x28)
    dummy_input = torch.randn(1, 1, 28, 28)

    # 5. Esportazione ONNX
    print(f"🚀 Generazione grafico ONNX per Netron...")
    try:
        torch.onnx.export(
            model, 
            (dummy_input,),       # <--- MODIFICA: Ora è una tupla ufficiale (Tensor,)
            str(OUTPUT_ONNX), 
            export_params=True, 
            opset_version=11, 
            do_constant_folding=True, 
            input_names=['Input_Immagine'], 
            output_names=['Output_Previsione']
        )
        
        print("-" * 50)
        print(f"✨ ESPORTAZIONE COMPLETATA ✨")
        print(f"📍 File creato: {OUTPUT_ONNX}")
        print(f"👉 Ora vai su https://netron.app e trascina il file lì dentro!")
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Errore durante l'esportazione ONNX: {e}")

if __name__ == "__main__":
    export_to_onnx()