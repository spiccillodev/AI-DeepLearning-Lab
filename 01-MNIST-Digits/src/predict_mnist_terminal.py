"""
Ispezione dei Pesi del Modello MNIST
------------------------------------
Questo script carica il file dei pesi (.pt) addestrato e ne
stampa l'architettura interna (strati e dimensioni dei tensori).
Ottimo per verificare il corretto salvataggio della rete neurale.
"""

import torch
from pathlib import Path

# 1. Importiamo il nostro pannello di controllo!
from config import MODEL_SAVE_PATH

def inspect_model_weights(model_path: Path) -> None:
    """
    Carica un file state_dict di PyTorch e stampa a schermo
    una tabella con i nomi degli strati e la forma dei tensori.
    """
    # Controllo di sicurezza
    if not model_path.exists():
        print(f"❌ Errore: Il file '{model_path.name}' non esiste.")
        print(f"📂 Cercato in: {model_path.parent}")
        print("💡 Suggerimento: Esegui prima mnist_ai.py per addestrare il modello!")
        return

    print(f"\n🔍 Analisi dei tensori in: {model_path.name}")
    print("=" * 60)

    # Carichiamo il "cervello" in RAM (CPU) per ispezionarlo in modo leggero
    cervello = torch.load(model_path, map_location="cpu", weights_only=True)

    # Iteriamo su tutti gli strati usando .items() per avere sia nome che tensore
    for strato, tensore in cervello.items():
        # Trucco da Pro: {strato:<25} aggiunge spazi per allineare le colonne!
        forma = list(tensore.shape)
        print(f" Strato: {strato:<25} | Dimensione: {forma}")
        
    print("=" * 60)
    print("✅ Ispezione completata con successo!\n")

# --- GUARDIA DI ESECUZIONE ---
if __name__ == "__main__":
    # Passiamo semplicemente il percorso che avevamo già salvato nel config
    inspect_model_weights(MODEL_SAVE_PATH)