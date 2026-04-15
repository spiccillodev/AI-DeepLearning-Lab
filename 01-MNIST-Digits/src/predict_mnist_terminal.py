import os
import torch

# Questo comando trova la cartella dove si trova fisicamente lo script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Costruiamo il percorso corretto verso il modello
model_path = os.path.join(script_dir, "models", "mnist_classifier_v1.pt")

# Ora carichiamo usando il percorso dinamico
cervello = torch.load(model_path, weights_only=True)

# Vediamo cosa c'è dentro (i nomi degli strati della rete)
for strato in cervello.keys():
    print(f"Strato: {strato} | Dimensione: {cervello[strato].shape}")