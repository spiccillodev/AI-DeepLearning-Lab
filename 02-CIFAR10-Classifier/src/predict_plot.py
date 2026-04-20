import time
import random
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle # Fix Pylance Rectangle
from PIL import Image

# --- STANDARD LAB IMPORTS (Assicurati che config.py sia aggiornato) ---
from config import (
    DEVICE, CLASSES, MODEL_PATH, DATA_DIR, TEST_IMAGES_DIR, 
    LOGS_DIR, PREDICTIONS_DIR, BASE_DIR, NORM_MEAN, NORM_STD, HISTORY_FILE
)
from model import CifarNet

# --- CONFIGURAZIONE ESTETICA ---
plt.style.use('dark_background')
COLOR_PALETTE = ['#2ecc71', '#3498db', '#e74c3c'] # Verde, Blu, Rosso

# --- MAPPA DEL RAGIONAMENTO (XAI) ---
REASONING_MAP = {
    'aereo': "Gradienti orizzontali netti (ali) e sfondo uniforme (cielo/pista).",
    'auto': "Riflessi metallici speculari e pattern circolari (ruote/fari).",
    'uccello': "Texture organica piumata e silhouette con appendici sottili.",
    'gatto': "Feature micro-geometriche: orecchie e gradienti radiali oculari.",
    'cervo': "Silhouette snella con arti verticali e ramificazioni superiori.",
    'cane': "Volumetria del muso prominente e texture di pelliccia irregolare.",
    'rana': "Saturazione cromatica specifica e riflessi umidi sulla pelle.",
    'cavallo': "Asse dorsale muscoloso e profilo del cranio allungato.",
    'nave': "Massa scura orizzontale su texture periodica bluastra (onde).",
    'camion': "Box-volumes massicci con ripetizione di elementi rotanti."
}

# --- LOGICA GRAD-CAM (XAI) ---
class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        # Usiamo Optional per i type hint di Pylance
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output): 
            self.activations = output
            
        def backward_hook(module, grad_in, grad_out): 
            self.gradients = grad_out[0]
        
        # Registriamo gli hook per catturare attivazioni e gradienti
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        # Controllo di sicurezza dinamico
        if self.gradients is None or self.activations is None:
            return np.zeros((32, 32), dtype=np.float32)

        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        # Global Average Pooling dei gradienti
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        heatmap = torch.sum(weights * activations, dim=1).squeeze()
        
        # ReLU sulla heatmap (teniamo solo le zone che contribuiscono positivamente)
        heatmap_np = np.maximum(heatmap.cpu().numpy(), 0)
        max_val = np.max(heatmap_np)
        if max_val > 0:
            heatmap_np /= max_val
            
        return heatmap_np

# --- UTILITIES ---
def get_reference_images():
    """Scarica (se necessario) immagini di riferimento dal test set ufficiale."""
    dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True)
    refs = {}
    for img, label in dataset:
        if label not in refs: refs[label] = img
        if len(refs) == 10: break
    return refs

def get_smart_file():
    """Seleziona un'immagine di test evitando ripetizioni basandosi sulla cronologia."""
    files = [f.name for f in TEST_IMAGES_DIR.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    if not files:
        raise FileNotFoundError(f"Nessuna immagine trovata in {TEST_IMAGES_DIR}")

    history = []
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f: 
            history = [l.strip() for l in f.readlines()]
    
    available = [f for f in files if f not in history]
    selected = random.choice(available if available else files)
    
    # Aggiorna la cronologia (mantiene gli ultimi 3)
    history = (history + [selected])[-3:]
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f: 
        for s in history: f.write(f"{s}\n")
    return selected

def log_result(filename: str, pred: str, conf: float):
    """Registra l'esito nel CSV dentro outputs/logs/ con encoding UTF-8."""
    # PUNTO 2: Percorso corretto outputs/logs/
    csv_path = LOGS_DIR / "classificazioni_log.csv"
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    target = Path(filename).stem.lower()
    status = "✅ CORRETTO" if pred.lower() == target else "❌ SBAGLIATO"
    
    # encoding='utf-8' per supportare ✅ e ❌ su Windows
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([now, filename, pred.upper(), target.upper(), f"{conf:.1f}%", status])
    return status

# --- UI DASHBOARD ---
def render_dashboard(img_path: Path, top_data: tuple, heatmap: np.ndarray, refs: dict, hw_info: str, status: str):
    """Genera, salva e mostra la dashboard diagnostica XAI."""
    best_label, best_conf, top_probs, top_idxs = top_data
    
    fig = plt.figure(figsize=(16, 10), facecolor='#121212')
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.2, 1], hspace=0.3)
    
    status_color = COLOR_PALETTE[0] if "✅" in status else COLOR_PALETTE[2]
    fig.suptitle(f"CIFAR-10 XAI DIAGNOSTIC | {status}\n{hw_info}", 
                    color=status_color, fontsize=20, fontweight='bold', y=0.97)

    # 1. Pannello Heatmap (Grad-CAM) overlayed sull'originale
    ax_cam = fig.add_subplot(gs[0, 0])
    img_bgr = cv2.imread(str(img_path))
    heatmap_res = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap_col = cv2.applyColorMap(np.uint8(255 * heatmap_res), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_col, 0.4, 0)
    
    ax_cam.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax_cam.set_title(f"ATTENZIONE IA: {best_label.upper()}", color=COLOR_PALETTE[0], pad=12)
    ax_cam.axis('off')

    # 2. Pannello Grafico a Barre delle Probabilità
    ax_bar = fig.add_subplot(gs[0, 1:])
    y_labels = [CLASSES[int(i)].upper() for i in top_idxs]
    values = [float(p) for p in top_probs]
    
    bars = ax_bar.barh(y_labels, values, color=COLOR_PALETTE, height=0.6)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 110)
    for bar in bars:
        ax_bar.text(bar.get_width() + 2, bar.get_y() + 0.35, f"{bar.get_width():.1f}%", 
                    color='white', fontweight='bold')
    
    # Check Ambiguità spaziale tra TOP 1 e TOP 2
    if (values[0] - values[1]) < 15:
        ax_bar.text(50, -0.5, "⚠️ SEGNALE DEBOLE / AMBIGUITÀ RILEVATA", color='#f1c40f', 
                    ha='center', bbox=dict(facecolor='#121212', edgecolor='#f1c40f', pad=5))

    # 3. Pannelli Confronto con immagini di riferimento ufficiali
    for i in range(3):
        idx = int(top_idxs[i])
        ax_ref = fig.add_subplot(gs[1, i])
        ax_ref.imshow(refs[idx])
        ax_ref.axis('off')
        
        # Cornice colorata (Fix Rectangle per Pylance)
        rect = Rectangle((0,0), 31, 31, linewidth=6, edgecolor=COLOR_PALETTE[i], facecolor='none')
        ax_ref.add_patch(rect)
        ax_ref.set_title(f"{CLASSES[idx].upper()} ({values[i]:.1f}%)", color=COLOR_PALETTE[i], pad=8)
        
        # Aggiungiamo la logica XAI solo per la predizione principale
        if i == 0:
            logic = REASONING_MAP.get(best_label, "Analisi pattern complessi.")
            ax_ref.text(16, 42, f"LOGICA: {logic}", color='white', fontsize=10, ha='center', style='italic', wrap=True)

    # --- PUNTO 3: GESTIONE MANIACALE OUTPUT ---
    # Generiamo un nome file unico basato sulla classe predetta e sull'ora esatta
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"pred_{best_label}_{timestamp}.png"
    # Costruiamo il percorso completo: outputs/predictions/pred_CLASSE_DATA_ORA.png
    save_path = PREDICTIONS_DIR / save_name
    
    # Salviamo l'immagine DPI-optimized PRIMA di mostrarla (plt.show() resetta la figura)
    plt.savefig(save_path, dpi=150, facecolor='#121212')
    print(f"🖼️  Analisi diagnostica salvata in ordinatamente in: {save_path}")
    
    # Mostriamo la dashboard a video
    plt.show()

# --- EXECUTION ---
if __name__ == "__main__":
    try:
        hw_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        
        # Caricamento Modello CifarNet modulare
        model = CifarNet().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        
        # Preparazione Dati
        ref_images = get_reference_images()
        filename = get_smart_file()
        full_path = TEST_IMAGES_DIR / filename
        
        # Pipeline Preprocessing sincronizzata con il training
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD)
        ])
        
        # Caricamento Immagine
        raw_img = Image.open(full_path).convert('RGB')
        # Hint per Pylance: specifichiamo che è un Tensor
        input_tensor: torch.Tensor = transform(raw_img) # type: ignore
        input_batch = input_tensor.unsqueeze(0).to(DEVICE)

        # 1. Inferenza Standard
        model.eval()
        with torch.no_grad():
            output = model(input_batch)
            probs = F.softmax(output, dim=1)[0] * 100
            top_p, top_i = torch.topk(probs, 3)
            
        # Cast espliciti (Anti-Pylance warning)
        best_class = CLASSES[int(top_i[0])]
        best_conf = float(top_p[0])
        
        # Registriamo l'esito
        status = log_result(filename, best_class, best_conf)

        # 2. Generazione XAI (Richiede gradienti abilitati)
        # Puntiamo all'ultimo strato convoluzionale (conv3) della CifarNet
        cam = GradCAM(model, model.conv3) 
        heatmap = cam.generate(input_batch, int(top_i[0]))

        # 3. Visualizzazione e Salvataggio Ordinato (Point 3 inside)
        render_dashboard(full_path, (best_class, best_conf, top_p, top_i), heatmap, ref_images, hw_name, status)

    except Exception as e:
        print(f"❌ ERRORE CRITICO: {e}")