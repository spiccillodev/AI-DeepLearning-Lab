import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cv2
import csv
from datetime import datetime

# --- CONFIGURAZIONE ESTETICA E PERCORSI ---
plt.style.use('dark_background')
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "models", "modello_cifar10.pt")
test_folder = os.path.join(script_dir, "test_images")
data_path = os.path.join(script_dir, "data")
history_file = os.path.join(script_dir, ".prediction_history.txt")
log_file = os.path.join(script_dir, "classificazioni_log.csv")

classes = ('aereo', 'auto', 'uccello', 'gatto', 'cervo', 
            'cane', 'rana', 'cavallo', 'nave', 'camion')

# --- MAPPA DEL RAGIONAMENTO (IL PENSIERO DELL'IA) ---
reasoning_map = {
    'aereo': "Vettori orizzontali dominanti e contrasto netto con sfondo uniforme (cielo/pista).",
    'auto': "Riflessi metallici speculari e pattern circolari ad alta densità (ruote/fari).",
    'uccello': "Texture organica piumata e silhouette con appendici sottili (becco/zampe).",
    'gatto': "Feature micro-geometriche: orecchie triangolari e gradienti radiali oculari.",
    'cervo': "Silhouette snella con pattern di arti verticali lunghi e ramificazioni superiori.",
    'cane': "Volumetria del muso prominente e texture di pelliccia a grana grossa/irregolare.",
    'rana': "Saturazione cromatica specifica e riflessi di tipo speculare/umido sulla pelle.",
    'cavallo': "Asse dorsale orizzontale muscoloso e cranio allungato con profilo lineare.",
    'nave': "Massa scura orizzontale posizionata su texture periodica bluastra (onde).",
    'camion': "Box-volumes massicci e squadrati con ripetizione seriale di elementi rotanti."
}

# --- ARCHITETTURA ELITE ---
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
        self.gradients = None

    def activations_hook(self, grad): self.gradients = grad
    def get_activations_gradient(self): return self.gradients
    def get_activations(self, x): return self.conv2_final(self.conv1_stage(x))

    def forward(self, x):
        x = self.conv1_stage(x)
        x = self.conv2_final(x)
        if x.requires_grad: x.register_hook(self.activations_hook)
        x = self.final_stage(x)
        return x

# --- LOGGING CSV ---
def log_prediction_result(filename, prediction, confidence):
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    real_class = os.path.splitext(filename)[0].lower()
    status = "✅ AZZECCATO" if prediction.lower() == real_class else "❌ SBAGLIATO"
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Data/Ora", "File", "Predizione IA", "Classe Reale", "Confidenza", "Esito"])
        writer.writerow([now, filename, prediction.upper(), real_class.upper(), f"{confidence:.2f}%", status])
    return status

# --- GRAD-CAM ---
def generate_grad_cam(model, input_tensor, target_index):
    model.eval()
    input_tensor.requires_grad_(True)
    output = model(input_tensor)
    score = output[0, target_index]
    model.zero_grad()
    score.backward()
    gradients = model.get_activations_gradient()
    activations = model.get_activations(input_tensor).detach()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(64):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze().cpu()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (torch.max(heatmap) + 1e-8)
    return heatmap.numpy()

# --- UTILS ---
def get_ref_images():
    ds = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True)
    refs = {}
    for img, label in ds:
        if label not in refs: refs[label] = img
        if len(refs) == 10: break
    return refs

def get_smart_file(files):
    history = []
    if os.path.exists(history_file):
        with open(history_file, 'r') as f: history = [l.strip() for l in f.readlines()]
    available = [f for f in files if f not in history]
    selected = random.choice(available) if available else random.choice(files)
    history = (history + [selected])[-2:]
    with open(history_file, 'w') as f: [f.write(f"{s}\n") for s in history]
    return selected

# --- DASHBOARD MASTER ---
def show_master_dashboard(img_path, result_data, heatmap, ref_imgs, hw_info, ambiguity_msg, status):
    label, conf, top_probs, top_idxs = result_data
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    fig = plt.figure(figsize=(16, 11), facecolor='#121212')
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.2, 1])
    
    fig.suptitle(f"CIFAR-10 ELITE ANALYSIS | {status}\n{hw_info}", 
                    color='white' if "✅" in status else '#e74c3c', 
                    fontsize=18, fontweight='bold', y=0.96)

    # 1. Focus Heatmap
    ax_main = fig.add_subplot(gs[0, 0])
    img_bgr = cv2.imread(img_path)
    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)
    ax_main.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    ax_main.set_title(f"DETTAGLI FEATURE: {label.upper()}", color=colors[0], pad=10, fontweight='bold')
    ax_main.axis('off')

    # 2. Probabilità
    ax_bar = fig.add_subplot(gs[0, 1:])
    y_labs = [classes[i.item()].upper() for i in top_idxs]
    v = [p.item() for p in top_probs]
    bars = ax_bar.barh(y_labs, v, color=colors, height=0.6)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 115)
    for bar in bars:
        ax_bar.text(bar.get_width()+1, bar.get_y()+0.35, f"{bar.get_width():.1f}%", color='white', fontweight='bold')

    if ambiguity_msg:
        ax_bar.text(55, -0.6, ambiguity_msg, color='#f1c40f', fontweight='bold', ha='center', 
                    bbox=dict(facecolor='#121212', edgecolor='#f1c40f', pad=5, alpha=0.8))

    # 3. Confronto e Ragionamento
    for i in range(3):
        idx = top_idxs[i].item()
        current_label = classes[idx]
        ax_ref = fig.add_subplot(gs[1, i])
        ax_ref.imshow(ref_imgs[idx])
        ax_ref.axis('off')
        
        rect = plt.Rectangle((0,0), 31, 31, linewidth=8, edgecolor=colors[i], facecolor='none', alpha=0.8)
        ax_ref.add_patch(rect)
        
        ax_ref.set_title(f"{current_label.upper()} ({v[i]:.1f}%)", color=colors[i], fontsize=11, fontweight='bold')
        
        # Aggiungiamo il ragionamento solo per la predizione principale
        if i == 0:
            reasoning = reasoning_map.get(current_label, "Analisi pattern complessa.")
            ax_ref.text(16, 42, f"LOGICA: {reasoning}", color='white', 
                        fontsize=9, ha='center', style='italic', wrap=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()

# --- MAIN ---
if __name__ == "__main__":
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hw = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        
        net = SimpleCNN().to(device)
        net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        
        refs = get_ref_images()
        files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not files: exit()
        
        selected_file = get_smart_file(files)
        img_p = os.path.join(test_folder, selected_file)
        
        t = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        input_t = t(Image.open(img_p).convert('RGB')).unsqueeze(0).to(device)
        
        net.eval()
        with torch.no_grad():
            out = net(input_t)
            probs = F.softmax(out, dim=1)[0] * 100
            top_p, top_i = torch.topk(probs, 3)
            pred_class = classes[top_i[0].item()]
            conf_val = top_p[0].item()

        status = log_prediction_result(selected_file, pred_class, conf_val)

        with torch.enable_grad():
            heatmap = generate_grad_cam(net, input_t, top_i[0].item())
        
        ambiguity = None
        gap = top_p[0] - top_p[1]
        if gap < 15.0:
            ambiguity = f"⚠️ AMBIGUITÀ: Diff {pred_class.upper()}/{classes[top_i[1].item()].upper()} {gap:.1f}%"

        show_master_dashboard(img_p, (pred_class, conf_val, top_p, top_i), heatmap, refs, hw, ambiguity, status)

    except Exception as e:
        print(f"⚠️ Errore: {e}")