import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import time
import csv
from datetime import datetime
from tqdm import tqdm # <--- Fondamentale per la barra progressiva

# --- CONFIGURAZIONE PERCORSI ---
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "models", "modello_cifar10.pt")
test_folder = os.path.join(script_dir, "test_images")
history_file = os.path.join(script_dir, ".prediction_history.txt")
log_file = os.path.join(script_dir, "classificazioni_log.csv")

classes = ('aereo', 'auto', 'uccello', 'gatto', 'cervo', 
            'cane', 'rana', 'cavallo', 'nave', 'camion')

reasoning_map = {
    'aereo': "Gradienti orizzontali netti (ali) e sfondo uniforme (cielo).",
    'auto': "Riflessi metallici e pattern circolari ad alta densità (ruote).",
    'uccello': "Texture organiche (piume) e contrasto elevato con l'ambiente.",
    'gatto': "Orecchie triangolari e gradienti radiali oculari.",
    'cervo': "Arti sottili e possibili biforcazioni superiori (corna).",
    'cane': "Texture variabile (pelo) e volumetria del muso prominente.",
    'rana': "Saturazione cromatica specifica e riflessi cutanei umidi.",
    'cavallo': "Muscolatura lineare e asse della testa allungato.",
    'nave': "Massa orizzontale su texture periodica bluastra (onde).",
    'camion': "Volumi rettangolari massicci e ripetizione di assi meccanici."
}

# --- ARCHITETTURA ELITE (Sincronizzata) ---
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

def print_header(hw_name):
    print("\n" + "="*60)
    print(f"       🧠 CIFAR-10 NEURAL INFERENCE SYSTEM 🧠")
    print(f"       Hardware: {hw_name}")
    print("="*60)

def log_to_csv(filename, pred, conf):
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    real_class = os.path.splitext(filename)[0].lower()
    status = "✅ AZZECCATO" if pred.lower() == real_class else "❌ SBAGLIATO"
    
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Data", "File", "IA", "Target", "Conf", "Esito"])
        writer.writerow([now, filename, pred.upper(), real_class.upper(), f"{conf:.2f}%", status])
    return status, real_class.upper()

def smart_select(files):
    if os.path.exists(history_file):
        with open(history_file, 'r') as f: history = [l.strip() for l in f.readlines()]
    else: history = []
    
    available = [f for f in files if f not in history]
    selected = random.choice(available) if available else random.choice(files)
    
    history = (history + [selected])[-2:]
    with open(history_file, 'w') as f: [f.write(f"{s}\n") for s in history]
    return selected

if __name__ == "__main__":
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hw = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        print_header(hw)

        # Caricamento Modello
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not files:
            print("❌ Errore: Cartella 'test_images' vuota!")
            exit()

        selected_file = smart_select(files)
        img_path = os.path.join(test_folder, selected_file)

        # --- SIMULAZIONE ANALISI CON TASKBAR ---
        print(f"\n📂 File in esame: {selected_file}")
        for _ in tqdm(range(100), desc="🧬 Analisi tensoriale", ascii=False, ncols=75):
            time.sleep(0.01) # Simula il calcolo per rendere la barra fluida

        # Predizione effettiva
        transform = transforms.Compose([
            transforms.Resize((32, 32)), transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        img = Image.open(img_path).convert('RGB')
        input_t = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(input_t)
            probs = F.softmax(out, dim=1)[0] * 100
            top_p, top_i = torch.topk(probs, 3)

        pred_label = classes[top_i[0].item()]
        confidence = top_p[0].item()

        # Logging
        esito, target = log_to_csv(selected_file, pred_label, confidence)

        # --- OUTPUT PROFESSIONALE ---
        print("\n" + "─"*60)
        print(f"🎯 RISULTATO:  {pred_label.upper()} ({confidence:.2f}%)")
        print(f"🏳️  TARGET:     {target}")
        print(f"🏁 ESITO:      {esito}")
        print(f"💭 LOGICA:     {reasoning_map.get(pred_label)}")
        print("─"*60)

        print("\n📊 DISTRIBUZIONE PROBABILITÀ (TOP 3):")
        for i in range(3):
            c_idx = top_i[i].item()
            c_prob = top_p[i].item()
            bar_len = int(c_prob / 2)
            bar = "█" * bar_len + "░" * (50 - bar_len)
            print(f"   {classes[c_idx].upper():<10} |{bar}| {c_prob:>5.1f}%")
        print("─"*60 + "\n")

    except Exception as e:
        print(f"\n⚠️  CRITICAL ERROR: {e}")