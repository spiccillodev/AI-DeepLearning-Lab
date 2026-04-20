import time
import csv
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# --- STANDARD LAB IMPORTS ---
from config import (
    DEVICE, CLASSES, MODEL_PATH, TEST_IMAGES_DIR, 
    LOGS_DIR, BASE_DIR, NORM_MEAN, NORM_STD, HISTORY_FILE
)
from model import CifarNet

# --- COSTANTI DI LOGICA XAI ---
REASONING_MAP = {
    'aereo': "Gradienti orizzontali netti (ali) e sfondo uniforme (cielo/pista).",
    'auto': "Riflessi metallici speculari e pattern circolari (ruote/fari).",
    'uccello': "Texture organica piumata e silhouette con appendici sottili.",
    'gatto': "Feature micro-geometriche: orecchie e gradienti radiali oculari.",
    'cervo': "Silhouette snella con arti verticali e ramificazioni superiori.",
    'cane': "Volumetria del muso prominente e texture di pelliccia irregolare.",
    'rana': "Saturazione cromatica specifica e riflessi umidi sulla pelle.",
    'cavallo': "Muscolatura lineare e asse della testa allungato.",
    'nave': "Massa scura orizzontale su texture periodica bluastra (onde).",
    'camion': "Box-volumes massicci con ripetizione di elementi rotanti."
}

def log_prediction_to_csv(filename: str, prediction: str, confidence: float):
    """Registra l'esito dell'inferenza in outputs/logs/ con encoding UTF-8."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    target_class = Path(filename).stem.lower()
    is_correct = "✅ CORRETTO" if prediction.lower() == target_class else "❌ SBAGLIATO"
    
    # Percorso centralizzato in outputs/logs/
    csv_path = LOGS_DIR / "classificazioni_log.csv"
    headers = ["Data", "File", "IA_Pred", "Target", "Confidenza", "Esito"]
    
    file_exists = csv_path.exists()
    # encoding='utf-8' fondamentale per Windows
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([
            now, filename, prediction.upper(), 
            target_class.upper(), f"{confidence:.1f}%", is_correct
        ])
    return is_correct, target_class.upper()

def get_smart_selection(file_list: list[str]) -> str:
    """Seleziona un file evitando ripetizioni immediate basandosi sulla cronologia."""    
    history = []
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = [line.strip() for line in f.readlines()]
    
    available = [f for f in file_list if f not in history]
    selected = random.choice(available if available else file_list)
    
    # Mantiene una cronologia di 3 elementi per evitare ripetizioni immediate
    history = (history + [selected])[-3:]
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        for item in history:
            f.write(f"{item}\n")
            
    return selected

def run_inference():
    """Inference Engine via Terminale."""
    try:
        # 1. Init UI
        print("\n" + "="*65)
        print(f"      🧠 CIFAR-10 TERMINAL INFERENCE 🧠")
        print(f"      Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print("="*65)

        # 2. Caricamento Modello
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Pesi non trovati in: {MODEL_PATH}")

        model = CifarNet().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        model.eval()

        # 3. Selezione Immagine
        valid_ext = {'.jpg', '.jpeg', '.png', '.JPG', '.PNG'}
        image_files = [f.name for f in TEST_IMAGES_DIR.iterdir() if f.suffix in valid_ext]
        
        if not image_files:
            raise FileNotFoundError(f"Nessuna immagine in: {TEST_IMAGES_DIR}")

        selected_file = get_smart_selection(image_files)
        img_path = TEST_IMAGES_DIR / selected_file

        # 4. Simulazione Analisi
        print(f"\n📂 File in esame: {selected_file}")
        for _ in tqdm(range(100), desc="🧬 Analisi tensoriale", ascii=False, ncols=75):
            time.sleep(0.005)

        # 5. Preprocessing
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD)
        ])

        raw_image = Image.open(img_path).convert('RGB')
        input_tensor: torch.Tensor = transform(raw_image) # type: ignore
        input_batch = input_tensor.unsqueeze(0).to(DEVICE)

        # 6. Inferenza
        with torch.no_grad():
            output = model(input_batch)
            probs = F.softmax(output, dim=1)[0] * 100
            top_p, top_i = torch.topk(probs, 3)

        # Estrazione dati con cast anti-warning
        best_label = CLASSES[int(top_i[0])]
        best_conf = float(top_p[0])

        # 7. Logging & Display
        esito, target = log_prediction_to_csv(selected_file, best_label, best_conf)

        print("\n" + "─"*65)
        print(f"🎯 RISULTATO:  {best_label.upper()} ({best_conf:.2f}%)")
        print(f"🏳️  TARGET:     {target}")
        print(f"🏁 ESITO:      {esito}")
        print(f"💭 LOGICA:     {REASONING_MAP.get(best_label, 'Analisi pattern.')}")
        print("─"*65)

        print("\n📊 DISTRIBUZIONE PROBABILITÀ (TOP 3):")
        for i in range(3):
            idx = int(top_i[i])
            prob = float(top_p[i])
            bar_len = int(prob / 2)
            bar = "█" * bar_len + "░" * (50 - bar_len)
            print(f"   {CLASSES[idx].upper():<10} |{bar}| {prob:>5.1f}%")
        print("─"*65 + "\n")
        print(f"📄 Registro aggiornato in: {LOGS_DIR / 'classificazioni_log.csv'}")

    except Exception as e:
        print(f"\n❌ ERRORE: {e}")

if __name__ == "__main__":
    run_inference()