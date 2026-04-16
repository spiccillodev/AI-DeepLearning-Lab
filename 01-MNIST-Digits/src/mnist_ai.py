import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import os
import shutil # Modulo per la gestione dei file e cartelle
from tqdm import tqdm
import matplotlib.pyplot as plt # Plan B: Visualizzazione nativa

# --- CONFIGURAZIONE ---
script_dir = os.path.dirname(os.path.abspath(__file__))
local_data_path = os.path.join(script_dir, "data")
runs_dir = os.path.join(script_dir, "runs")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- OTTIMIZZAZIONE CARTELLA RUNS ---
# Pulizia automatica: rimuove la cartella runs se esiste per ripartire da zero
if os.path.exists(runs_dir):
    print(f"🧹 Pulizia in corso: rimozione vecchi log in {runs_dir}...")
    shutil.rmtree(runs_dir)

os.makedirs(runs_dir, exist_ok=True)

# Caricamento Dati (una sola volta per tutti gli esperimenti)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_set = datasets.MNIST(root=local_data_path, train=True, download=True, transform=transform)
val_set = datasets.MNIST(root=local_data_path, train=False, download=True, transform=transform)

# --- DEFINIZIONE MODELLO ---
class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*11*11, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x): return self.conv(x)

# --- FUNZIONE DI TRAINING PER SINGOLO ESPERIMENTO ---
def run_experiment(lr, batch_size, epochs=3):
    experiment_name = f"LR_{lr}_BS_{batch_size}"
    log_dir = os.path.join(runs_dir, experiment_name)
    
    # Tentativo TensorBoard (opzionale se fallisce)
    try:
        writer = SummaryWriter(log_dir=log_dir)
    except:
        writer = None
    
    print(f"\n🚀 Avvio esperimento: {experiment_name}")
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    model = DigitNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'loss': [], 'acc': []}

    for epoca in range(epochs):
        model.train()
        r_loss, correct, total = 0, 0, 0
        for data, target in tqdm(train_loader, desc=f"Epoca {epoca+1}/{epochs}"):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            r_loss += loss.item()

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                out = model(data)
                v_loss += criterion(out, target).item()
                _, pred = out.max(1)
                v_total += target.size(0)
                v_correct += pred.eq(target).sum().item()

        # Salvataggio storia per grafico interno
        epoch_loss = v_loss/len(val_loader)
        epoch_acc = 100. * v_correct / v_total
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)
        
        # Telemetria TensorBoard (se funziona)
        if writer:
            writer.add_scalar('Loss/Validation', epoch_loss, epoca)
            writer.add_scalar('Accuracy/Validation', epoch_acc, epoca)
        
    if writer: writer.close()
    return history

# --- MAIN LOOP: TESTIAMO DIVERSI LEARNING RATE ---
learning_rates = [0.01, 0.001, 0.0001]
all_results = {}

for lr in learning_rates:
    history = run_experiment(lr=lr, batch_size=256)
    all_results[lr] = history

# --- 📊 VISUALIZZAZIONE ALTERNATIVA (PLAN B) ---
print("\n🎨 Generazione grafici comparativi...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

for lr, data in all_results.items():
    ax1.plot(data['loss'], label=f'LR {lr}')
    ax2.plot(data['acc'], label=f'LR {lr}')

ax1.set_title('Confronto Loss (Validation)')
ax1.set_xlabel('Epoca')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

ax2.set_title('Confronto Accuracy (Validation)')
ax2.set_xlabel('Epoca')
ax2.set_ylabel('Accuracy %')
ax2.legend()
ax2.grid(True)

# Salva il risultato come immagine
plot_path = os.path.join(runs_dir, "comparison_results.png")
plt.savefig(plot_path)
print(f"✅ Grafico salvato in: {plot_path}")

# Mostra il grafico a schermo (se il terminale lo supporta)
plt.show()

print("\n--- CLASSIFICA FINALE ---")
for lr, data in all_results.items():
    print(f"LR: {lr} -> Miglior Accuracy: {max(data['acc']):.2f}%")