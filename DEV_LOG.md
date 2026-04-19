# 📓 Engineering Dev-Log | AI Deep Learning Lab

Questo documento costituisce il registro tecnico dell'evoluzione architetturale del laboratorio. Non è una semplice lista di modifiche, ma una documentazione delle **scelte ingegneristiche** effettuate per garantire che ogni modello di Deep Learning sia scalabile, portabile e pronto per una produzione ipotetica.

- --

## 🏗️ Core Engineering Standards (Consolidati nel Modulo 01)

*Standard obbligatori applicati per mitigare il debito tecnico e garantire la manutenibilità.*

### 1\. Architettura Disaccoppiata (Decoupled Design)

Il sistema è diviso in tre domini logici separati per minimizzare le dipendenze incrociate:

-    **Domain 01: Configuration (`config.py`)**: L'unica "fonte di verità" per iperparametri, percorsi e costanti di normalizzazione.
    
-    **Domain 02: Model Definition**: La logica dei pesi e dei layer è isolata in classi dedicate (es. `DigitNet`) per essere importata senza effetti collaterali (side-effects).
    
-    **Domain 03: Execution**: Script di Training e Inferenza che consumano i domini precedenti senza ridefinire la logica di base.
    

### 2\. Gestione dei Percorsi con Agnosticismo OS

Sostituzione integrale delle stringhe hardcoded con la libreria `pathlib`.

-    **Logica**: Utilizzo di `Path(__file__).resolve().parent.parent` per definire la `BASE_DIR`.
    
-    **Vantaggio**: Il codice calcola dinamicamente la propria posizione, funzionando istantaneamente su **Windows** (Desktop/RTX 3070), **Linux** (Server/Cloud) o **macOS**.
    

### 3\. Rigore Sintattico & Type Safety

-    **Type Hinting**: Utilizzo sistematico delle annotazioni di tipo per migliorare l'autocompletamento e la leggibilità (es. `def predict(img: Image.Image) -> int:`).
    
-    **Pylance Diagnostic**: Risoluzione del 100% dei warning di tipo "Optional" e gestione dei membri dinamici dei tensori PyTorch tramite controlli di tipo espliciti.
    
- --
### ⚠️ Sfide Risolte: Migrazione Hardware & Stabilità Ambiente

#### 1. Conflitto OpenMP (OMP Error #15)
- **Problema**: Errore critico di inizializzazione multipla della libreria `libiomp5md.dll` su Windows, causato dalla coesistenza di build diverse di PyTorch e NumPy/MKL.
- **Scelta Ingegneristica**: Rifiutata l'implementazione del fix via codice (`os.environ`) per mantenere la purezza del sorgente e l'agnosticismo rispetto all'hardware.
- **Soluzione Professionale**: Configurazione della variabile d'ambiente a livello di Conda (`conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE`). Questo garantisce che il fix sia legato all'ambiente di calcolo e non al software.

#### 2. Dependency Resolution (Protobuf & ONNX)
- **Problema**: Conflitto di versioni tra `protobuf` (richiesto da Google/TensorBoard) e `onnx` (per Netron).
- **Soluzione**: Rimosso il vincolo di versione rigido su `protobuf` nel `requirements.txt`, permettendo a `pip` di risolvere la dipendenza verso la v4.25+, necessaria per i grafi ONNX moderni.

#### 3. Validazione Portabilità Pathlib
- **Test**: Passaggio da Google Drive (Portatile) a cartella locale (Desktop).
- **Risultato**: Il sistema di mappatura dinamica dei percorsi ha gestito correttamente il cambio di volumi e permessi senza alcuna modifica manuale alle stringhe dei path.

- --

## 📂 Modulo 01: MNIST Digits - Refactoring & Optimization

### 🗓️ Analisi dello Stato Legacy

Inizialmente, il modulo presentava criticità tipiche dello sviluppo rapido:

-    Percorsi assoluti legati a unità disco specifiche (`Z:\`, `G:\`).
    
-    **Model Drift**: La classe della rete neurale veniva copiata e incollata tra i file, rendendo impossibile aggiornare l'architettura globalmente.
    
-    **Incoerenza di Pre-processing**: Parametri di normalizzazione instabili tra fase di addestramento e validazione.
    

### 🛠️ Soluzioni Ingegneristiche Implementate

| Area | File / Script | Descrizione |
|------|--------------|-------------|
| Configurazione | `src/config.py` | Controllo centralizzato di LR, Batch Size e normalizzazione. |
| Inferenza | Mirroring dell'architettura | Gli script di test importano `DigitNet`, eliminando errori di mismatch dei pesi. |
| Diagnostica | `inspect_model.py` | Analisi granulare dei tensori e dei layer senza avviare il motore di inferenza. |
| Visualizzazione | `export_to_netron.py` | Generazione di grafi computazionali in formato ONNX per l'analisi strutturale. |

#### Sincronizzazione della Normalizzazione

È stata standardizzata la pipeline di trasformazione per garantire la coerenza tra i dati di addestramento e le immagini caricate esternamente:

$$x' = \\frac{x - 0.1307}{0.3081}$$

- --

### ⚠️ Post-Mortem: Sfide Tecniche & Migrazione Hardware

Il passaggio dal **Portatile (Google Drive)** al **Desktop (RTX 3070)** ha evidenziato diverse sfide:

-    **Dependency Resolution (Protobuf)**: Risolto conflitto critico tra `onnx` e `protobuf` aggiornando il `requirements.txt` alla versione 4.25+ per supportare i grafi di modelli moderni.
    
-    **Ambiente CUDA/Torch**: Sincronizzazione delle versioni di `torch` (2.11+), `torchvision` e `torchaudio` per sfruttare l'architettura Ampere della RTX 3070, evitando errori di "Version Mismatch".
    
-    **Gestione EBUSY (Windows)**: Implementata gestione delle eccezioni durante la pulizia automatica della cartella `output/` per prevenire crash causati dal blocco dei file da parte di VS Code.
    

- --

## 📈 Roadmap Modulo 02: CIFAR-10 (Livello 2)

L'obiettivo è scalare la complessità dal bianco e nero (1 canale) al colore (3 canali RGB) in un contesto di immagini naturali.

-    **RGB Normalization**: Calcolo delle medie e deviazioni standard specifiche per i canali R, G e B di CIFAR-10.
    
-    **Complex Feature Extraction**: Evoluzione dell'architettura CNN per identificare pattern complessi (ali, ruote, musi) invece di semplici tratti grafici.
    
-    **Data Augmentation**: Implementazione di trasformazioni randomiche (`Flip`, `Rotation`) per mitigare l'overfitting su dataset di piccole dimensioni.
    

- --

**Ultimo Aggiornamento:** 20 Aprile 2026

**Hardware Target:** NVIDIA GeForce RTX 3070/4060

**Status:** Modulo 01 Verificato, Stabilizzato e Pushato.

- --