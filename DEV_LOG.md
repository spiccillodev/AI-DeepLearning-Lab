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
## 🟦 Modulo 02: Deep Learning Image Classification (CIFAR-10)

**Status**: 🏁 Production Ready | **Hardware Target**: NVIDIA GeForce RTX 3070 (Ampere)

- --

### 1\. 🏗️ Software Architecture & Design Patterns

#### 1.1 Decoupling & Asset Management (Elite Standard)

Il sistema è stato ingegnerizzato seguendo il principio della **Separation of Concerns (SoC)**. La struttura delle directory è stata standardizzata per garantire portabilità e scalabilità:

-    **Source Logic (`/src`)**: Contiene esclusivamente script eseguibili e definizioni di classe. Il file `config.py` funge da **Single Source of Truth (SSoT)**, centralizzando iperparametri e costanti globali.
    
-    **Physical Separation of Assets**: I dataset (`/data`) e i checkpoint del modello (`/models`) sono stati estratti dalla cartella sorgente. Questo approccio previene il gonfiamento del repository Git e rispecchia i workflow di produzione dove gli asset risiedono in storage dedicati.
    
-    **Path Agnosticism**: Implementazione integrale della libreria **`pathlib`**. A differenza del legacy `os.path`, `pathlib` tratta i percorsi come oggetti, garantendo la risoluzione automatica delle discrepanze tra i separatori di directory (Windows `\` vs Unix `/`).
    

#### 1.2 Inference State & Persistence

Per ottimizzare l'esperienza di testing, è stata introdotta una logica di **Short-term Memory Management**:

-    **`prediction_history.txt`**: File di stato persistente (locazione: `outputs/logs/`) che memorizza gli identificativi univoci degli ultimi 3 asset processati.
    
-    **Algoritmo di Selezione**: Implementazione di un filtro di esclusione dinamico che garantisce una distribuzione pseudo-casuale delle immagini di test, evitando la ridondanza visuale durante le sessioni di validazione manuale.
    

- --

### 2\. 🧠 Model Engineering: CifarNet Pipeline

#### 2.1 Mathematical Preprocessing

Il dataset CIFAR-10, composto da immagini RGB a 3 canali, richiede una normalizzazione statistica rigorosa per prevenire la saturazione delle attivazioni e stabilizzare la discesa del gradiente.

Applicazione della trasformazione lineare:

$$x\_{norm} = \\frac{x - \\mu}{\\sigma}$$

Valori applicati (Standard CIFAR-10):

- $\mu_{RGB} = [0.4914, 0.4822, 0.4465]$

- $\sigma_{RGB} = [0.2023, 0.1994, 0.2010]$

#### 2.2 Deep CNN Architecture

La **CifarNet** è stata strutturata per l'estrazione gerarchica di feature complesse:

-    **Feature Extraction Stage**: 3 Blocchi Convoluzionali profondi con kernel $3 \\times 3$, stride 1 e padding 1 per mantenere la risoluzione spaziale durante il filtraggio.
    
-    **Dimensionality Reduction**: Utilizzo di strati di `MaxPool2d(2, 2)` per il dimezzamento della risoluzione spaziale, forzando la rete ad apprendere pattern invarianti alla traslazione.
    
-    **Regularization Strategy**: Integrazione di uno strato di **Dropout (0.25)** nel blocco fully-connected per mitigare il fenomeno del *co-adaptation* dei neuroni, riducendo drasticamente il rischio di overfitting su dataset di piccola scala.
    

- --

### ⚠️ Technical Problem Solving (Fix Log)

| Issue                  | Root Cause Analysis (RCA)                                            | Resolution Method                                           |
| ---------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------- |
| OpenMP Error #15       | Conflitto di runtime causato da istanze multiple di `libiomp5md.dll` | Impostazione variabile `KMP_DUPLICATE_LIB_OK=TRUE`          |
| Charmap Encoding Crash | Incompatibilità encoding `cp1252` con caratteri Unicode              | Migrazione completa a UTF-8 (`encoding='utf-8'`)            |
| Type Inference Issue   | Limiti static analysis di Pylance su oggetti dinamici PyTorch        | Introduzione type hints + soppressione con `# type: ignore` |

- --

### 🔍 XAI: Explainable AI & Spatial Diagnostics

#### 3.1 Grad-CAM Methodology

Per validare il processo decisionale del modello, è stata implementata la tecnica **Grad-weighted Class Activation Mapping (Grad-CAM)**:

-    **Hook Mechanism**: Utilizzo di *Forward* e *Backward Hooks* per catturare le mappe di attivazione e i gradienti dell'ultimo strato convoluzionale (`conv3`).
    
-    **Global Average Pooling (GAP)**: Calcolo dell'importanza dei canali tramite la media dei gradienti retropropagati:
    
  $$
\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial Y^c}{\partial A_{ij}^k}
$$
    
-    **Heatmap Generation**: Generazione di una mappa termica spaziale tramite funzione di attivazione ReLU, sovrapposta all'immagine originale con Colormap JET (OpenCV) per evidenziare i "Region of Interest" (ROI) che hanno influenzato la classificazione.
    

#### 3.2 Master Dashboard

Lo script `predict_plot.py` genera un output composito ad alta risoluzione (150 DPI) che include:

1.  **Analisi Spaziale**: Heatmap Grad-CAM overlayed.
    
2.  **Probability Distribution**: Grafico a barre TOP-3 classi con indicatore di ambiguità (gap < 15%).
    
3.  **Visual Comparison**: Confronto diretto con campioni reali del dataset CIFAR-10 per validazione morfologica.
    

- --

### 📊 Operations & Benchmarking

-    **Logging**: Sistema di logging persistente in formato CSV con tracking di metadati temporali, confidenza e target ground-truth.
    
-    **GPU Utilization**: Ottimizzazione del trasferimento dei tensori (`.to(DEVICE)`) per minimizzare il bottleneck CPU-GPU sulla RTX 3070.
- --
**Ultimo Aggiornamento:** 20 Aprile 2026

**Hardware Target:** NVIDIA GeForce RTX 3070/4060

**Status:** Modulo 02 Verificato, Stabilizzato e Pushato.

- --