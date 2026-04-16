# AI Deep Learning Lab

<p align="center">

<img src="https://img.shields.io/badge/Python-3.11-blue?logo=python">
<img src="https://img.shields.io/badge/ML-Deep%20Learning-orange">
<img src="https://img.shields.io/badge/Framework-PyTorch-red">

</p>

![Lab Banner](assets/img/Banner/banner_lab.png)

## Project Overview

Benvenuti nel mio laboratorio di ***Computer Vision***. Questo repository rappresenta un **ecosistema modulare** dedicato alla ricerca avanzata sull'efficienza delle <kbd>***CNN***</kbd> (_Convolutional Neural Networks_) ad alte prestazioni. Il progetto copre l'intero **ciclo di vita del modello**: dalla progettazione architetturale e analisi tensoriale, fino all'inferenza scalabile e all'implementazione di tecniche di **Hyperparameter Tuning** e un sistema di telemetria duale per il monitoraggio delle performance.

>[!TIP]
   > I benchmark dettagliati e i test sono disponibili nei Jupyter Notebook all'interno di ogni modulo.

> _Real-time Inference & Neural Activation Mapping._ ![Model](https://img.shields.io/badge/Model-CNN%20Deep%20Learning-purple)


![General Overview](assets\img\generale\classification.jpg)

> _Riferimento progettuale._ 
<img src="https://img.shields.io/badge/Scope-MNIST%20%7C%20CIFAR10%20Experiments-orange">

<!-- Immagine es_generale -->

![Esempio](assets\img\generale\es_generale.png)



   
>[!WARNING]
   >

>[!SUCCESS]
   >

>[!ERROR]
   >

##  Features 

<img src="https://img.shields.io/badge/Features-Experimental%20ML%20Lab-yellow">

   >- **<u>GUI:</u>** Visualizzazione avanzata integrata con Matplotlib (Heatmap + Top 3 + Reasoning).
   >- **<u>Logica di Ragionamento:</u>** Ogni classificazione include una spiegazione testuale dei pattern rilevati dal modello.
   >- **<u>Smart Random Inference:</u>** Algoritmo di selezione file intelligente che evita la ripetizione delle immagini di test durante le demo.




## Specifiche Hardware & Performance Stack

<img src="https://img.shields.io/badge/GPU-Optimized%20Stack-green?logo=nvidia">
<img src="https://img.shields.io/badge/Performance-High%20Throughput-blue">
<img src="https://img.shields.io/badge/Architecture-RTX%2030%20%7C%2040%20Series-success">


>[!INFO]
   >  Sistema ottimizzato per hardware di fascia alta, capace di massimizzare il throughput dei tensori sfruttando la VRAM (fino a 10× rispetto alla CPU), riducendo i tempi di training e migliorando la reattività in inferenza. Testato con successo su GPU NVIDIA serie 30 e 40, garantendo versatilità e prestazioni stabili su più generazioni.

![Deep Learning GPU Stack](assets/img/gpu_stack.png)

| Componente | Specifiche Tecniche | Note sulle Prestazioni |
|---|---|---|
| _GPU Primaria_ | NVIDIA GeForce RTX 3070 (Ampere) | Elevata densità di CUDA Cores; utilizzata come dispositivo principale per il training |
| _GPU Secondaria_ | NVIDIA GeForce RTX 4060 (Ada Lovelace) | Supporto DLSS 3, architettura efficiente per inferenza |
| _Compute Capability_ |  8.6 (RTX 3070) / 8.9 (RTX 4060) | Compatibilità cross-generation con ottimizzazioni FP32 per stabilità numerica |
| _Deep Learning Stack_ | PyTorch + CUDA 12.1 + cuDNN | Accelerazione hardware nativa con pieno supporto alle GPU NVIDIA |
| _Ambiente Sviluppo_ | Anaconda / Python 3.11 | Gestione ambienti isolati e dipendenze ottimizzata |

## 📂 Repository Structure

Il progetto è organizzato in modo modulare per garantire scalabilità, separazione dei modelli e chiarezza dell’architettura.

```
AI-DeepLearning-Lab/
├── 01-MNIST-Digits/
├── 02-CIFAR10-Classifier/
└── assets/ 
```







###### Moduli principali
- [MNIST Digits](./01-MNIST-Digits)
  <img src="https://img.shields.io/badge/Module-01%20MNIST%20Digits-blue">

- [CIFAR10 Classifier](./02-CIFAR10-Classifier) 
<img src="https://img.shields.io/badge/Module-02%20CIFAR10%20Classifier-orange">
- [assets](./assets) (Risorse grafiche per documentazione) 
  <img src="https://img.shields.io/badge/Assets-Visualization%20%7C%20Data-lightgrey">

---
### Guida all'Installazione

Segui questi passaggi per configurare l'ambiente di lavoro sul tuo PC e massimizzare le prestazioni della tua GPU.

***1. Prerequisiti Hardware & Software***

- Driver NVIDIA aggiornati.

- Anaconda o Miniconda installato.

***2. Setup dell'Ambiente***
Apri il terminale (o Anaconda Prompt) e digita:

> ***Clona la Repository:**
```bash
git clone https://github.com/tuo-username/AI-DeepLearning-Lab.git
cd AI-DeepLearning-Lab
```

>***Configura l'Ambiente Conda:***
```bash
conda create --name ai_gpu python=3.11 -y
conda activate ai_gpu
pip install -r requirements.txt
```

>[!SUCCESS]
 > Verifica l'Accelerazione GPU:
Esegui questo test rapido per confermare che PyTorch "veda" la tua RTX:

```bash
python -c "import torch; print(f'GPU Disponibile: {torch.cuda.is_available()} - {torch.cuda.get_device_name(0)}')"
```

---


---
da fare dopo implementarlo nei singoli esempi sotto cartelle e file specifici:
```
AI-DeepLearning-Lab/
│
├── 02-CIFAR10-Classifier/   # Progetto principale: classificazione immagini CIFAR-10
│
│   ├── data/                # Dataset di training e test
│
│   ├── models/              # Modelli addestrati e pesi salvati
│
│   ├── notebooks/           # Esperimenti, analisi e test in Jupyter
│
│   ├── outputs/             # Risultati generati dal modello
│   │   ├── logs/            # Log e statistiche delle predizioni
│   │   └── predictions/     # Storico delle predizioni
│
│   ├── src/                 # Codice sorgente del progetto (training e inference)
│
│   ├── test/                # Immagini personalizzate per test manuali
│
│   └── requirements.txt     # Librerie necessarie per eseguire il progetto
```
---

## Development Methodology

<img src="https://img.shields.io/badge/Methodology-Standardized%20Pipeline-blueviolet">
<img src="https://img.shields.io/badge/Step-Tensor%20Analysis-blue">
<img src="https://img.shields.io/badge/Step-Iterative%20Training-orange">
<img src="https://img.shields.io/badge/Training-Monitoring%20Stack-success">
<img src="https://img.shields.io/badge/Feature-Activation%20Mapping-purple">
<img src="https://img.shields.io/badge/Logging-Persistent%20CSV-lightgrey">


Il processo di creazione e validazione di ogni modello segue rigorosamente una pipeline standardizzata:

1. **Tensor Analysis:** Calibrazione delle dimensioni Input/Output, normalizzazione e strategie di stochastic data augmentation.

2. **Iterative Training:** Monitoraggio della convergenza tramite **TensorBoard**, utilizzando tracker di progresso <kbd>tqdm</kbd> e l'ottimizzatore *Adam*.


3. **Brain Scanning:** Estrazione e visualizzazione delle mappe di attivazione per verificare  cosa i filtri convoluzionali stiano "percependo" a diverse profondità *(feature extraction)*.

4. **Persistent Logging:** Registrazione storica di ogni predizione, punteggio di confidenza ed errore in formato CSV per analisi post-training.

5. **Model Export:** Conversione in formato <img src="https://img.shields.io/badge/ONNX-Neural%20Network%20Exchange-blue">  per la visualizzazione universale del grafo e il deployment. 

![Tensorboard curves](assets/tensorboard_curves.png)

*Esempio di monitoraggio della convergenza (Loss e Accuracy) via TensorBoard.*

## 🔬 Modelli e Ricerca Applicata

### 🔢 1. MNIST: Handwritten Digit Recognition
##### Structure Project
```
├── 01-MNIST-Digits/                # 🔢 Modulo 1: Classificazione Cifre (0-9)
│   ├── data/                       # Dataset raw (download automatico) e processati
│   ├── models/                     # Pesi addestrati (.pt)
│   ├── netron_visualization/       # Export ONNX e grafi
|   ├── notebook/                   # Jupyter Notebook con benchmark e test    
│   ├── output/                       # Log di telemetria
│   │   └── comparison_results.png  # Grafico comparativo finale (ultima sessione)
│   └── test/                # Immagini custom per inferenza manuale
```
Focus sulla scomposizione analitica della grafia in pattern geometrici puri.

- **Dataset:** 60.000 immagini di training e 10.000 di test, dimensioni 28x28 pixel in scala di grigi.
- **Architecture:** `Dual Conv2d` + `Dual MaxPool` + `Linear Classification Head`.

- **Technical Insight:** Il modello gestisce una transizione da una spatial feature map 2D a un flattened vector da **7.744 dimensioni**, ridotto poi a **128 neuroni latenti** per la decisione finale.

- **Visual Audit:** Studio approfondito dei pesi dello strato `conv.0.weight` tramite Netron.

#### Caratteristiche Principali:

***Auto-Cleanup***: La cartella runs/ viene ottimizzata automaticamente a ogni avvio per evitare accumuli di spazio.

***Hyperparameter Testing***: Il sistema testa sequenzialmente diversi Learning Rates (es. 0.01, 0.001, 0.0001) per trovare il setup ideale.

***Dual Visualization***: 1.  TensorBoard (Interactive): Analisi profonda dei gradienti e degli istogrammi.
1.  Matplotlib (Static): Generazione automatica di un grafico comparativo comparison_results.png per una consultazione rapida senza dipendenze esterne.


#### 🧠 Benchmark MNIST
<p align="center">

 













</p>

<div align="center">

<!-- Immagine es_generale
- ![AI Animation](https://media0.giphy.com/media/v1. Y2lkPTc5MGI3NjExNHYzdXI0ODZlOXFnbGJzZGhzcW5tYjk0azZ0OXJ5cXJ3bzI1YXJtNiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/nNOAPjUdo4mpZFkDf8/giphy.gif) 
-->
</div>



---

### 🖥️ Performance Hardware (*RTX 3070*) ![GPU](https://img.shields.io/badge/NVIDIA-RTX%203070%20%2F%204060-green?logo=nvidia) ![CUDA](https://img.shields.io/badge/CUDA-Accelerated-green?logo=nvidia) ![Performance](https://img.shields.io/badge/Performance-High%20Throughput-blue)

| **Parametro** | **Valore Riscontrato** | **Note Tecniche** |
|---|---|---|
| **Velocità Media** | ~**22.8 it/s** | Flusso costante senza throttling |
| **Tempo per Epoca** | ~**10.0 s** | Carico ottimale per architettura Ampere |
| **Tempo Totale (5 Epoche)** | **60.41 s** | Include overhead di validazione e logging |
| **Stato Termico** | **Stabile** | Frequenze operative tra *21.5–23.8 it/s* |

---

### 🧪 Analisi Comparativa Hyperparameters 

<img src="https://img.shields.io/badge/LR-0.0001%20Slow-yellow">
<img src="https://img.shields.io/badge/LR-0.01%20Unstable-red">
<img src="https://img.shields.io/badge/LR-0.001%20Best%20Result-brightgreen">

| **Learning Rate** | **Accuratezza Max** | **Comportamento Rilevato** | **Esito** |
|---|---|---|---|
| **0.01** | 98.27% | Aggressivo / Oscillante | ⚠️ *Instabile* |
| **0.001** | **98.89%** | Bilanciato / Convergenza rapida | ✅ *Vincitore* |
| **0.0001** | 97.96% | Conservativo / Lento | ⏳ *Incompleto* |

---

### 📈 Metriche Sessione Finale (*Best Model*) 

<img src="https://img.shields.io/badge/Best%20Model-99.41%25%20Accuracy-brightgreen">

<img src="https://img.shields.io/badge/Validation-98.89%25-blue">
<img src="https://img.shields.io/badge/Loss-Final%200.0115-success">
<img src="https://img.shields.io/badge/Best%20Epoch-Epoch%205-gold">

</p>

| **Epoca** | **Train Accuracy** | **Validation Accuracy** | **Loss (Final)** |
|---|---|---|---|
| 1 | 94.09% | 98.25% | 0.0494 |
| 2 | 98.53% | 98.69% | 0.0184 |
| 3 | 98.92% | 98.81% | 0.0302 |
| 4 | 99.21% | 98.69% | 0.0368 |
| 5 | **99.41%** | **98.89%** | **0.0115** |

---

### 🛡️ Verifica Integrità Modello 
![Status](https://img.shields.io/badge/Status-Stable-success)
![Training](https://img.shields.io/badge/Training-Stable-success)
![Overfitting](https://img.shields.io/badge/Overfitting-Minimal-brightgreen)
<p align="center">

| **Check** | **Risultato** | **Stato** |
|---|---|---|
| **Delta Train/Val** | < **1.0%** | 🟢 *Eccellente* |
| **Rischio Overfitting** | Minimo | 🟢 *Sotto controllo* |
| **Livello Performance** | *Top Tier CNN* | 🟢 *Human-level performance* |

---

Il modello dimostra:

- ✔ **Elevata stabilità di training**
- ✔ **Generalizzazione robusta**
- ✔ **Prestazioni consistenti su GPU NVIDIA**
- ✔ **Ottimizzazione efficace degli hyperparameters**

---

><mark>
> Sotto un esempio di inferenza a bassa latenza con generazione dinamica di heatmap.
 </mark>

![MNIST Demo](assets\img\MNIST\demo.png)
![Inference Demo](assets\img\MNIST\3.gif)
><mark>
> Inference Demo:
> Riconoscimento cifre e analisi dei gradienti in tempo reale.
 </mark>

### 2. CIFAR-10: Elite Object Recognition


Sfida di riconoscimento su 10 classi (RGB 32x32), potenziata da tecniche di **Interpretability**.

- **XAI (Explainable AI):** Implementazione di **Grad-CAM** per generare heatmap dinamiche che evidenziano le aree decisionali.

- **Ambiguity Detection:** Algoritmo per il calcolo del *Confidence Gap* tra classi visivamente simili (es. Auto vs Camion).

##### 📊 Benchmark CIFAR-10 
| Training Loss | Validation Accuracy | Stato |
|---|---|---|
| Risultato Corrente | 0.682 | 78.5% | 📈 In Ottimizzazione 

![Grad-CAM Demo](assets/gradcam_cifar.gif)

*Analisi Grad-CAM: Il modello distingue tra Auto e Camion analizzando feature specifiche.*

## 🚀 Roadmap: Visione Futura

- \[ \] **ResNet-50 Integration:** Implementazione di Skip-Connections per superare il limite delle 50 epoche di training senza degradazione.

- \[ \] **Advanced Data Augmentation:** Introduzione di Color Jittering e Random Flipping per aumentare la robustezza dei modelli.

- \[ \] **Real-time Camera Lab:** Overlay delle heatmap Grad-CAM in tempo reale direttamente via webcam.

**Sviluppato con dedizione da Spiccillo**