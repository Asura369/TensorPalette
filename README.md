# TensorPalette

**High-Fidelity Neural Style Transfer Engine**

**Language:** Python (PyTorch) | **Training:** Google Colab (T4 GPU) | **Inference:** Local (CPU)

## 1. Executive Summary

**TensorPalette** is a production-ready Generative AI application that transforms user photos into specific artistic styles in real-time.

Unlike slow optimization-based methods that take minutes per image, TensorPalette utilizes a **Fast Neural Style Transfer** architecture. We leverage a "Hybrid Workflow": high-intensity training is performed on cloud GPUs (Google Colab) to create lightweight **Transformer Networks**. These optimized models are then deployed locally, allowing for instant, offline artistic rendering on standard consumer hardware.

---

## 2. System Architecture

The system relies on a **Teacher-Student** training loop (Johnson et al. Perceptual Loss architecture).

### **A. The Transformer Network (The "Artist")**
* **Type:** ResNet-based Autoencoder.
* **Input:** Raw RGB Photograph (Any resolution).
* **Output:** Stylized Image.
* **Optimization:** Uses **Instance Normalization** to normalize contrast per image (preserving artistic textures) and **Reflection Padding** to eliminate border artifacts.

### **B. The Loss Network (The "Critic")**
* **Type:** Pre-trained **VGG-16** (Frozen).
* **Function:** Extracts feature maps to mathematically measure "Style" and "Content."
* **Loss Calculation:**
    * **Content Loss**: Euclidean distance at layer `relu2_2`.
    * **Style Loss**: Gram Matrix distance at layers `relu1_2`, `relu2_2`, `relu3_3`, `relu4_3`.

---

## 3. Application Features

The local inference engine (`app.py`) includes advanced features for end-users:
* **Real-Time Inference:** Sub-second processing on standard CPUs.
* **Style Mixing:** "Style Strength" slider allowing users to blend the original photo with the stylized output.
* **Smart Resolution:**
    * **Standard Mode:** Auto-resizes large images to 1280px for speed.
    * **High Res Mode:** Preserves original 4K+ quality for printing.

---

## 4. Directory Structure

```
TensorPalette/
├── models/                  # PRE-TRAINED MODELS
│   ├── anime.pth
│   ├── sketch.pth
│   ├── oil.pth
│   └── eastern.pth
│
├── src/                     # CORE LOGIC
│   ├── transformer.py       # Model Architecture
│   ├── vgg.py               # Loss Network
│   ├── utils.py             # Image Utilities
│   └── train.py             # Training Script
│
├── styles/                  # REFERENCE ART
│
├── app.py                   # STREAMLIT INTERFACE
├── TensorPalette.ipynb      # COLAB TRAINING NOTEBOOK
└── README.md
```

---

## 5. Getting Started (Local Inference)

### Prerequisites
- Python 3.10+
- PyTorch
- Streamlit

### Installation

1. **Clone & Setup:**
```
git clone git@github.com:Asura369/TensorPalette.git
cd TensorPalette

# Create & Activate Virtual Environment
python3 -m venv venv
source venv/bin/activate
```

2. **Install Dependencies:**
```
pip install -r requirements.txt

```

3. **Run the App:**
```
streamlit run app.py

```

---

## 6. Training (Cloud Workflow)

Training requires significant GPU power. We utilize **Google Colab (Free Tier / T4 GPU)** for the heavy lifting.

1. **Prepare Project:** Zip your `src/` and `styles/` folders to `project.zip` and upload it to your Colab session.
2. **Open Notebook:** Run `TensorPalette.ipynb`.
3. **Automatic Setup:**
    * The notebook automatically downloads the **COCO Dataset** (Validation set ~1GB).
    * It downloads the pre-trained **VGG-16 weights** (~500MB).
4. **Execute Training:**
    * Run the training cell. A typical high-fidelity run takes **~45 minutes** for 4 epochs on 5,000+ images.
5. **Export:**
    * Download the resulting `.pth` file (e.g., `oil.pth`) and the loss graph.
    * Place the `.pth` file into your local `models/` directory.


**Training Command Reference:**
*If running manually in a terminal environment:*

```
python src/train.py train \
    --dataset training_content \
    --style-image styles/oil.jpg \
    --save-model-dir models \
    --save-model-name oil \
    --cuda 1 \
    --epochs 4 \
    --limit 10000 \
    --image-size 400 \
    --style-size 512

```

---
