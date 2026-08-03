# StyleForge Repository Structure

This document provides a complete guide to the StyleForge codebase, including directory structure, data flow, model lifecycle, and troubleshooting.

---

## Directory Tree

```
StyleForge/
├── styleforge/                  # Core Python package
│   ├── __init__.py             # Package exports
│   ├── transformer.py          # Shared conv layers (used by cin.py)
│   ├── cin.py                  # Conditional Instance Norm (Dumoulin 2017)
│   ├── adain.py                # AdaIN arbitrary style (Huang 2017)
│   ├── engine.py               # Unified inference engine
│   ├── tiling.py               # 4K overlapping tile + linear-blend stitch
│   ├── vgg.py                  # VGG-16 feature extractor
│   ├── utils.py                # Image I/O, Gram matrix, normalize
│   └── train_cin.py            # Multi-style CIN training (AMP)
│
├── server/                     # FastAPI backend
│   └── main.py                 # API endpoints + StaticFiles mount
│
├── frontend/                   # Vite + React + TypeScript
│   ├── index.html              # Entry HTML
│   ├── package.json            # Node dependencies
│   ├── vite.config.ts          # Vite config with API proxy
│   ├── tsconfig.json           # TypeScript config
│   └── src/
│       ├── main.tsx            # React entry point
│       ├── App.tsx             # Main app component
│       ├── index.css           # Global styles
│       ├── types.ts            # TypeScript interfaces
│       └── components/
│           ├── UploadZone.tsx      # Drag-drop image upload
│           ├── StyleGallery.tsx    # Curated style grid
│           ├── StyleMixSlider.tsx  # γ/β interpolation controls
│           └── CompareView.tsx     # Before/after wipe slider
│
├── styles/                     # Curated style images
│   └── catalog.yaml            # Style roster + attribution
│
├── models/                     # Trained checkpoints (gitignored)
│   └── multistyle.pth          # CIN model (5 styles, ~6MB)
│
├── configs/                    # Training hyperparameters
│   └── default.yaml            # Default CIN training config
│
├── docs/                       # Documentation
│   └── training_pipeline.md    # Training pipeline reference
│
├── scripts/                    # Utility scripts
│   ├── check_data.py           # Validate training dataset integrity
│   ├── create_colab_zip.py     # Package project for Colab training
│   └── evaluate.py             # Benchmark harness → docs/benchmarks.md
│
├── tests/                      # pytest suite
│   ├── test_check_data.py      # Data scan counts, --min-valid exit codes
│   ├── test_cin.py             # CIN module + interpolation
│   ├── test_engine.py          # InferenceEngine integration
│   ├── test_models.py          # CIN model load + inference
│   ├── test_server.py          # FastAPI endpoints + validation
│   ├── test_tiling.py          # 4K tiling + blending
│   ├── test_train_cin.py       # Train/val split determinism
│   ├── test_utils.py           # Gram matrix, normalize scaling
│   └── test_vgg.py             # VGG feature layers (training regression)
│
├── .github/workflows/          # CI/CD
│   └── ci.yml                  # Lint + typecheck + pytest + frontend build
│
├── StyleForge.ipynb            # Colab training notebook
├── pyproject.toml              # Tool config (ruff, mypy, packaging)
├── requirements.txt            # All dependencies (pinned)
├── README.md                   # Project overview + quick start
└── REPO_STRUCTURE.md           # This file
```

---

## Data Flow

### Inference Pipeline

```
┌─────────────────┐
│  User Upload    │
│  (JPEG/PNG)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocess     │
│  - Resize       │
│  - Normalize    │
│  - ToTensor     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Path     │
│  Selection      │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│  CIN   │ │ AdaIN  │
│(curated│ │(custom │
│ styles)│ │ style) │
└────┬───┘ └───┬────┘
     │         │
     └────┬────┘
          │
          ▼
┌─────────────────┐
│  4K Tiling      │
│  (if >1280px)   │
│  - 512px tiles  │
│  - 64px overlap │
│  - Blend stitch │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Postprocess    │
│  - Clamp [0,255]│
│  - ToPIL        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  JPEG Output    │
│  (quality=92)   │
└─────────────────┘
```

### Training Pipeline

```
┌─────────────────┐
│  COCO Dataset   │
│  (train2017)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Style Images   │
│  (5 curated)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  train_cin.py   │
│  - DataLoader   │
│  - VGG features │
│  - Gram matrices│
│  - AMP (fp16)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CIN Model      │
│  (multistyle.pth│
│   ~6MB)         │
└─────────────────┘
```

---

## Model Lifecycle

### 1. Training Configuration

Edit `configs/default.yaml` or pass CLI args (CLI overrides the config file):

```yaml
epochs: 8
batch-size: 4
image-size: 256
style-size: 512
cuda: 1
seed: 42
content-weight: 1.0e5
style-weight: 1.0e9
lr: 1.0e-3
log-interval: 500
checkpoint-interval: 2000
val-fraction: 0.02  # held out for validation
patience: 3         # early-stopping patience
# amp: 1 is CLI-only (any default key is honored if added here)
```

### 2. Colab Training

Open `StyleForge.ipynb` in Google Colab:

1. Upload `project.zip` (styleforge/, styles/, configs/)
2. Run setup cells (downloads COCO train2017 + VGG weights)
3. Run training cell (calls `train_cin.py`)
4. Download `multistyle.pth` + loss plot

### 3. Local Inference

Place `multistyle.pth` in `models/`:

```bash
# FastAPI server loads it automatically on startup
uvicorn server.main:app

# Or load manually in Python
from styleforge.engine import InferenceEngine
engine = InferenceEngine()
engine.load_cin("cin", "models/multistyle.pth", num_styles=5)
result = engine.stylize(image, "cin", style_id=0)
```

### 4. Style Interpolation

```python
# Blend between style 0 and style 1 at 50%
result = engine.stylize(
    image, "cin",
    style_a=0,
    style_b=1,
    alpha=0.5
)
```

---

## How to Add a New Style

### Step 1: Prepare Style Image

- Public domain artwork (WikiArt, museum collections)
- Save as JPEG, 512x512px or larger
- Place in `styles/` directory

### Step 2: Update Catalog

Edit `styles/catalog.yaml`:

```yaml
styles:
  - id: 5
    name: "new_style"
    display_name: "New Style Name"
    filename: "new_style.jpg"
    artist: "Artist Name"
    year: 1900
    source: "WikiArt / Museum (public domain)"
    description: "Brief description of visual characteristics"
```

### Step 3: Retrain CIN Model

Update `configs/default.yaml` or pass `--style-images`:

```bash
python -m styleforge.train_cin cin \
    --dataset training_content \
    --style-images styles/starry_night.jpg,styles/great_wave.jpg,...,styles/new_style.jpg \
    --save-model-dir models/ \
    --save-model-name multistyle.pth \
    --cuda 1 --amp 1 --epochs 4
```

### Step 4: Restart the Server — no frontend change needed

The style list is served at runtime from `styles/catalog.yaml` via
`/api/styles` and the React gallery renders it automatically, so no
frontend edit or rebuild is required. Restart the FastAPI server to pick
up the catalog change.

---

## Troubleshooting

### "Model not found" Error

**Symptom:** Server starts but `/api/health` shows `model_loaded: false`

**Cause:** `models/multistyle.pth` doesn't exist

**Fix:** Train the CIN model or download a pre-trained checkpoint

---

### Out of Memory on 4K Images

**Symptom:** CUDA OOM or slow CPU inference on large images

**Cause:** Full-image inference exceeds memory

**Fix:** The tiling module automatically splits images >1280px into 512px tiles with 64px overlap. Ensure `tiling.py` is being used in the inference path.

---

### Frontend Not Loading

**Symptom:** FastAPI returns 404 for `/`

**Cause:** `frontend/dist/` doesn't exist

**Fix:**
```bash
cd frontend
npm install
npm run build
```

The server mounts `frontend/dist/` if it exists.

---

### Style Interpolation Produces Identical Output

**Symptom:** Changing `alpha` doesn't change the result

**Cause:** Model was trained with only 1 style, or `style_a == style_b`

**Fix:** Ensure the CIN model was trained with multiple styles and that `style_a != style_b`

---

### Tests Fail After Model Update

**Symptom:** `pytest` fails on `test_models.py`

**Cause:** Test expects `multistyle.pth` with `num_styles=5`

**Fix:** Update `tests/test_models.py` to match the new model's `num_styles` parameter

---

### Training Loss Not Decreasing

**Symptom:** Loss plateaus or increases

**Cause:** Learning rate too high, or content/style weights imbalanced

**Fix:**
- Reduce `--lr` (try 1e-4)
- Adjust `--content-weight` and `--style-weight` ratio
- Check `--limit` isn't too small — the trainer errors clearly if
  `drop_last` empties the training set (need more images than `--batch-size`)

---

### AMP Produces NaN Loss

**Symptom:** Training crashes with NaN after enabling `--amp 1`

**Cause:** fp16 instability in content loss or extreme weights; the style
loss is computed in fp32 outside autocast by design

**Fix:** Reduce `--style-weight` (try 1e9 instead of 1e10) or `--lr`, or
disable AMP with `--amp 0`

---

## Development Workflow

### Run Tests

```bash
pytest tests/ -v
```

### Lint Code

```bash
ruff check styleforge/ tests/ scripts/ server/
ruff check --fix .  # Auto-fix
```

### Type Check

```bash
mypy styleforge/ --ignore-missing-imports
```

### Run Locally (Dev Mode)

```bash
# Backend
uvicorn server.main:app --reload

# Frontend (separate terminal)
cd frontend
npm run dev  # Proxies /api to localhost:8000
```

### Build for Production

```bash
# Frontend
cd frontend
npm run build

# Backend serves frontend/dist/ automatically
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `styleforge/cin.py` | CIN model architecture (γ/β per style) |
| `styleforge/engine.py` | Unified inference API |
| `styleforge/tiling.py` | 4K tiling + linear-blend stitching |
| `styleforge/train_cin.py` | Multi-style training with AMP |
| `server/main.py` | FastAPI endpoints + static serving |
| `frontend/src/App.tsx` | Main React component |
| `styles/catalog.yaml` | Style metadata + attribution |
| `configs/default.yaml` | Training hyperparameters |
| `tests/test_models.py` | CIN model smoke tests |
| `scripts/evaluate.py` | Benchmark harness |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | (unset) | Set to `production` to disable CORS |
| `STYLES_CATALOG` | `styles/catalog.yaml` | Path to style catalog YAML |
| `CIN_MODEL_PATH` | `models/multistyle.pth` | Path to CIN checkpoint |

---

## License

MIT
