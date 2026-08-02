# StyleForge Training Pipeline

Authoritative reference for CIN model training: data preparation, architecture,
training, validation, and evaluation. Source of truth is the code; references
are by symbol name so they survive line edits.

- **Scope:** multi-style CIN training (`styleforge/train_cin.py`).
- **Stack:** Python 3.12+, PyTorch, CUDA (CPU works with `--amp 0`).

---

## 1. Overview

```mermaid
graph TD
    A[COCO train2017<br/>~19GB / 118K imgs] --> B[check_data.py<br/>integrity scan]
    B --> C[training_content/<br/>one class dir]
    C --> D[ImageFolder + transforms<br/>256px crops, flip, x255]
    D --> E[DataLoader<br/>bs=4, 2 workers, shuffle]
    E --> F[train_cin.py loop]
    G[5 style paintings<br/>512x512 square] --> H[VGG16 features<br/>precomputed once]
    H --> I[Gram matrices<br/>per style, per STYLE_LAYERS]
    I --> F
    J[VGG16 loss net<br/>frozen, ImageNet weights] --> F
    F --> K[train/val split<br/>val-fraction 0.02]
    K --> F
    F --> L[best model = {name}.pth<br/>~6MB]
    F --> M[{name}_last.pth + cin_loss_plot.png]
    F --> N[checkpoints cin_ckpt_*.pth<br/>full trainer state]
    L --> O[InferenceEngine / server]
    L --> P[evaluate.py: latency + quality]
```

**Expected run characteristics** (per README, empirical — T4 GPU, 5 curated
styles, full COCO): ~30–45 min, loss ~1e10 → ~1e8, model ~6MB.

**Entry points**

| Entry point | Command |
|---|---|
| Colab (recommended) | `StyleForge.ipynb`, cells 0–4 |
| Local CLI | `python -m styleforge.train_cin cin ...` |
| Data scan | `python scripts/check_data.py training_content/` |
| Verification | `pytest tests/ -v`; `python scripts/evaluate.py ...` |

---

## 2. Prerequisites

- **Dataset:** COCO train2017 — `wget http://images.cocodataset.org/zips/train2017.zip`, `unzip train2017.zip -d training_content/`.
- **VGG16 weights** at `models/vgg16.pth` (repo-root resolved). If absent,
  `Vgg16` downloads the pinned torchvision weights
  (`vgg16-397923af.pth`) — vendor the file locally for offline/reproducible runs.
- **Style images:** ≥2 JPEGs (5 curated in `styles/`, listed in `styles/catalog.yaml`).

If `--cuda 1` is set but no CUDA device exists, training warns and falls back
to CPU (pass `--cuda 0` to silence).

---

## 3. Stage 1 — Data Preparation

### 3.1 Integrity scan — `scripts/check_data.py`

Scans every file, forces `img.load()` (detects corruption), rejects non-JPEG
and non-RGB images. Dry run by default; `--delete` removes flagged files.

- Exit code 0 only if valid count ≥ `--min-valid` (default 1) — a corrupt or
  empty dataset fails the scan instead of passing silently.
- The directory must exist, else exit 1.

### 3.2 Directory layout

`torchvision.datasets.ImageFolder` requires ≥1 class subdirectory:

- ✅ `training_content/train2017/*.jpg` — works (COCO is flat; the single
  class label is unused).
- ❌ Flat images under `--dataset` root — ImageFolder raises
  `"Found 0 files in subfolders"`.

### 3.3 Content transforms (`train_cin.py`)

Applied lazily per split (`_AugmentSubset`): **train** gets the stochastic
augmentation below; **val** gets a deterministic `Resize(+32) → CenterCrop →
ToTensor → ×255` (no random crop/flip), so val loss is reproducible and
comparable across epochs. All stochastic ops are seeded from the dataset
index — multi-worker loaders race on the index queue, so worker-local RNG
would otherwise be nondeterministic across runs.

Train ops, in order:

| Op | Purpose |
|---|---|
| `Resize(image_size + 32)` | Scale up so crops vary (256 → 288) |
| `RandomCrop(image_size)` (index-seeded) | Spatial augmentation, fixed input size |
| `RandomHorizontalFlip` (index-seeded) | Geometric augmentation |
| `ToTensor()` | HWC uint8 → CHW float [0,1] |
| `×255` (`_mul255`) | Scale to [0,255] — module-level fn so DataLoader can pickle it |

> ⚠ The pipeline is consistently **[0,255]**: training transform, style
> preprocessing, inference (`engine.py`), and VGG normalization
> (`normalize_batch` divides by 255 before ImageNet mean/std). Change all
> four sides together.

### 3.4 DataLoader (`train_cin.py`)

| Setting | Value | Configurable? |
|---|---|---|
| `batch_size` | 4 | `batch-size` / `--batch-size` |
| `shuffle` | True | Hardcoded |
| `num_workers` | 2 | Hardcoded |
| `pin_memory` | True | Hardcoded |
| `drop_last` | True | Hardcoded (pairs with precomputed style grams) |

`--limit N` truncates the dataset for smoke tests — never leave it on for a
real run.

### 3.5 Style image preprocessing

1. Each style painting resized to square `style-size` (512) via
   `utils.load_image` — aspect-preserving cover resize + center-crop
   (`_resize_cover_center_crop`).
2. Tensor, scaled ×255, moved to device.
3. **Precomputed once before training:** each style runs through frozen VGG16;
   Gram matrices for `STYLE_LAYERS` are cached, repeated to `batch_size` and
   indexed per batch.

---

## 4. Stage 2 — Model & Loss Architecture

### 4.1 Generator — `CINTransformer` (`styleforge/cin.py`)

Johnson-style residual network with Conditional Instance Normalization
(Dumoulin 2017). One model serves all styles; style is carried by per-style
γ/β parameters (`IN(x)·(1+γ_s) + β_s`).

| Block | Spec |
|---|---|
| `conv1` | 9×9 conv, 3→32 ch |
| `conv2`, `conv3` | 3×3 stride-2 convs, 32→64→128 |
| `res1..res5` | Residual blocks, 128 ch (conv+CIN×2 each) |
| `deconv1`, `deconv2` | upsample×2 convs, 128→64→32 |
| `deconv3` | 9×9 conv, 32→3 (output) |
| Style blending | `forward_interpolated` — linear γ/β interpolation |

Adding a style adds only 5×2×channels parameters (~6MB for 5 styles).

### 4.2 Loss network — `Vgg16` (`styleforge/vgg.py`)

Frozen ImageNet VGG16, sliced into 5 feature blocks:

| Field | VGG indices | Layer |
|---|---|---|
| `relu1_2` | 0–3 | conv1_2 |
| `relu2_2` | 4–8 | conv2_2 |
| `relu3_3` | 9–15 | conv3_3 |
| `relu4_2` | 16–20 | conv4_2 — content + style loss |
| `relu4_3` | 21–22 | conv4_3 — kept for the AdaIN path (`adain.py`) |

### 4.3 Losses (`train_cin.py`)

Layer sets are module constants `STYLE_LAYERS` / `CONTENT_LAYERS`.

| Loss | Formula | Weight |
|---|---|---|
| Content | MSE over VGG `relu2_2` + `relu4_2` (each 1.0) of output vs input | × `content-weight` (1e5) |
| Style | MSE over Gram(`relu1_2`, `relu2_2`, `relu3_3`, `relu4_2`) of output vs style | × `style-weight` (1e9) |

Helpers: `gram_matrix` normalizes by `ch·h·w` (`utils.py`); `normalize_batch`
applies /255 + ImageNet stats (`utils.py`).

---

## 5. Stage 3 — Training Execution

### 5.1 Entry point & config precedence

`python -m styleforge.train_cin cin [args...]`

**Precedence: CLI args > YAML config > built-in defaults.** YAML keys are
kebab-case, normalized to underscores; only keys in the defaults dict are
honored. `cuda`/`amp` are coerced with `bool()`. Required: `--dataset`,
`--save-model-dir`, `--style-images` (≥2, comma-separated).

### 5.2 Parameters

Defaults shown below; precedence CLI > YAML > built-in defaults. `amp` is not
present in `configs/default.yaml` but any default key is honored if added
there.

| Config key | CLI flag | Default | Meaning |
|---|---|---|---|
| `epochs` | `--epochs` | 8 | Epochs; also cosine `T_max` |
| `batch-size` | `--batch-size` | 4 | Images per batch; all share one random style |
| `image-size` | `--image-size` | 256 | Content crop size |
| `style-size` | `--style-size` | 512 | Style image square size |
| `cuda` | `--cuda` | 1 | Use CUDA |
| `amp` | `--amp` | 1 | fp16 autocast (CUDA only; not in default YAML) |
| `seed` | `--seed` | 42 | RNG seed (training + val split) |
| `content-weight` | `--content-weight` | 1.0e5 | Content loss scale |
| `style-weight` | `--style-weight` | 1.0e9 | Style loss scale |
| `lr` | `--lr` | 1.0e-3 | Adam learning rate |
| `log-interval` | `--log-interval` | 500 | Batches between loss prints |
| `checkpoint-interval` | `--checkpoint-interval` | 2000 | Batches between intra-epoch checkpoints |
| `val-fraction` | `--val-fraction` | 0.02 | Held out for validation (must be in [0, 1)) |
| `patience` | `--patience` | 3 | Early-stopping patience in epochs |
| — | `--limit` | None | Truncate dataset (smoke tests only) |
| — | `--resume` | None | Full trainer checkpoint to resume from |
| — | `--checkpoint-model-dir` | None | Checkpoint directory; None disables |
| — | `--save-model-name` | `cin_epoch_{epochs}.pth` | Final artifact name |

### 5.3 Determinism

Seeds numpy, Python `random`, torch, and CUDA with `--seed`; sets
`cudnn.deterministic = True` / `cudnn.benchmark = False` for bit-exact runs.
The val split uses a dedicated `torch.Generator` seeded from `--seed`, and
augmentation is seeded from the dataset index (§3.3) so multi-worker data
loading is reproducible run-to-run. Verified: two identical runs with the
same seed produce identical training and val losses.

### 5.4 Per-step flow

1. **Style selection** — one random style per batch (`random.randint`); ids
   broadcast to the whole batch.
2. **Forward** — `transformer(x, style_ids)` inside fp16 autocast.
3. **Content loss** — `content_loss()` inside autocast.
4. **Style loss** — `style_loss()` in **fp32 outside autocast**: Gram values
   are O(10–100) and ×1e9 exceeds fp16's max (65504). Do not move it into
   autocast.
5. **Backward** — `scaler.scale(total).backward()` → `scaler.step(optimizer)`
   → `scaler.update()`.

No gradient clipping or weight decay. AMP requires `--amp 1` **and** `--cuda 1`.

### 5.5 LR schedule & early stopping

- `CosineAnnealingLR(optimizer, T_max=epochs)`, stepped once per epoch.
- Early stopping: patience (`--patience`, default 3) on the **signal** — val
  total loss when a val set exists (default), else training total. No
  improvement below `best_loss * (1 - 0.01)` (1% relative, fixed) increments
  patience; `patience` consecutive plateaus stop training.
- ⚠ Early stop before `epochs` ends the cosine schedule mid-anneal (LR saved
  non-zero).

### 5.6 Checkpoints & artifacts

| Artifact | When | Location | Contents |
|---|---|---|---|
| Intra-epoch checkpoint | every `checkpoint-interval` batches | `--checkpoint-model-dir` | Full trainer state |
| Epoch checkpoint | end of each epoch | `--checkpoint-model-dir` | Full trainer state |
| Best model | end of training | `--save-model-dir` | `{name}.pth` — best-val (or best-train) weights |
| Final-epoch model | end of training | `--save-model-dir` | `{stem}_last.pth` |
| Loss plot | end of training | `--save-model-dir` | `cin_loss_plot.png` (incl. val-total) |

`build_checkpoint` saves model, optimizer, scheduler, epoch, best loss,
patience, best weights, history, RNG states, and a full config snapshot
(hyperparameters, dataset, style images, limit, amp/cuda) — so `--resume <ckpt>`
continues at the saved epoch + 1. Old `state_dict`-only checkpoints are
rejected with an error; resuming with different settings (including a
different `--dataset`/`--style-images`/`--seed`/`--limit`, which would change
the seeded val split) warns per mismatched setting.

---

## 6. Stage 4 — Validation

On by default (`val-fraction 0.02`):

- Deterministic per-run seeded split — no separate download; `--limit` applies
  before the split, so small smoke runs get no val set (N × 0.02 rounds to 0).
- Each epoch, after training, the val set runs under `torch.no_grad()` and
  reports `[Val]` content/style/total loss. Val inputs are fully
  deterministic: the val transform uses `CenterCrop` (no random crop/flip,
  see §3.3) and val styles use a fixed per-batch cycle
  (`batch_index % num_styles`) — so the val signal is reproducible across
  runs and early stopping is not driven by sampling noise.
- Early stopping and best-model selection use val loss; `--val-fraction 0`
  falls back to training loss.
- Loss plot includes the val-total curve.

Known gap: no fixed content×style image grid during training — loss numbers
cannot detect style bleeding or broken γ/β.

---

## 7. Stage 5 — Evaluation

### 7.1 Test suite — `tests/` (41 tests)

| File | Covers |
|---|---|
| `test_models.py` | CIN load/inference, interpolation |
| `test_vgg.py` | `relu4_2` + `relu4_3` presence (training-crash regression) |
| `test_check_data.py` | Scan counts, `--delete`, exit below `--min-valid` |
| `test_train_cin.py` | `split_dataset` determinism + disjointness |
| `test_cin.py` | CIN block shapes, style distinctness |
| `test_engine.py` | Stylize paths, interpolation, tiling dispatch |
| `test_tiling.py` | Blend weights, large-image approx-identity |
| `test_utils.py` | Gram math, normalize scaling |
| `test_server.py` | API edge cases, EXIF, async jobs |

Model-dependent tests skip with a message when `models/` weights are missing;
set `STYLEFORGE_REQUIRE_MODEL=1` to turn skips into failures (CI strict mode).

### 7.2 Benchmark — `scripts/evaluate.py`

- **Latency:** CIN inference (ms) at 512/1280 px, warmup + CUDA sync.
- **Quality** (`--images-dir <heldout>`, `--max-images`): unscaled mean
  content loss and per-style style loss over `STYLE_LAYERS` (reuses the
  training loss helpers).
- Device selection matches the engine (cuda → mps → cpu, `pick_device`).
- Writes `docs/benchmarks.md`.

### 7.3 Inference consistency — `styleforge/engine.py`

- Device: cuda → mps → cpu; model loads with `weights_only=True`.
- Content transform matches training (`ToTensor` + ×255).
- Images > `TILE_THRESHOLD` (1280px) are processed via overlapping 512px tiles
  (64px overlap) with linear blending (`styleforge/tiling.py`).

---

## 8. Verification Commands

```bash
# Data integrity (dry run first; --delete removes flagged files)
python scripts/check_data.py training_content/
python scripts/check_data.py training_content/ --min-valid 1000  # exit 1 below

# Lint + types + tests
ruff check styleforge/ tests/ scripts/ server/
mypy styleforge/ --ignore-missing-imports
pytest tests/ -v

# Smoke-training (tiny subset, CPU) — sanity before a full run
python -m styleforge.train_cin cin \
    --dataset training_content \
    --style-images styles/starry_night.jpg,styles/great_wave.jpg \
    --save-model-dir /tmp/cin-smoke \
    --cuda 0 --amp 0 --epochs 1 --limit 8 --batch-size 4

# Resume from a checkpoint (continues at saved epoch + 1)
# --epochs: original total (or more to extend); a mismatch prints a warning
python -m styleforge.train_cin cin \
    --dataset training_content \
    --style-images styles/starry_night.jpg,styles/great_wave.jpg \
    --save-model-dir /tmp/cin-smoke \
    --cuda 0 --amp 0 --epochs 4 --limit 8 --batch-size 4 \
    --resume /tmp/cin-smoke/cin_ckpt_epoch_1.pth

# Full training (T4 reference: ~30-45 min)
python -m styleforge.train_cin cin \
    --dataset training_content \
    --style-images styles/starry_night.jpg,styles/great_wave.jpg,styles/girl_pearl.jpg,styles/composition_viii.jpg,styles/water_lilies.jpg \
    --save-model-dir models/ --save-model-name multistyle.pth \
    --cuda 1 --amp 1 --epochs 8 --batch-size 4 --lr 1e-3

# Post-training checks
pytest tests/ -v
python scripts/evaluate.py --cin-model models/multistyle.pth --num-styles 5 \
    --images-dir /path/to/heldout/images   # adds the quality table
```

---

## 9. Current Limitations

- No visual validation grid during training (see §6).
- Checkpoints store a full config snapshot (incl. dataset/style paths) but not
  commit hash or dataset version.
- DataLoader worker count, pinning, and `drop_last` are hardcoded (see §3.4).
