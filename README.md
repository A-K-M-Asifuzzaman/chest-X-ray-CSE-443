# Pneumonia Detection from Paediatric Chest X-Rays

**CSE 443 — Problem Set 01**

A convolutional neural network built and trained from scratch in **PyTorch** to classify
paediatric chest X-rays (anterior-posterior, ages 1–5) as **NORMAL** or **PNEUMONIA**.

The whole project is one executed notebook:

**[`notebooks/chest_xray_pneumonia_cnn.ipynb`](notebooks/chest_xray_pneumonia_cnn.ipynb)**

---

## Headline result

| | Test set (618 images) |
|---|---|
| ROC-AUC | **0.9701** |
| PR-AUC | **0.9775** |
| Accuracy @ 0.50 | **90.45%** |
| Pneumonia recall @ 0.50 | **98.97%** |
| Specificity @ 0.50 | 76.19% |

The more interesting result is *not* the headline: validation specificity is 100% while test
specificity is 68–76%. That gap is analysed in section 10 of the notebook and summarised below —
it is the honest limitation of this model.

---

## Repository layout

```
.
├── data/
│   └── chest_xray/          # NOT in git (~1.3 GB) — see "Getting the data"
│       ├── train/{NORMAL,PNEUMONIA}
│       ├── val/{NORMAL,PNEUMONIA}
│       └── test/{NORMAL,PNEUMONIA}
├── notebooks/
│   └── chest_xray_pneumonia_cnn.ipynb   # full pipeline, outputs included
├── models/
│   └── pneumonia_cnn.pt                 # weights + preprocessing constants + threshold
├── results/
│   ├── figures/                         # 11 generated figures
│   ├── metrics.json
│   ├── test_metrics.csv
│   ├── test_predictions.csv             # per-image probabilities
│   └── training_history.csv             # per-epoch log
├── requirements.txt
└── README.md
```

### Getting the data

The dataset is not committed (1.3 GB). Download it and unpack so the folders sit at
`data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}` exactly as above. The notebook builds its
own image cache in `data/cache/` on first run (also untracked).

```bash
pip install -r requirements.txt
jupyter notebook notebooks/chest_xray_pneumonia_cnn.ipynb
```

Training takes about **15 minutes** on an Apple M3 Pro GPU (via the PyTorch MPS backend). The
notebook auto-selects MPS, CUDA or CPU.

---

## Approach

### 1. Integrity audit before any modelling

Rather than trusting the folder layout, I checked it. Three findings changed the methodology:

**Duplicate images.** MD5-hashing all 5,856 files found **30 duplicate pairs (32 redundant
files)**, every one of them *inside* a single split. **No image crosses a split boundary**, so the
official test set is uncontaminated and its numbers can be trusted. The redundant copies were
dropped, leaving 5,824 images.

**Patient identity repeats.** Filenames encode the patient (`person617_bacteria_2288.jpeg`,
`NORMAL2-IM-0588-0001.jpeg`). The training folder holds 5,216 images from only **2,846 patients**
— 1.83 images each, with one patient contributing **30** X-rays and 1,075 patients contributing
more than one.

> A naive random train/validation split would put the same child's ribcage on both sides and
> measure memorisation instead of generalisation. The split here is **grouped by patient** using
> `StratifiedGroupKFold`, so every patient lands entirely in train or entirely in validation.
> Verified: **zero patient overlap**.

**The validation folder is unusable.** It contains **16 images** (8 per class) — an accuracy
estimate with roughly a ±12-point error bar. It cannot support early stopping or threshold
selection. I folded those 16 images into the training pool and carved a proper **559-image**
validation set out of the combined pool.

Final splits: **4,647 train / 559 validation / 618 test**.

### 2. Preprocessing

- **180×180 greyscale.** Consolidation is a large, diffuse feature that survives downsampling.
  Source images range from 415px to 2,444px wide and a few are stored as RGB, so everything is
  converted to single-channel `L` and resized.
- **In-memory cache.** The resized dataset is only ~190 MB as `uint8`, so all images are decoded
  once (~25s) into a single array. This removes JPEG decoding and disk I/O from the training loop
  entirely, and is why `num_workers=0` is fine (it also avoids DataLoader's multiprocessing-fork
  problems on macOS).
- **Normalisation statistics computed on the training split only** (mean 0.4828, std 0.2363), so
  no test information leaks into preprocessing.

### 3. Augmentation chosen for radiographs, not for photos

| Transform | Used | Why |
|---|---|---|
| Rotation ±8°, translate 6%, scale 0.9–1.1 | yes | Real patient positioning varies this much |
| Brightness / contrast ±15% | yes | Exposure settings differ between machines and operators |
| **Horizontal flip** | **no** | A mirrored chest X-ray is anatomically impossible — the heart is on the left. Flipping teaches the network to accept images it will never see and destroys the left/right asymmetry radiologists rely on |
| Vertical flip / large rotation | no | Same reason, more so |

`RandomHorizontalFlip` appears in almost every image-classification tutorial and is wrong for this
modality. That decision is stated explicitly in the notebook.

### 4. Architecture

A compact VGG-style stack — four blocks of `Conv-BN-ReLU ×2` + max-pool, channels doubling
32 → 64 → 128 → 256 as the spatial size halves 180 → 90 → 45 → 22 → 11, then global average
pooling and a single logit.

**1,172,897 parameters.** The design decision that matters: a `Flatten` of the final 11×11×256
feature map into a 512-unit dense layer would be ~16M parameters on its own — the great majority
of the network and the part most likely to memorise 5,000 images. **Global average pooling removes
it entirely.**

- `BatchNorm` after every convolution, because X-ray exposure varies widely between images.
- Two 3×3 convolutions per block instead of one 5×5: same receptive field, fewer parameters, an
  extra non-linearity.
- `BCEWithLogitsLoss` with **`pos_weight = 0.355`** (= n_normal / n_pneumonia) to counter the
  74%-pneumonia training imbalance.
- AdamW (lr 1e-3, weight decay 1e-4), `ReduceLROnPlateau`, 35 epochs, early stopping on
  **validation AUC** — threshold-free, so the class imbalance cannot distort model selection.

Best epoch: **32** (validation AUC 0.9989). Training time: **14.7 minutes**.

### 5. Threshold selection

The 0.5 default assumes false positives and false negatives cost the same, which is false for a
screening test. The threshold is picked on **validation** and applied to test exactly once.

For the high-recall rule I take the **highest** threshold that still achieves ≥98% validation
recall. Recall falls monotonically as the threshold rises, so the qualifying thresholds form a
range starting at zero — taking the *lowest* would satisfy the target while classifying nearly
everything as pneumonia, which is a degenerate operating point.

---

## Results

Held-out test set, 618 images (231 normal / 387 pneumonia):

| Operating point | Accuracy | Precision | Recall | Specificity | F1 |
|---|---:|---:|---:|---:|---:|
| Default (0.50) | **0.9045** | 0.8744 | 0.9897 | **0.7619** | **0.9285** |
| F1-optimal (0.105) | 0.8576 | 0.8161 | **0.9974** | 0.6234 | 0.8977 |
| High-recall (0.217) | 0.8786 | 0.8406 | 0.9948 | 0.6840 | 0.9112 |

ROC-AUC **0.9701** and PR-AUC **0.9775** at every threshold (both are threshold-independent).

### Per-pathogen breakdown

| True class | n | Mean predicted prob | Recall @ 0.217 |
|---|---:|---:|---:|
| NORMAL | 231 | 0.254 | — (specificity 0.684) |
| PNEUMONIA — bacterial | 240 | 0.987 | 0.9917 (2 missed) |
| PNEUMONIA — viral | 147 | 0.980 | **1.0000** (0 missed) |

---

## Findings

**1. Recall generalises; specificity does not. This is the main result.**

| | Validation (559) | Test (618) |
|---|---:|---:|
| ROC-AUC | 0.9989 | 0.9701 |
| Pneumonia recall | 0.9813 | 0.9948 |
| **Specificity** | **1.0000** | **0.6840** |
| False alarms | **0 of 131** | **73 of 231** |

The model finds pneumonia just as reliably on test as on validation. What collapses is its
handling of *normal* images. The likely cause: my validation split is carved out of the official
**training** folder, so it shares that folder's acquisition characteristics — equipment, time
window, operators. The official test folder appears to be a separate batch. Patient-grouped
splitting removed patient leakage, but it cannot remove *batch* similarity when there is only one
batch available to split.

**Validation AUC therefore overstates real-world performance, and the test set is the only honest
estimate here.** It is also why the validation-selected threshold transfers imperfectly: at 0.217
the threshold costs nothing on validation (specificity still 100%) but 8 points of specificity on
test.

**2. The pathogen result contradicted my expectation.** Bacterial pneumonia shows dense lobar
consolidation while viral produces a subtler diffuse interstitial pattern, so viral "should" be
harder. It was not — viral was caught at **100%** (147/147), bacterial at **99.2%** (238/240). At
this resolution the distinction does not matter to the model.

**3. A third of the false alarms are confident ones.** Of 73 false positives, **33 score above
0.90 and 21 above 0.99**. The two missed pneumonia cases scored 0.11 and 0.01. This asymmetry
matters operationally: a confident false positive cannot be filtered out by raising the threshold,
because it sits at the top of the score range — it needs a human reader.

**4. Global average pooling did the heavy lifting on overfitting.** Replacing the conventional
flatten-and-dense head removed ~16M of ~17M parameters with no loss of accuracy.

**5. Grad-CAM confirms the model uses anatomy, not artefacts.** On pneumonia cases the heat sits
over the mid and lower lung fields — where consolidation appears — not on the image borders, the
`R` marker or the edges. On confidently-normal images the maps are scattered and peripheral, which
is expected rather than alarming: Grad-CAM is computed on the pneumonia logit, so on a negative
image there is no positive evidence to localise. These maps should not be over-read.

### Which threshold I would actually ship

The notebook selects **0.217** on a ≥98% validation-recall rule, which is defensible for a
*screening* tool where a missed pneumonia is far worse than a re-read.

But it is worth being explicit that on this test set the default **0.5 gives better accuracy
(90.5% vs 87.9%) and better specificity (76.2% vs 68.4%) while still catching 98.97% of
pneumonia**. If the context were triage support with a limited radiologist budget rather than pure
screening, 0.5 would be the better operating point. Both are reported so the choice is visible
rather than buried.

### Limitations

- 618 test images from a single hospital and a single age group (1–5 years). Performance on
  adults, other equipment or other populations is unknown.
- The test class ratio (62.6% pneumonia) is nothing like a real screening prevalence. In a clinic
  where pneumonia is rare, precision would fall sharply even with an unchanged model — a base-rate
  effect, not a model failure.
- The specificity gap means this model would need re-validation on any new site's data.
- A pretrained backbone (ResNet/DenseNet) would very likely score higher. The task asked for a CNN
  developed for this problem, so this is trained from scratch; transfer learning is the obvious
  next step.
- **This is a coursework model. It is not validated for clinical use and must not be used as one.**

---

## Environment

Python 3.13, PyTorch 2.9 (MPS), torchvision 0.24, scikit-learn 1.6. See
[`requirements.txt`](requirements.txt).

## Dataset

Kermany, D., Zhang, K., Goldbaum, M. (2018). *Labeled Optical Coherence Tomography (OCT) and Chest
X-Ray Images for Classification.* Mendeley Data.
Chest X-rays of paediatric patients aged 1–5 from Guangzhou Women and Children's Medical Center,
collected as part of routine clinical care.
