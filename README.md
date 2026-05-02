# LongiTumorSense — Brain Tumor Analysis System

AI-assisted brain MRI analysis for tumor detection, WHO grading, segmentation, and longitudinal progression tracking.

> **Research use only — NOT FDA cleared / NOT CE marked.**
> All AI findings must be reviewed by a qualified radiologist before any clinical action.

---

## Features

- Brain MRI upload (PNG, JPEG, DICOM `.dcm`, NIfTI `.nii/.nii.gz`)
- Tumor detection & WHO grading (Grade I–IV Glioma)
- Multi-model segmentation ensemble (U-Net, MONAI SegResNet, SwinUNETR, SMP EfficientNet-B4)
- Real brain-tumor classifiers (Medical ViT, rad-dino, PubMedCLIP)
- Longitudinal progression comparison (RANO-informed)
- RAG-augmented analysis using TCGA clinical database
- Brain MRI validator (rejects non-MRI images — portraits, X-rays, etc.)
- Clinical PDF report generation (ACR-structured, 4 report types)
- HIPAA-compliant audit logging

---

## System Requirements

| Requirement | Minimum |
|---|---|
| Python | 3.9 or 3.10 (recommended) |
| RAM | 8 GB |
| Disk space | ~5 GB (for models + packages) |
| GPU | Not required — runs on CPU |
| Internet | Required for first-time model download |

---

## Setup (Step by Step)

### 1. Clone the repository

```bash
git clone https://github.com/HassanSoftwareDeveloper/LongiTumorSense-Brain-Tumor-Detection.git
cd LongiTumorSense-Brain-Tumor-Detection
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

> Make sure you see `(venv)` in your terminal before continuing.

### 3. Install all dependencies

```bash
pip install -r requirements.txt
```

> This installs PyTorch, MONAI, Transformers, Streamlit, and all other packages.
> May take 5–10 minutes depending on your internet speed.

### 4. Download pretrained model weights

```bash
python download_models.py
```

This downloads all required model weights automatically (free, no API key needed):

| # | Model | Size | Source |
|---|---|---|---|
| 1 | mateuszbuda U-Net | ~31 MB | PyTorch Hub |
| 2 | MONAI BraTS SegResNet | ~18 MB | MONAI Model Zoo |
| 3 | SMP EfficientNet-B4 | ~80 MB | auto-cached by SMP |
| 4 | Medical ViT (brain-tumor fine-tuned) | ~330 MB | HuggingFace |
| 5 | microsoft/rad-dino (radiology ViT) | ~330 MB | HuggingFace |

**Optional** — also download PubMedCLIP zero-shot classifier (~605 MB):
```bash
python download_models.py --full
```

**Check what is installed** at any time:
```bash
python download_models.py --check
```

> If any model fails to download, the app still runs using the available models.
> The mateuszbuda U-Net (model 1) will also auto-download on first app run if missing.

### 5. Run the app

```bash
streamlit run frontend.py
```

The app opens in your browser at **http://localhost:8501**

---

## Login Credentials

| Username | Password | Role |
|---|---|---|
| `demo` | `demo123` | Clinician (demo) |
| `hassan` | `hassan123!` | Radiologist |
| `radiology.ai` | `RadAI456!` | Radiologist |
| `brain.surgeon` | `Surgery789!` | Neurosurgeon |

---

## Project Structure

```
├── frontend.py              # Main Streamlit UI (all tabs)
├── inference.py             # AI analysis engine (ensemble v6.0)
├── models.py                # Model definitions and loaders
├── brain_validator.py       # MRI image validator
├── preprocessing.py         # Image preprocessing pipeline
├── clinical.py              # Clinical PDF report generation
├── rag_engine.py            # RAG engine (TCGA database augmentation)
├── helper.py                # Utility functions
├── config.py                # System configuration & credentials
├── audit_logger.py          # HIPAA-compliant audit logging
├── logging_constants.py     # Logging setup
├── download_models.py       # Model downloader — run once before first use
├── requirements.txt         # Python dependencies
│
├── models/                  # Model weights (downloaded by download_models.py)
│   ├── brats/
│   │   └── pytorch_hub_unet_trained.pth   # mateuszbuda U-Net weights
│   ├── monai/
│   │   └── brats_mri_segmentation/        # MONAI BraTS bundle
│   └── rad_dino/                          # microsoft/rad-dino weights
│
├── clinical_data/           # TCGA clinical CSV data (for RAG engine)
│   ├── tcga_gbm_clinical.csv
│   ├── tcga_lgg_clinical.csv
│   └── tcga_glioma_combined.csv
│
├── data/                    # Clinical patient database (JSON)
├── reports/                 # Generated PDF reports (output)
├── logs/                    # Audit and system logs
└── temp/                    # Temporary files
```

---

## How the AI Works

The system uses a **multi-model ensemble** (v6.0):

1. **Segmentation** — mateuszbuda U-Net + MONAI BraTS SegResNet + SMP EfficientNet-B4
2. **Classification** — Medical ViT + rad-dino + PubMedCLIP (real brain-tumor classifiers)
3. **Radiomics** — 14 GLCM texture features (Haralick)
4. **Fusion** — weighted ensemble with temperature scaling + evidence consolidation
5. **RAG** — TCGA clinical database augmentation for grade refinement

Model quality tiers:
- **HIGH** — 2+ real brain-tumor classifiers active
- **MEDIUM** — 1 real classifier active
- **LOW** — ImageNet fallback only (run `download_models.py` to upgrade)

---

## Troubleshooting

**`ModuleNotFoundError`**
Make sure your virtual environment is activated and run:
```bash
pip install -r requirements.txt
```

**Models not loading / 0% confidence in reports**
Run the check to see what is missing:
```bash
python download_models.py --check
```
Then re-download:
```bash
python download_models.py
```

**Streamlit not found**
```bash
pip install streamlit>=1.28.0
```

**Windows encoding errors**
The app handles this automatically. If you still see issues, run:
```bash
set PYTHONIOENCODING=utf-8
streamlit run frontend.py
```

**Port already in use**
```bash
streamlit run frontend.py --server.port 8502
```

**Out of memory**
The app runs on CPU by default. Close other applications to free RAM.
Minimum 8 GB RAM recommended.

---

## Disclaimer

This system is a **research decision-support tool**. It has NOT received FDA 510(k) clearance, CE marking, or any other regulatory approval for clinical diagnostic use. All AI findings must be independently reviewed and confirmed by a qualified, licensed radiologist before any clinical action is taken. Reported confidence values are model probabilities and do not represent calibrated clinical certainty.
