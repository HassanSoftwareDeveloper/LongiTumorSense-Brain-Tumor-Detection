#!/usr/bin/env python3
"""
download_models.py — LongiTumorSense v6.0 Model Downloader
============================================================
Run this ONCE before starting the app to pre-download all pretrained models.
Models are cached locally so subsequent runs are instant.

Usage:
    python download_models.py           # download all required models
    python download_models.py --full    # also download BiomedCLIP (~605 MB)
    python download_models.py --check   # check what is installed / cached

All downloads use FREE, publicly available pretrained weights.
No GPU, no API key, no custom training required.

Models downloaded:
  1. mateuszbuda U-Net       — real LGG brain MRI weights (Dice ~0.85, ~31 MB)
  2. MONAI BraTS SegResNet   — BraTS 2021 pretrained (Dice=0.85, ~18 MB)
  3. SMP EfficientNet-B4     — ImageNet encoder (auto-cached by SMP)
  4. Medical ViT             — brain-tumor fine-tuned ViT (HuggingFace, ~330 MB)
  5. microsoft/rad-dino      — radiology pretrained ViT (HuggingFace, ~330 MB)
  6. PubMedCLIP              — optional zero-shot classifier (~605 MB, --full only)
"""

import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Verify Python dependencies before downloading anything
# ─────────────────────────────────────────────────────────────────────────────

def check_dependencies() -> bool:
    """Check that all required packages are installed."""
    print("\n" + "=" * 60)
    print("  Checking Python dependencies...")
    print("=" * 60)

    required = {
        "torch":                       "pip install torch==2.0.1",
        "torchvision":                 "pip install torchvision==0.15.2",
        "transformers":                "pip install transformers>=4.40.0",
        "huggingface_hub":             "pip install huggingface_hub>=0.20.0",
        "monai":                       "pip install monai>=1.3.0",
        "segmentation_models_pytorch": "pip install segmentation-models-pytorch>=0.3.3",
        "cv2":                         "pip install opencv-python",
        "PIL":                         "pip install Pillow",
        "numpy":                       "pip install numpy<2",
        "streamlit":                   "pip install streamlit>=1.28.0",
    }

    missing = []
    for pkg, install_cmd in required.items():
        try:
            __import__(pkg)
            print(f"  OK  {pkg}")
        except ImportError:
            print(f"  MISSING  {pkg:<35}  →  {install_cmd}")
            missing.append(pkg)

    if missing:
        print(f"\n  {len(missing)} package(s) missing.")
        print("  Fix: pip install -r requirements.txt")
        print("  Then re-run: python download_models.py")
        return False

    print(f"\n  All dependencies OK.")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — mateuszbuda U-Net  (PRIMARY segmentation model)
# ─────────────────────────────────────────────────────────────────────────────

def download_mateuszbuda_unet() -> bool:
    """
    Download real pretrained U-Net from PyTorch Hub.
    Trained on LGG MRI Segmentation dataset (Kaggle), Dice ~0.85.
    Paper: https://arxiv.org/abs/1810.04008
    """
    print("\n[1/5] mateuszbuda U-Net  (LGG brain MRI weights, Dice ~0.85, ~31 MB)")
    save_path = ROOT / "models" / "brats" / "pytorch_hub_unet_trained.pth"

    if save_path.exists():
        mb = save_path.stat().st_size / 1024 / 1024
        print(f"  OK  Already cached ({mb:.1f} MB) → {save_path}")
        return True

    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hub_model = torch.hub.load(
                'mateuszbuda/brain-segmentation-pytorch', 'unet',
                in_channels=3, out_channels=1, init_features=32,
                pretrained=True, trust_repo=True
            )
        sd = hub_model.state_dict()

        # Remap hub keys to our MateuszbrudaUNet format
        try:
            from models import _remap_mateuszbuda_keys
            sd = _remap_mateuszbuda_keys(sd)
        except ImportError:
            pass

        # Adapt first conv from 3-channel to 1-channel
        first_key = next((k for k in sd if 'conv1.weight' in k or 'enc1' in k), None)
        if first_key and sd[first_key].shape[1] == 3:
            sd[first_key] = sd[first_key].mean(dim=1, keepdim=True)

        torch.save({
            'state_dict':    sd,
            'architecture':  'MateuszbrudaUNet',
            'source':        'PyTorch Hub — mateuszbuda/brain-segmentation-pytorch',
            'dataset':       'LGG MRI Segmentation (Kaggle)',
            'performance':   'Dice ~0.85',
            'in_channels':   3,
            'out_channels':  1,
            'init_features': 32,
            'trained':       True,
        }, str(save_path))

        mb = save_path.stat().st_size / 1024 / 1024
        print(f"  OK  Downloaded and saved ({mb:.1f} MB) → {save_path}")
        return True

    except Exception as e:
        print(f"  FAIL  {e}")
        print("  Tip: check your internet connection and try again.")
        print("  The app will auto-retry this download on first run.")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — MONAI BraTS SegResNet
# ─────────────────────────────────────────────────────────────────────────────

def download_monai_brats() -> bool:
    """
    Download MONAI BraTS MRI Segmentation bundle.
    Pretrained on BraTS 2021, Dice=0.85, ~18 MB.
    """
    print("\n[2/5] MONAI BraTS SegResNet  (BraTS 2021 pretrained, Dice=0.85, ~18 MB)")
    model_path = ROOT / "models/monai/brats_mri_segmentation/models/model.pt"

    if model_path.exists():
        mb = model_path.stat().st_size / 1024 / 1024
        print(f"  OK  Already cached ({mb:.1f} MB) → {model_path}")
        return True

    try:
        import monai
        print(f"  Downloading brats_mri_segmentation bundle (MONAI {monai.__version__})...")
        monai.bundle.download(
            "brats_mri_segmentation",
            version="0.5.4",
            bundle_dir=str(ROOT / "models/monai"),
        )
        if model_path.exists():
            mb = model_path.stat().st_size / 1024 / 1024
            print(f"  OK  Downloaded ({mb:.1f} MB) → {model_path}")
            return True
        else:
            print("  FAIL  model.pt not found after download")
            return False
    except ImportError:
        print("  FAIL  monai not installed — run: pip install monai>=1.3.0")
        return False
    except Exception as e:
        print(f"  FAIL  MONAI bundle download failed: {e}")
        print("  The app will use mateuszbuda U-Net as primary segmentation model.")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — SMP EfficientNet-B4 U-Net
# ─────────────────────────────────────────────────────────────────────────────

def download_smp_model() -> bool:
    """
    Pre-cache SMP EfficientNet-B4 ImageNet encoder weights.
    These are auto-downloaded by SMP on first use — this just pre-caches them.
    """
    print("\n[3/5] SMP EfficientNet-B4 U-Net  (ImageNet encoder, auto-cached by SMP)")
    try:
        import segmentation_models_pytorch as smp
        _ = smp.Unet(
            encoder_name='efficientnet-b4',
            encoder_weights='imagenet',
            in_channels=1,
            classes=1,
        )
        print(f"  OK  SMP {smp.__version__} — EfficientNet-B4 encoder weights cached")
        return True
    except ImportError:
        print("  FAIL  segmentation-models-pytorch not installed")
        print("  Fix: pip install segmentation-models-pytorch>=0.3.3")
        return False
    except Exception as e:
        print(f"  FAIL  {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 4 — Medical ViT  (brain-tumor fine-tuned)
# ─────────────────────────────────────────────────────────────────────────────

def download_vit_model() -> bool:
    """
    Download brain-tumor fine-tuned Vision Transformer from HuggingFace.
    Tries multiple verified public repos in order.
    """
    print("\n[4/5] Medical ViT  (brain-tumor fine-tuned, HuggingFace, ~330 MB)")

    # These are the exact same repos used in models.py MedicalViTClassifier.CANDIDATES
    candidates = [
        "DunnBC22/vit-base-patch16-224-in21k-Brain_Tumors_Image_Classification",
        "Devarshi/Brain_Tumor_Classification_using_swin",
        "Devarshi/Brain_Tumor_Classification",
    ]

    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError:
        print("  FAIL  transformers not installed — run: pip install transformers>=4.40.0")
        return False

    for repo in candidates:
        try:
            print(f"  Trying {repo} ...")
            AutoImageProcessor.from_pretrained(repo)
            m = AutoModelForImageClassification.from_pretrained(repo)
            print(f"  OK  {repo}  ({m.config.num_labels} classes)")
            return True
        except Exception as e:
            print(f"  SKIP  {repo}: {str(e)[:80]}")

    print("  FAIL  All ViT candidates failed.")
    print("  The app will use GLCM + segmentation only (still functional).")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 5 — microsoft/rad-dino  (radiology pretrained ViT)
# ─────────────────────────────────────────────────────────────────────────────

def download_radimagenet() -> bool:
    """
    Download microsoft/rad-dino — ViT pretrained on 1M+ radiology images.
    Used by RadImageNetResNet in models.py as primary radiology feature extractor.
    Falls back to ImageNet ResNet-50 if unavailable.
    """
    print("\n[5/5] microsoft/rad-dino  (radiology pretrained ViT, ~330 MB)")
    cache_dir = ROOT / "models" / "rad_dino"

    # Check if already downloaded
    if (cache_dir / "model.safetensors").exists():
        mb = (cache_dir / "model.safetensors").stat().st_size / 1024 / 1024
        print(f"  OK  Already cached ({mb:.1f} MB) → {cache_dir}")
        return True

    try:
        from huggingface_hub import snapshot_download
        cache_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="microsoft/rad-dino",
            allow_patterns=["model.safetensors", "config.json", "preprocessor_config.json"],
            local_dir=str(cache_dir),
        )
        if (cache_dir / "model.safetensors").exists():
            mb = (cache_dir / "model.safetensors").stat().st_size / 1024 / 1024
            print(f"  OK  Downloaded ({mb:.1f} MB) → {cache_dir}")
            return True
        else:
            print("  FAIL  model.safetensors not found after download")
            return False
    except ImportError:
        print("  FAIL  huggingface_hub not installed — run: pip install huggingface_hub>=0.20.0")
        return False
    except Exception as e:
        print(f"  FAIL  {e}")
        print("  The app will use ImageNet ResNet-50 as fallback (automatic).")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 6 — PubMedCLIP  (optional, ~605 MB)
# ─────────────────────────────────────────────────────────────────────────────

def download_biomedclip(skip: bool = True) -> bool:
    """
    Download PubMedCLIP zero-shot classifier.
    Optional — only downloaded with --full flag.
    Model: flaviagiammarino/pubmed-clip-vit-base-patch32
    """
    print("\n[6/6] PubMedCLIP  (flaviagiammarino/pubmed-clip-vit-base-patch32, ~605 MB)")
    if skip:
        print("  SKIPPED  (pass --full to enable)")
        return True  # not a failure — it's optional

    try:
        from transformers import CLIPProcessor, CLIPModel
        repo = "flaviagiammarino/pubmed-clip-vit-base-patch32"
        print(f"  Downloading {repo} (~605 MB) ...")
        CLIPProcessor.from_pretrained(repo)
        CLIPModel.from_pretrained(repo)
        print("  OK  PubMedCLIP downloaded and cached")
        return True
    except ImportError:
        print("  FAIL  transformers not installed — run: pip install transformers>=4.40.0")
        return False
    except Exception as e:
        print(f"  FAIL  {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CHECK — show installed libraries and cached model files
# ─────────────────────────────────────────────────────────────────────────────

def check_availability():
    print("\n" + "=" * 60)
    print("  LongiTumorSense v6.0 — Availability Check")
    print("=" * 60)

    print("\n── Python Libraries " + "─" * 38)
    libs = {
        "torch":                       lambda: __import__('torch').__version__,
        "torchvision":                 lambda: __import__('torchvision').__version__,
        "transformers":                lambda: __import__('transformers').__version__,
        "huggingface_hub":             lambda: __import__('huggingface_hub').__version__,
        "monai":                       lambda: __import__('monai').__version__,
        "segmentation_models_pytorch": lambda: __import__('segmentation_models_pytorch').__version__,
        "cv2":                         lambda: __import__('cv2').__version__,
        "PIL":                         lambda: __import__('PIL').__version__,
        "numpy":                       lambda: __import__('numpy').__version__,
        "streamlit":                   lambda: __import__('streamlit').__version__,
        "nibabel":                     lambda: __import__('nibabel').__version__,
        "pydicom":                     lambda: __import__('pydicom').__version__,
        "SimpleITK":                   lambda: __import__('SimpleITK').Version(),
        "reportlab":                   lambda: __import__('reportlab').__version__,
        "scikit-learn":                lambda: __import__('sklearn').__version__,
        "scikit-image":                lambda: __import__('skimage').__version__,
        "scipy":                       lambda: __import__('scipy').__version__,
        "pandas":                      lambda: __import__('pandas').__version__,
        "plotly":                      lambda: __import__('plotly').__version__,
    }
    for lib, get_ver in libs.items():
        try:
            v = get_ver()
            print(f"  OK      {lib:<35} {v}")
        except Exception:
            print(f"  MISSING {lib:<35} NOT INSTALLED")

    print("\n── Cached Model Files " + "─" * 36)
    model_files = {
        "mateuszbuda U-Net (primary seg)":   ROOT / "models/brats/pytorch_hub_unet_trained.pth",
        "MONAI BraTS SegResNet":             ROOT / "models/monai/brats_mri_segmentation/models/model.pt",
        "rad-dino (radiology ViT)":          ROOT / "models/rad_dino/model.safetensors",
        "MONAI SwinUNETR (optional)":        ROOT / "models/monai/swinunetr_brats.pt",
    }
    for name, path in model_files.items():
        if path.exists():
            mb = path.stat().st_size / 1024 / 1024
            print(f"  OK      {name:<40} {mb:>7.1f} MB")
        else:
            print(f"  MISSING {name:<40} not found")

    print("\n── HuggingFace Cache (ViT models) " + "─" * 24)
    hf_repos = [
        "DunnBC22/vit-base-patch16-224-in21k-Brain_Tumors_Image_Classification",
        "Devarshi/Brain_Tumor_Classification_using_swin",
        "Devarshi/Brain_Tumor_Classification",
        "microsoft/rad-dino",
        "flaviagiammarino/pubmed-clip-vit-base-patch32",
    ]
    try:
        from huggingface_hub import try_to_load_from_cache, HfFileSystem
        for repo in hf_repos:
            # Check if any file from this repo is cached
            cached = try_to_load_from_cache(repo, "config.json")
            if cached is not None:
                print(f"  OK      {repo}")
            else:
                print(f"  MISSING {repo}")
    except Exception:
        print("  (huggingface_hub not available for cache check)")

    print("\n── Live Model Load Test " + "─" * 34)
    import logging as _logging
    _logging.disable(_logging.CRITICAL)
    try:
        from models import (
            load_mateuszbuda_unet, SMPEfficientNetUNet, MONAIBraTSAdapter,
            MONAISwinUNETRWrapper, RadImageNetResNet, MedicalViTClassifier,
            BiomedCLIPClassifier,
        )
        tests = [
            ("mateuszbuda U-Net",     lambda: load_mateuszbuda_unet(device='cpu') is not None),
            ("SMP EfficientNet-B4",   lambda: SMPEfficientNetUNet(device='cpu')._loaded),
            ("MONAI BraTS SegResNet", lambda: MONAIBraTSAdapter(device='cpu')._loaded),
            ("MONAI SwinUNETR",       lambda: MONAISwinUNETRWrapper(device='cpu')._loaded),
            ("rad-dino / ResNet-50",  lambda: RadImageNetResNet(device='cpu')._loaded),
            ("Medical ViT",           lambda: MedicalViTClassifier(device='cpu')._loaded),
            ("PubMedCLIP",            lambda: BiomedCLIPClassifier(device='cpu')._loaded),
        ]
        for name, test_fn in tests:
            try:
                ok = test_fn()
                status = "OK  " if ok else "FAIL"
                print(f"  {status}    {name}")
            except Exception as e:
                print(f"  FAIL    {name}: {str(e)[:60]}")
    except ImportError as e:
        print(f"  Cannot import models.py: {e}")
    finally:
        _logging.disable(_logging.NOTSET)

    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LongiTumorSense v6.0 — Model Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--check', action='store_true',
                        help="Check installed libraries and cached models only (no download)")
    parser.add_argument('--full',  action='store_true',
                        help="Also download PubMedCLIP (~605 MB, optional)")
    args = parser.parse_args()

    if args.check:
        check_availability()
        return

    print("=" * 60)
    print("  LongiTumorSense v6.0 — Model Download")
    print("  All models are FREE and publicly available.")
    print("  No GPU or API key required.")
    print("=" * 60)

    # Step 0: verify dependencies first
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n  Please install missing packages first:")
        print("    pip install -r requirements.txt")
        print("  Then re-run: python download_models.py")
        sys.exit(1)

    # Download all models
    results = [
        ("mateuszbuda U-Net  (primary segmentation)",  download_mateuszbuda_unet()),
        ("MONAI BraTS SegResNet  (Dice=0.85)",          download_monai_brats()),
        ("SMP EfficientNet-B4 U-Net",                   download_smp_model()),
        ("Medical ViT  (brain-tumor fine-tuned)",        download_vit_model()),
        ("microsoft/rad-dino  (radiology pretrained)",   download_radimagenet()),
        ("PubMedCLIP  (optional zero-shot classifier)",  download_biomedclip(skip=not args.full)),
    ]

    print("\n" + "=" * 60)
    print("  Download Summary")
    print("=" * 60)
    n_ok = 0
    for name, ok in results:
        icon = "OK  " if ok else "FAIL"
        print(f"  {icon}  {name}")
        if ok:
            n_ok += 1

    print(f"\n  {n_ok}/{len(results)} models ready")
    print("=" * 60)

    # Minimum required: mateuszbuda U-Net (model 1) must succeed
    unet_ok = results[0][1]
    if not unet_ok:
        print("\n  WARNING: Primary segmentation model (mateuszbuda U-Net) failed to download.")
        print("  The app will attempt to auto-download it on first run.")
        print("  Check your internet connection and try again.")
    elif n_ok >= 3:
        print("\n  System is ready. Start the app with:")
        print("    streamlit run frontend.py\n")
    else:
        print("\n  Some models failed. The app will still run using available models.")
        print("  Check your internet connection and re-run this script.")
        print("  Start the app with:")
        print("    streamlit run frontend.py\n")


if __name__ == '__main__':
    main()
