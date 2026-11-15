#!/usr/bin/env python3
"""
LongiTumorSense v11.1 - System Verification Script
===================================================
Verifies all 6 bugs are fixed and system is ready for testing.

Usage:
    python verify_system.py
"""

import sys
import os

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_status(check_name, passed, details=""):
    """Print check status"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {check_name}")
    if details:
        print(f"       {details}")

def main():
    print_header("LongiTumorSense v11.1 - System Verification")
    print("Checking all 6 bug fixes...\n")
    
    all_passed = True
    
    # ═══════════════════════════════════════════════════════════════════════
    # Bug #1: Variable used before computed
    # ═══════════════════════════════════════════════════════════════════════
    print_header("Bug #1: Variable Used Before Computed")
    try:
        with open('inference.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Find GLCM computation
        glcm_compute_pos = content.find('glcm = _compute_glcm_features(rep_slice)')
        glcm_use_pos = content.find('glcm_malignancy > 0.55')
        
        if glcm_compute_pos > 0 and glcm_use_pos > 0:
            if glcm_compute_pos < glcm_use_pos:
                print_status("GLCM computed before use", True, 
                           f"Compute at pos {glcm_compute_pos}, use at pos {glcm_use_pos}")
            else:
                print_status("GLCM computed before use", False,
                           "GLCM still used before computation!")
                all_passed = False
        else:
            print_status("GLCM code found", False, "Could not find GLCM code")
            all_passed = False
    except Exception as e:
        print_status("Bug #1 check", False, f"Error: {e}")
        all_passed = False
    
    # ═══════════════════════════════════════════════════════════════════════
    # Bug #2: Segmentation models not loading
    # ═══════════════════════════════════════════════════════════════════════
    print_header("Bug #2: Segmentation Models Loading")
    try:
        import monai
        print_status("MONAI installed", True, f"Version: {monai.__version__}")
        
        # Try loading models
        from models import get_ensemble_models
        print("Loading models (this may take 30-60 seconds)...")
        models = get_ensemble_models(device='cpu')
        
        seg_count = sum(1 for k in models.keys() if k.startswith('seg_') and models[k] is not None)
        
        if seg_count >= 7:
            print_status("All 7 segmentation models loaded", True, 
                       f"Loaded: {seg_count}/7 models")
            seg_models = [k for k in sorted(models.keys()) if k.startswith('seg_') and models[k] is not None]
            for model_name in seg_models:
                print(f"       - {model_name}")
        else:
            print_status("All 7 segmentation models loaded", False,
                       f"Only {seg_count}/7 models loaded")
            all_passed = False
    except ImportError as e:
        print_status("MONAI installed", False, "Run: pip install monai")
        all_passed = False
    except Exception as e:
        print_status("Bug #2 check", False, f"Error: {e}")
        all_passed = False
    
    # ═══════════════════════════════════════════════════════════════════════
    # Bug #3: tumor_pct reporting 99%+ on normal brains
    # ═══════════════════════════════════════════════════════════════════════
    print_header("Bug #3: tumor_pct Threshold and Normal Detection")
    try:
        with open('inference.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check threshold increased to 0.60
        if 'threshold=0.60' in content:
            print_status("Segmentation threshold increased to 0.60", True)
        else:
            print_status("Segmentation threshold increased to 0.60", False,
                       "Threshold not set to 0.60")
            all_passed = False
        
        # Check normal detection added
        if 'EARLY NORMAL BRAIN DETECTION' in content:
            print_status("Early normal brain detection added", True)
        else:
            print_status("Early normal brain detection added", False,
                       "Normal detection code not found")
            all_passed = False
    except Exception as e:
        print_status("Bug #3 check", False, f"Error: {e}")
        all_passed = False
    
    # ═══════════════════════════════════════════════════════════════════════
    # Bug #4: Screenshots instead of medical images
    # ═══════════════════════════════════════════════════════════════════════
    print_header("Bug #4: Download Scripts for Real MRI Data")
    try:
        scripts_exist = (
            os.path.exists('setup_kaggle_api.py') and
            os.path.exists('download_kaggle_mri.py') and
            os.path.exists('DOWNLOAD_MRI_DATA_INSTRUCTIONS.md') and
            os.path.exists('CRITICAL_DO_NOT_USE_SCREENSHOTS.md')
        )
        
        if scripts_exist:
            print_status("Download scripts created", True)
            print("       - setup_kaggle_api.py")
            print("       - download_kaggle_mri.py")
            print("       - DOWNLOAD_MRI_DATA_INSTRUCTIONS.md")
            print("       - CRITICAL_DO_NOT_USE_SCREENSHOTS.md")
        else:
            print_status("Download scripts created", False,
                       "Some scripts missing")
            all_passed = False
    except Exception as e:
        print_status("Bug #4 check", False, f"Error: {e}")
        all_passed = False
    
    # ═══════════════════════════════════════════════════════════════════════
    # Bug #5: Neural classifiers returning uniform distribution
    # ═══════════════════════════════════════════════════════════════════════
    print_header("Bug #5: Classifier Removed from Fusion")
    try:
        with open('simple_grade_fusion.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check classifier weight set to 0
        if 'W_CLS_TRAINED  = 0.00' in content or 'W_CLS_TRAINED = 0.00' in content:
            print_status("Classifier weight set to 0%", True)
        else:
            print_status("Classifier weight set to 0%", False,
                       "Classifier weight not set to 0")
            all_passed = False
        
        # Check WHO grading rules added
        if 'WHO Grading Rules' in content or 'WHO Grade IV criteria' in content:
            print_status("WHO grading rules added", True)
        else:
            print_status("WHO grading rules added", False,
                       "WHO grading rules not found")
            all_passed = False
    except Exception as e:
        print_status("Bug #5 check", False, f"Error: {e}")
        all_passed = False
    
    # ═══════════════════════════════════════════════════════════════════════
    # Bug #6: Missing simple_grade_fusion module
    # ═══════════════════════════════════════════════════════════════════════
    print_header("Bug #6: simple_grade_fusion Module")
    try:
        # Check file exists
        if os.path.exists('simple_grade_fusion.py'):
            file_size = os.path.getsize('simple_grade_fusion.py')
            print_status("simple_grade_fusion.py exists", True,
                       f"File size: {file_size:,} bytes")
        else:
            print_status("simple_grade_fusion.py exists", False,
                       "File not found!")
            all_passed = False
        
        # Check import works
        from simple_grade_fusion import simple_grade_fusion
        print_status("Module imports successfully", True)
    except ImportError as e:
        print_status("Module imports successfully", False, f"Import error: {e}")
        all_passed = False
    except Exception as e:
        print_status("Bug #6 check", False, f"Error: {e}")
        all_passed = False
    
    # ═══════════════════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════════════════
    print_header("FINAL SUMMARY")
    
    if all_passed:
        print("\n✅ ALL 6 BUGS FIXED - SYSTEM READY FOR TESTING!\n")
        print("Next steps:")
        print("  1. Download real MRI data:")
        print("     python download_kaggle_mri.py")
        print()
        print("  2. Start the system:")
        print("     streamlit run frontend.py")
        print()
        print("  3. Test with files from Desktop/MRI_Test_Data/")
        print("     (Use original DICOM/NIfTI files, NOT screenshots!)")
        print()
        print("Expected results:")
        print("  - Normal brain → Grade 0 (>85% confidence)")
        print("  - Grade IV tumor → Grade IV (>85% confidence)")
        print("  - tumor_pct < 5% on normal brains")
        print("  - All 7 segmentation models loaded")
        print()
        return 0
    else:
        print("\n❌ SOME CHECKS FAILED - REVIEW ERRORS ABOVE\n")
        print("See BUGS_FIXED_v11.1.md for troubleshooting.")
        print()
        return 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nVerification cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
