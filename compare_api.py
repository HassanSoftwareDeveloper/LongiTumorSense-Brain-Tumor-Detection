from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import torch
import json
from io import BytesIO
from inference import RealBrainTumorInferenceEngine
from preprocessing import MedicalImagePreprocessor
# from your_compare_module import compare_analysis  # Implement this if needed

app = FastAPI()

class PatientData(BaseModel):
    age: int = None
    sex: str = None
    # Add other fields as needed

@app.post("/predict")
async def predict(mri: UploadFile = File(...), patient: str = Form(...)):
    try:
        mri_bytes = await mri.read()
        # Use MedicalImagePreprocessor to support all formats
        preprocessor = MedicalImagePreprocessor()
        # Create a temporary UploadFile-like object for compatibility
        class TempUploadFile:
            def __init__(self, name, content):
                self.name = name
                self._content = content
            def read(self):
                return self._content
            def seek(self, pos):
                pass
        temp_file = TempUploadFile(mri.filename, mri_bytes)
        try:
            image, metadata = preprocessor.load_medical_image(temp_file)
        except Exception as file_err:
            return JSONResponse(content={
                "success": False,
                "error": f"Could not load MRI file. Supported formats: .npy, .dcm, .nii, .nii.gz, .png, .jpg, .jpeg, .bmp, .tiff. Details: {str(file_err)}"
            }, status_code=400)
        try:
            patient_data = json.loads(patient)
        except Exception as json_err:
            return JSONResponse(content={
                "success": False,
                "error": f"Could not parse patient JSON. Ensure patient is a valid JSON string. Details: {str(json_err)}"
            }, status_code=400)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inference_engine = RealBrainTumorInferenceEngine(device=device)
        result = inference_engine.detect_tumor_comprehensive(image, patient_data)

        # Helper to convert ndarrays to lists recursively
        def convert_ndarray(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_ndarray(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_ndarray(v) for v in obj]
            else:
                return obj

        result_serializable = convert_ndarray(result)
        metadata_serializable = convert_ndarray(metadata)
        return JSONResponse(content={"success": True, "result": result_serializable, "metadata": metadata_serializable})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return JSONResponse(content={"success": False, "error": str(e), "traceback": tb}, status_code=500)

@app.post("/compare")
async def compare(
    baseline_mri: UploadFile = File(...),
    followup_mri: UploadFile = File(...),
    patient: str = Form(...)
):
    try:
        baseline_bytes = await baseline_mri.read()
        followup_bytes = await followup_mri.read()
        baseline_image = np.load(BytesIO(baseline_bytes))
        followup_image = np.load(BytesIO(followup_bytes))
        patient_data = json.loads(patient)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inference_engine = RealBrainTumorInferenceEngine(device=device)
        # Replace with your actual progression analysis logic
        result1 = inference_engine.detect_tumor_comprehensive(baseline_image, patient_data)
        result2 = inference_engine.detect_tumor_comprehensive(followup_image, patient_data)
        progression_metrics = {}  # Fill with your logic
        return JSONResponse(content={"success": True, "baseline_result": result1, "followup_result": result2, "progression_metrics": progression_metrics})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.post("/visualize")
async def visualize(mri: UploadFile = File(...), patient: str = Form(...)):
    try:
        mri_bytes = await mri.read()
        image = np.load(BytesIO(mri_bytes))
        patient_data = json.loads(patient)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inference_engine = RealBrainTumorInferenceEngine(device=device)
        result = inference_engine.detect_tumor_comprehensive(image, patient_data)
        # Replace with your visualization logic, e.g., save PNG and return path
        visualization_path = "path/to/visualization.png"  # Implement actual logic
        return JSONResponse(content={"success": True, "visualization": visualization_path, "result": result})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.post("/report")
async def report(mri: UploadFile = File(...), patient: str = Form(...)):
    try:
        mri_bytes = await mri.read()
        image = np.load(BytesIO(mri_bytes))
        patient_data = json.loads(patient)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inference_engine = RealBrainTumorInferenceEngine(device=device)
        result = inference_engine.detect_tumor_comprehensive(image, patient_data)
        # Replace with your report generation logic, e.g., save PDF and return path
        report_path = "path/to/report.pdf"  # Implement actual logic
        return JSONResponse(content={"success": True, "report": report_path, "result": result})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

# To run: uvicorn compare_api:app --reload
