from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import torch
import json
from io import BytesIO
from inference import RealBrainTumorInferenceEngine

app = FastAPI()

class PatientData(BaseModel):
    age: int = None
    sex: str = None
    # Add other fields as needed

@app.post("/predict")
async def predict(mri: UploadFile = File(...), patient: str = Form(...)):
    try:
        mri_bytes = await mri.read()
        # Assume .npy file for now
        image = np.load(BytesIO(mri_bytes))
        patient_data = json.loads(patient)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inference_engine = RealBrainTumorInferenceEngine(device=device)
        result = inference_engine.detect_tumor_comprehensive(image, patient_data)
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

# Add more endpoints for progression analysis, report generation, etc. as needed

# To run: uvicorn api_server:app --reload
