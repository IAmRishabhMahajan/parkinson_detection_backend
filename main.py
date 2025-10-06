from fastapi import FastAPI, UploadFile, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import uuid
import csv
import os
import numpy as np
import pandas as pd
import librosa
import pickle
import io
from pathlib import Path
from datetime import datetime

app = FastAPI()

def load_doctors_from_csv() -> Dict[str, List[Dict]]:
    doctors_by_postcode: Dict[str, List[Dict]] = {}
    csv_path = FilePath('parkinson_core_services_uk.csv')
    
    if not csv_path.exists():
        print(f"Warning: CSV file not found at {csv_path}")
        return {}
        
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                postcode = row.get('postcode', '').strip()
                if postcode:
                    if postcode not in doctors_by_postcode:
                        doctors_by_postcode[postcode] = []
                    # Clean and transform the row data
                    service_info = {
                        'organization': row.get('organization', '').strip(),
                        'service_type': row.get('service_type', '').strip(),
                        'address': row.get('address', '').strip(),
                        'city': row.get('city', '').strip(),
                        'state': row.get('state', '').strip(),
                        'country': row.get('country', '').strip(),
                        'latitude': row.get('latitude', '').strip(),
                        'longitude': row.get('longitude', '').strip(),
                        'public_telephone': row.get('public_telephone', '').strip()
                    }
                    # Remove any empty fields
                    service_info = {k: v for k, v in service_info.items() if v}
                    doctors_by_postcode[postcode].append(service_info)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return {}
    
    return doctors_by_postcode

# Load doctors data from CSV
DOCTORS_DB = load_doctors_from_csv()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the ML model and scaler
MODEL_PATH = Path('model_feature_analysis/model.pkl')
SCALER_PATH = Path('model_feature_analysis/scaler.pkl')
SESSIONS_PATH = Path('sessions.csv')

# Define the features we'll use
FEATURES = ['HNR15', 'HNR25', 'HNR35', 'HNR38', 'MFCC0', 'MFCC3', 'MFCC4', 'MFCC5', 
           'MFCC7', 'MFCC9', 'MFCC10', 'MFCC12', 'Delta0', 'Delta1', 'Delta2', 
           'Delta3', 'Delta4', 'Delta5', 'Delta7', 'Delta10', 'Delta11']

# Load model and scaler
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# Create sessions.csv if it doesn't exist
if not SESSIONS_PATH.exists():
    with open(SESSIONS_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['session_id', 'timestamp', 'result', 'filename'])

def extract_audio_features(audio_data: bytes) -> np.ndarray:
    """Extract MFCC, HNR and delta features from audio data."""
    # Load audio from bytes
    y, sr = librosa.load(io.BytesIO(audio_data), sr=None)
    
    # Extract MFCC features (13 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Extract HNR features
    y_harmonic = librosa.effects.harmonic(y)
    hnr_values = []
    for freq in [15, 25, 35, 38]:  # HNR at different frequencies
        hnr = np.mean(y_harmonic[::freq])
        hnr_values.append(hnr)
    
    # Extract delta features
    deltas = librosa.feature.delta(mfccs)
    
    # Create feature vector matching the expected features
    feature_dict = {}
    
    # Add HNR features
    for i, freq in enumerate([15, 25, 35, 38]):
        feature_dict[f'HNR{freq}'] = hnr_values[i]
    
    # Add MFCC features
    for i in range(13):
        feature_dict[f'MFCC{i}'] = np.mean(mfccs[i])
    
    # Add Delta features
    for i in range(13):
        feature_dict[f'Delta{i}'] = np.mean(deltas[i])
    
    # Select only the features we need in the correct order
    features = np.array([feature_dict[f] for f in FEATURES])
    
    return features

def save_session(session_id: str, result: float, filename: str):
    """Save session information to CSV file."""
    with open(SESSIONS_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([session_id, datetime.now().isoformat(), result, filename])

def get_session_result(session_id: str) -> Optional[float]:
    """Get result for a specific session ID."""
    if not SESSIONS_PATH.exists():
        return None
        
    df = pd.read_csv(SESSIONS_PATH)
    session_data = df[df['session_id'] == session_id]
    
    if session_data.empty:
        return None
        
    return float(session_data.iloc[0]['result'])

@app.post("/upload-audio")
async def upload_audio(audio_file: UploadFile):
    if not audio_file.filename.endswith(('.mp3', '.wav', '.m4a')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an audio file.")
    
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Read the audio file
        audio_data = await audio_file.read()
        
        # Extract features
        features = extract_audio_features(audio_data)
        
        # Scale features
        scaled_features = scaler.transform(features.reshape(1, -1))
        
        # Get prediction
        result = float(model.predict_proba(scaled_features)[0][1])  # Probability of Parkinson's
        
        # Save session information
        save_session(session_id, result, audio_file.filename)
        
        return {"session_id": session_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.get("/get-result/{session_id}")
async def get_result(session_id: str):
    result = get_session_result(session_id)
    
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"result": result}

# Root endpoint for health check
@app.get("/")
async def root():
    return {"status": "healthy"}

@app.get("/nearby-services/{postcode}")
async def get_nearby_services(
    postcode: str = Path(..., description="UK postal code"),
    service_type: str | None = Query(None, description="Filter services by type (e.g., 'clinic', 'hospital')")
):
    # Check if postcode exists in our database
    if postcode not in DOCTORS_DB:
        raise HTTPException(
            status_code=404,
            detail=f"No services found for postcode {postcode}"
        )
    
    services = DOCTORS_DB[postcode]
    
    # Filter by service_type if provided
    if service_type:
        services = [
            service for service in services
            if service_type.lower() in service.get("service_type", "").lower()
        ]
    
    if not services:
        raise HTTPException(
            status_code=404,
            detail=f"No services found matching the criteria"
        )
    
    return {
        "postcode": postcode,
        "services": services,
        "total_count": len(services),
        "location": {
            "latitude": services[0].get("latitude"),
            "longitude": services[0].get("longitude")
        } if services else None
    }
