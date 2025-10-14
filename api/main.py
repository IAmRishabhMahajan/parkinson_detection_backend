import traceback
import torch
import pandas as pd
import numpy as np
from joblib import load
from torch import nn
import io
import librosa
import soundfile as sf

try:
    from fastapi import FastAPI, Form
    from fastapi.middleware.cors import CORSMiddleware
except Exception as e:
    print("❌ FastAPI import error:", e)
    traceback.print_exc()

print("✅ Starting app import process...")

app = FastAPI()
print("✅ FastAPI app created successfully.")

from fastapi import UploadFile, HTTPException, Query, Path
from typing import Dict, List, Optional
import uuid
import csv
import os
import numpy as np
import pandas as pd
import librosa
import joblib
import io
from pathlib import Path
from datetime import datetime


def load_doctors_from_csv() -> Dict[str, List[Dict]]:
    doctors_by_postcode: Dict[str, List[Dict]] = {}
    csv_path = Path('parkinson_core_services_uk.csv')
    
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
MODEL_PATH = Path('api/model_feature_analysis/lightgbmmodel.pkl')
SCALER_PATH = Path('api/model_feature_analysis/scaler.pkl')
SESSIONS_PATH = Path('sessions.csv')

# Define the features we'll use
FEATURES = ['HNR15', 'HNR25', 'HNR35', 'HNR38', 'MFCC0', 'MFCC3', 'MFCC4', 'MFCC5', 
           'MFCC7', 'MFCC9', 'MFCC10', 'MFCC12', 'Delta0', 'Delta1', 'Delta2', 
           'Delta3', 'Delta4', 'Delta5', 'Delta7', 'Delta10', 'Delta11']

# Load model and scaler
with open(MODEL_PATH, 'rb') as f:
    model = joblib.load(f)
with open(SCALER_PATH, 'rb') as f:
    scaler = joblib.load(f)

# === Load SAINT model and preprocessor once ===
preproc = load("api/model_feature_analysis/saint_preproc.joblib")
medians = preproc["medians"]
keep_cols = preproc["keep_cols"]
ilab_scaler = preproc["scaler"]
thr = preproc["threshold"]

ckpt = torch.load("api/model_feature_analysis/saint_model.pt", map_location="cpu")
conf = ckpt["model_config"]

class SAINTLite(nn.Module):
    def __init__(self, n_features, d_model=24, n_heads=4, n_layers=1, p_drop=0.30, p_tok=0.30):
        super().__init__()
        self.scalar_proj = nn.Linear(1, d_model)
        self.col_embed   = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.token_dropout = nn.Dropout(p_tok)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=p_drop, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        B, F = x.shape
        tok = self.scalar_proj(x.view(B, F, 1)) + self.col_embed.unsqueeze(0)
        tok = self.token_dropout(tok)
        seq = torch.cat([self.cls_token.expand(B, -1, -1), tok], dim=1)
        enc = self.encoder(seq)
        cls = self.norm(enc[:, 0, :])
        return self.head(cls).squeeze(-1)

# Load weights
ILAB_model = SAINTLite(**conf)
ILAB_model.load_state_dict(ckpt["state_dict"])
ILAB_model.eval()


# Create sessions.csv if it doesn't exist
if not SESSIONS_PATH.exists():
    with open(SESSIONS_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['session_id', 'timestamp', 'result', 'filename'])

def extract_audio_features(audio_data: bytes) -> pd.DataFrame:
    """Extract MFCC, HNR and delta features from audio data and return a DataFrame
    with columns ordered according to FEATURES.
    """
    # Load audio from bytes
    y, sr = librosa.load(io.BytesIO(audio_data), sr=None)

    # Extract MFCC features (13 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Extract HNR features (approximate via harmonic component statistics)
    y_harmonic = librosa.effects.harmonic(y)
    hnr_values = []
    for freq in [15, 25, 35, 38]:  # HNR at different frequencies (as proxy)
        # use decimated harmonic signal mean as a proxy for HNR at that frame
        if len(y_harmonic) >= freq:
            hnr = float(np.mean(y_harmonic[::freq]))
        else:
            hnr = float(np.mean(y_harmonic))
        hnr_values.append(hnr)

    # Extract delta features
    deltas = librosa.feature.delta(mfccs)

    # Build feature dict
    feature_dict = {}

    # Add HNR features
    for i, freq in enumerate([15, 25, 35, 38]):
        feature_dict[f'HNR{freq}'] = hnr_values[i]

    # Add MFCC features (mean over time)
    for i in range(13):
        feature_dict[f'MFCC{i}'] = float(np.mean(mfccs[i]))

    # Add Delta features (mean over time)
    for i in range(13):
        feature_dict[f'Delta{i}'] = float(np.mean(deltas[i]))

    # Create DataFrame with columns in FEATURES order. If a FEATURES entry
    # is missing from feature_dict, fill with NaN to keep columns aligned.
    row = {f: feature_dict.get(f, np.nan) for f in FEATURES}
    df = pd.DataFrame([row], columns=FEATURES)
    return df

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

def predict_probability(df: pd.DataFrame) -> np.ndarray:
    """
    Takes a DataFrame and returns probability predictions
    using the pre-loaded SAINT model.
    """
    
    X = df[keep_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(pd.Series(medians))
    X = ilab_scaler.transform(X)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        probs = torch.sigmoid(ILAB_model(X_tensor)).numpy().flatten()
    return probs

def extract_features_librosa_from_bytes(audio_data: bytes,
                                        sr: int = 16000,
                                        n_fft: int = 1024,
                                        hop_length: int = 512) -> pd.DataFrame:
    """
    Extract audio features from raw audio bytes using librosa.
    Returns a pandas DataFrame with one row of summary statistics.
    """

    def load_wave_from_bytes(data: bytes, target_sr: int):
        with io.BytesIO(data) as f:
            y, file_sr = sf.read(f, dtype='float32')
        if file_sr != target_sr:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=target_sr)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        return y, target_sr

    def add_stats(prefix: str, values: np.ndarray, out: dict):
        """Add mean, std, min, max, median to dict for a given feature array."""
        out[f"{prefix}_mean"] = float(np.nanmean(values))
        out[f"{prefix}_std"]  = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        out[f"{prefix}_min"]  = float(np.nanmin(values))
        out[f"{prefix}_max"]  = float(np.nanmax(values))
        out[f"{prefix}_med"]  = float(np.nanmedian(values))

    # ------------------ LOAD AUDIO ------------------
    y, sr = load_wave_from_bytes(audio_data, sr)
    row = {"duration_sec": float(len(y) / sr)}

    # ------------------ BASIC FEATURES ------------------
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)[0]
    add_stats("rms", rms, row)
    add_stats("zcr", zcr, row)

    # ------------------ SPECTRAL FEATURES ------------------
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) + 1e-9
    add_stats("spec_centroid",  librosa.feature.spectral_centroid(S=S, sr=sr)[0], row)
    add_stats("spec_bandwidth", librosa.feature.spectral_bandwidth(S=S, sr=sr)[0], row)
    add_stats("spec_rolloff95", librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)[0], row)
    add_stats("spec_flatness",  librosa.feature.spectral_flatness(S=S)[0], row)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    add_stats("spec_contrast",  np.mean(contrast, axis=0), row)

    # ------------------ MFCC FEATURES ------------------
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)
    for i in range(13):
        row[f"mfcc{i+1}_mean"]   = float(np.nanmean(mfcc[i]))
        row[f"mfcc{i+1}_std"]    = float(np.nanstd(mfcc[i], ddof=1)) if mfcc.shape[1] > 1 else 0.0
        row[f"dmfcc{i+1}_mean"]  = float(np.nanmean(d1[i]))
        row[f"dmfcc{i+1}_std"]   = float(np.nanstd(d1[i], ddof=1)) if d1.shape[1] > 1 else 0.0
        row[f"ddmfcc{i+1}_mean"] = float(np.nanmean(d2[i]))
        row[f"ddmfcc{i+1}_std"]  = float(np.nanstd(d2[i], ddof=1)) if d2.shape[1] > 1 else 0.0

    # ------------------ FUNDAMENTAL FREQUENCY ------------------
    try:
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=600, sr=sr, frame_length=n_fft, hop_length=hop_length)
    except Exception:
        f0 = None

    if f0 is None or f0.size == 0:
        row.update({"f0_mean":0.0,"f0_std":0.0,"f0_med":0.0,"f0_min":0.0,"f0_max":0.0,"f0_range":0.0,"voiced_ratio":0.0})
    else:
        voiced = ~np.isnan(f0)
        f0v = f0[voiced]
        if f0v.size == 0:
            row.update({"f0_mean":0.0,"f0_std":0.0,"f0_med":0.0,"f0_min":0.0,"f0_max":0.0,"f0_range":0.0,
                        "voiced_ratio": float(np.mean(voiced)) if f0.size else 0.0})
        else:
            row["f0_mean"]   = float(np.mean(f0v))
            row["f0_std"]    = float(np.std(f0v, ddof=1)) if f0v.size > 1 else 0.0
            row["f0_med"]    = float(np.median(f0v))
            row["f0_min"]    = float(np.min(f0v))
            row["f0_max"]    = float(np.max(f0v))
            row["f0_range"]  = float(np.max(f0v) - np.min(f0v))
            row["voiced_ratio"] = float(np.mean(voiced)) if f0.size else 0.0

    # ------------------ RETURN DF ------------------
    return pd.DataFrame([row])


@app.post("/api/upload-audio")
async def upload_audio(audio_file: UploadFile, sex: Optional[str] = Form("M") ,age: Optional[int] = Form(30)):
    
    if not audio_file.filename.endswith(('.mp3', '.wav', '.m4a')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an audio file.")
    
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Read the audio file
        audio_data = await audio_file.read()
        
        # Extract features as DataFrame
        features_df = extract_audio_features(audio_data)

        # Ensure columns order matches scaler expectations
        try:
            scaled_features = scaler.transform(features_df[FEATURES])
        except Exception:
            # As a fallback, convert to numpy array with correct order
            scaled_features = scaler.transform(features_df.values)

        # Get prediction (model expects 2D array)
        # If model supports predict_proba use that, otherwise use predict
        try:
            prob = model.predict_proba(scaled_features)[0]
        
            main_result = float(prob[1])
        except Exception:
            pred = model.predict(scaled_features)[0]
            main_result = float(pred)

   
        # Get prediction (model expects 2D array)
        # If model supports predict_proba use that, otherwise use predict
        # =========================================================
# Get prediction from main (LightGBM) model
# =========================================================
    

    # =========================================================
    # Combine both models' predictions
    # =========================================================
        ilabdf = extract_features_librosa_from_bytes(audio_data)
        if sex == "M":
            sex_val = 1
        else:
            sex_val = 0
        ilabdf['Sex']= sex_val
        ilabdf['Age'] = age
        ilab_result = predict_probability(ilabdf)[0]
        if ilab_result is not None:
            final_result = 0.7 * ilab_result + 0.3 * main_result
        else:
            final_result = main_result  # fallback if Ilab model missing or failed

        # Save session information
        save_session(session_id, final_result, audio_file.filename)

        return {"session_id": session_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.get("/api/get-result/{session_id}")
async def get_result(session_id: str):
    result = get_session_result(session_id)
    
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"result": result}

# Root endpoint for health check
@app.get("/api")
async def root():
    return {"status": "healthy"}

@app.get("/api/nearby-services/{postcode}")
async def get_nearby_services(
    postcode: str ,
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