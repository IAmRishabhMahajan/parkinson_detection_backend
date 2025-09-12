from fastapi import FastAPI, UploadFile, HTTPException, Query, Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import uuid
import random
import csv
from pathlib import Path as FilePath

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

# Store session IDs (in a real application, you might want to use a database)
sessions = set()

@app.post("/upload-audio")
async def upload_audio(audio_file: UploadFile):
    if not audio_file.filename.endswith(('.mp3', '.wav', '.m4a')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an audio file.")
    
    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    sessions.add(session_id)
    
    # In a real application, you would save the audio file and process it
    return {"session_id": session_id}

@app.get("/get-result/{session_id}")
async def get_result(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Generate a random number (this is just for demonstration)
    result = random.uniform(0, 1)
    
    return {"result": result}

# Root endpoint for health check
@app.get("/")
async def root():
    return {"status": "healthy"}

@app.get("/nearby-services/{postcode}")
async def get_nearby_services(
    postcode: str = PathParam(..., description="UK postal code"),
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
