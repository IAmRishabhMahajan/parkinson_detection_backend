from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
import random

app = FastAPI()

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
