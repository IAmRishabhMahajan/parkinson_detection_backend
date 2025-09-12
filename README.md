# Parkinson Detection Backend

A FastAPI-based backend service that provides the following endpoints:

1. `/upload-audio` (POST) - Upload an audio file and receive a session ID
2. `/get-result/{session_id}` (GET) - Get results for a given session ID
3. `/nearby-doctors/{area_code}` (GET) - Get list of nearby doctors based on Australian area code
   - Optional query parameter: `specialty` to filter doctors by specialty
   - Example: `/nearby-doctors/2000?specialty=neurologist`

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the development server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

After running the server, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Deployment

This project is configured for deployment on Vercel. To deploy:

1. Install Vercel CLI:
```bash
npm i -g vercel
```

2. Deploy:
```bash
vercel
```
