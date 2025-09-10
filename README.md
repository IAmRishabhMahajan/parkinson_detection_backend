# Parkinson Detection Backend

A FastAPI-based backend service that provides two main endpoints:

1. `/upload-audio` (POST) - Upload an audio file and receive a session ID
2. `/get-result/{session_id}` (GET) - Get results for a given session ID

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
