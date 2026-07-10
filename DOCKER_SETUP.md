# Docker Setup Guide

## Prerequisites
- Docker Desktop installed
- Docker Compose installed (usually comes with Docker Desktop)
- GEMINI_API_KEY ready (from https://aistudio.google.com/app/apikey)

## Quick Start

### 1. Setup Environment
```bash
# Copy the example env file and fill in your API key
cp .env.example .env

# Edit .env and add your GEMINI_API_KEY
nano .env
```

### 2. Build and Run
```bash
# Build and start both services with a single command
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

### 3. Access Applications
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000

### 4. Stop Services
```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Useful Docker Compose Commands

```bash
# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f backend
docker-compose logs -f frontend

# Restart services
docker-compose restart

# Rebuild images
docker-compose build --no-cache

# Run commands inside container
docker-compose exec backend bash
docker-compose exec frontend bash

# Remove all containers and images
docker-compose down --rmi all
```

## Project Structure
```
.
├── docker-compose.yml          # Docker Compose configuration
├── Dockerfile.backend          # Backend Flask Dockerfile
├── Dockerfile.frontend         # Frontend Vite Dockerfile
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── web_api/                   # Flask backend
│   ├── app.py
│   ├── controllers/
│   ├── services/
│   └── data/
├── web_app/                   # Vue.js + Vite frontend
│   ├── package.json
│   ├── vite.config.js
│   ├── src/
│   └── public/
└── *.keras/*.h5              # Model files
```

## Services

### Backend (Flask)
- **Port**: 5000
- **Container Name**: dermatology-backend
- **Environment**: Production
- **Features**:
  - Image prediction
  - Medical consultation via Gemini AI
  - CORS enabled for frontend communication

### Frontend (Vue.js + Vite)
- **Port**: 3000
- **Container Name**: dermatology-frontend
- **Build**: Multi-stage build for optimized production
- **Features**:
  - SPA (Single Page Application)
  - Served with Node.js serve

## Volumes

- `uploads/`: Shared folder for temporary image uploads

## Networking

Both services are connected via a custom Docker network `dermatology-network` for internal communication.

## Troubleshooting

### Port Already in Use
If ports 3000 or 5000 are already in use, modify them in `docker-compose.yml`:
```yaml
ports:
  - "3001:3000"  # Frontend on 3001
  - "5001:5000"  # Backend on 5001
```

### API Connection Issues
The frontend needs to communicate with the backend at `http://backend:5000` inside Docker. This is already configured in the frontend's environment.

### Model Files Not Found
Ensure the following files are in the root directory:
- `best_skin_model.keras` or `best_skin_model_v2.h5`
- `class_names.txt`

### GEMINI_API_KEY Not Set
If you get API key errors, make sure you've created `.env` file with a valid `GEMINI_API_KEY`.
