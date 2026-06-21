# Model 6 — ORACLE Mission Copilot

This package implements an LLM-powered multi-agent mission planning copilot for lunar exploration. It coordinates outputs from Model 1 (Ice Detection) through Model 5 (Rover Hazard Navigation) to synthesize complete landing, drilling, habitat, routing, and risk mitigation reports.

## Features

- **Multi-Agent Architecture**: 6 specialized agents coordinating via a shared memory whiteboard.
- **FastAPI API Layer**: JSON endpoints for planning and geospatial data queries.
- **Robust Geospatial Tools**: A* pathfinding routing and multi-criteria scoring of coordinates on 2D maps.
- **Deterministic LLM Fallback**: Simulated natural language analysis ensuring standalone execution offline.
- **Offline Client Tool**: Quick planning command-line execution client.

## Requirements

Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

## Quick Start

### Start the API Web Server
To start the copilot API server from the repository root:
```bash
python Model_6_ORACLE_Mission_Copilot/train.py
```
This runs the FastAPI server at `http://127.0.0.1:8000`.

### Run Offline Planner Query
To execute a quick local mission query:
```bash
python Model_6_ORACLE_Mission_Copilot/predict.py
```

## API Documentation

Once the server is running, visit `http://127.0.0.1:8000/docs` to interact with the OpenAPI schemas:

- `POST /mission/plan`: planning endpoint.
- `POST /mission/analyze`: map metrics endpoint.
