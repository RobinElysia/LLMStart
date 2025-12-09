# LLMStart

This project is a high-performance Python backend service built with FastAPI, 
Transformers, and FAISS, designed to seamlessly integrate remote model inference from Java backend systems.
The module provides a scalable RESTful API for efficient model inference services.

zh: [README_ZH](README_ZH.md)

## Tech Stack
- Python 3.11
- FastAPI
- Transformers
- FAISS
- Docker

## Technical Implementation
The entry point [main.py](main.py) file handles basic configuration by prompting 
the user for essential settings, such as choosing between remote API invocation or local models.(√)

## Quick Start
1. Clone this repository to your local machine.
2. Ensure that `uv` is installed in your environment, then run `uv sync`.
3. (Optional) Create a `.env` file and fill in the required parameters:
```bash
# Remote Configuration
KEY="YOUR_API_KEY"
SECRET="YOUR_API_SECRET"
URL="URL"

# Choose either Remote or Local

# Local Configuration
PATH="PATH" # Model path (specify the folder only)
```

4. Run [main.py](main.py) to start the service.

## Contribution
[RobinElysia](https://elysia.wiki:223/AboutUs.html)