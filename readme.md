# Moroccan ID Extraction API

This project exposes a small FastAPI service capable of receiving an image of a Moroccan
national ID card and returning the parsed information as JSON. The current
implementation ships with a mocked extraction routine so that the API contract can be
tested end-to-end before the OCR pipeline is integrated.

## Project structure

```
.
├── readme.md
└── src
    └── api
        ├── __init__.py
        ├── main.py
        └── schemas.py
```

* `src/api/main.py` hosts the FastAPI application and HTTP routes.
* `src/api/schemas.py` contains the Pydantic models that drive request validation and
  the response schema.

## Getting started

1. Create a virtual environment and install the runtime dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install fastapi uvicorn
   ```

2. Start the development server with auto-reload enabled:

   ```bash
   uvicorn src.api.main:app --reload
   ```

   The API will be available at <http://127.0.0.1:8000>. An OpenAPI specification and
   interactive Swagger UI are exposed at <http://127.0.0.1:8000/docs>.

## Usage

### Health check

Verify that the service is running:

```bash
curl http://127.0.0.1:8000/health
```

### Extract ID fields

Submit a Moroccan ID card image via multipart upload. An optional `include_address`
query parameter controls whether the parsed address should be included in the
response.

```bash
curl \
  -X POST "http://127.0.0.1:8000/extract?include_address=true" \
  -F "image=@/path/to/id-card.jpg"
```

A successful request returns the structured fields in JSON:

```json
{
  "fields": {
    "cin": "AA123456",
    "full_name": "Example Citizen",
    "date_of_birth": "1990-01-01",
    "address": "123 Rue de l'Example, Casablanca"
  },
  "message": "Extraction completed successfully."
}
```

Replace the sample OCR logic inside `src/api/main.py` with your preferred document
processing pipeline to produce real extraction results.
