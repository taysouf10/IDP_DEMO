# Moroccan ID Extraction API

This project exposes a small FastAPI service capable of receiving an image of a Moroccan
national ID card and returning the parsed information as JSON. The application now
ships with an OCR-based extraction pipeline powered by Tesseract, OpenCV and Pillow to
detect the CIN, holder name, date of birth and optional postal address.

## Project structure

```
.
├── readme.md
├── requirements.txt
├── src
│   └── api
│       ├── __init__.py
│       ├── main.py
│       ├── ocr.py
│       └── schemas.py
└── tests
    ├── conftest.py
    └── test_ocr.py
```

* `src/api/main.py` hosts the FastAPI application and HTTP routes.
* `src/api/ocr.py` contains the OCR pipeline and heuristics used to parse Moroccan ID
  cards.
* `src/api/schemas.py` contains the Pydantic models that drive request validation and
  the response schema.
* `tests/` bundles unit coverage for the OCR module. The tests synthesise an
  in-memory placeholder image so no binary fixtures are tracked in the
  repository.

## Getting started

1. Install the native [Tesseract OCR](https://tesseract-ocr.github.io/tessdoc/) binary.
   On Debian/Ubuntu based systems this can be achieved with:

   ```bash
   sudo apt-get update
   sudo apt-get install tesseract-ocr libtesseract-dev
   ```

   Refer to the Tesseract documentation for installation steps on Windows or macOS.

2. Create a virtual environment and install the Python dependencies listed in
   `requirements.txt`:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3. (Optional) Install additional tooling used for tests and linting:

   ```bash
   pip install pytest
   ```

4. Start the development server with auto-reload enabled:

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
response. When the underlying OCR pipeline cannot confidently detect one of the
expected fields the endpoint responds with an HTTP 422 status code describing the
missing data.

```bash
curl \
  -X POST "http://127.0.0.1:8000/extract?include_address=true" \
  -F "image=@/path/to/id-card.jpg"
```

A successful request returns the structured fields in JSON:

```json
{
  "fields": {
    "cin": "AB123456",
    "full_name": "REDACTED PERSON",
    "date_of_birth": "1985-02-01",
    "address": "99 RUE EXEMPLE RABAT"
  },
  "message": "Extraction completed successfully."
}
```

### OCR testing with sample images

The unit tests synthesise a neutral PNG image at runtime and mock the Tesseract
bindings. This keeps the repository free from binary fixtures while still
exercising the OCR parsing logic. If you would like to experiment with your own
samples, update the tests to point at a local image or call the FastAPI endpoint
directly with a multipart upload.

Run the automated checks locally with:

```bash
pytest
```

The tests mock the Tesseract bindings so they do not require the native binary to be
installed, but the application will need a working Tesseract executable at runtime to
process real uploads.
