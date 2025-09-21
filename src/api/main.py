"""FastAPI application exposing an ID card data extraction endpoint."""

from __future__ import annotations

from typing import Iterable

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from .ocr import OCRExtractionError, extract_id_card_fields
from .schemas import ExtractionRequest, ExtractionResponse

app = FastAPI(
    title="Moroccan ID Extraction API",
    version="0.1.0",
    description=(
        "Upload an image of a Moroccan national ID card to receive structured "
        "data such as CIN, name, date of birth and address."
    ),
)

SUPPORTED_IMAGE_TYPES: Iterable[str] = {
    "image/jpeg",
    "image/png",
    "image/jpg",
    "image/webp",
}


def _build_request(
    include_address: bool = Query(
        True,
        description="Return the postal address detected on the ID card when true.",
    ),
) -> ExtractionRequest:
    """Dependency that constructs an :class:`ExtractionRequest` from query params."""

    return ExtractionRequest(include_address=include_address)


@app.post("/extract", response_model=ExtractionResponse, status_code=status.HTTP_200_OK)
async def extract_id_card(
    image: UploadFile = File(..., description="Image of the Moroccan ID card to parse."),
    request_data: ExtractionRequest = Depends(_build_request),
) -> ExtractionResponse:
    """Process the uploaded ID card image and return structured information."""

    if image.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Upload a JPEG, PNG or WEBP image.",
        )

    contents = await image.read()

    try:
        fields = extract_id_card_fields(
            contents,
            include_address=request_data.include_address,
        )
    except OCRExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    finally:
        await image.close()

    return ExtractionResponse(fields=fields)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> dict[str, str]:
    """Simple endpoint to verify that the API is running."""

    return {"status": "ok"}
