"""Pydantic models used by the ID extraction API."""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class IDCardFields(BaseModel):
    """Structured data extracted from a Moroccan ID card."""

    cin: str = Field(..., description="Unique national identification number (CIN).")
    full_name: str = Field(..., description="Citizen's full name exactly as printed on the document.")
    date_of_birth: date = Field(..., description="Date of birth found on the ID card.")
    address: Optional[str] = Field(
        None, description="Primary address listed on the identification document."
    )


class ExtractionRequest(BaseModel):
    """User-configurable options that influence the extraction pipeline."""

    include_address: bool = Field(
        True,
        description=(
            "When set to true the API will attempt to extract the registered address "
            "from the ID image."
        ),
    )


class ExtractionResponse(BaseModel):
    """Response envelope returned by the extraction endpoint."""

    fields: IDCardFields = Field(..., description="Parsed information detected on the ID card.")
    message: str = Field(
        "Extraction completed successfully.",
        description="Human-readable summary of the extraction result.",
    )
