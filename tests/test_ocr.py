"""Tests for the OCR parsing pipeline."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import patch

import pytest
from PIL import Image

from api.ocr import OCRProcessingError, extract_id_fields

OCR_TEXT = """
ROYAUME DU MAROC
CIN AB123456
NOM ET PRENOM REDACTED PERSON
NAISSANCE 01/02/1985
ADRESSE 99 RUE EXEMPLE RABAT
""".strip()

OCR_DATA = {
    "text": [
        "ROYAUME",
        "DU",
        "MAROC",
        "CIN",
        "AB123456",
        "NOM",
        "ET",
        "PRENOM",
        "REDACTED",
        "PERSON",
        "NAISSANCE",
        "01/02/1985",
        "ADRESSE",
        "99",
        "RUE",
        "EXEMPLE",
        "RABAT",
    ],
    "conf": ["95"] * 17,
}


@pytest.fixture()
def sample_image_bytes() -> bytes:
    """Return an in-memory PNG image suitable for OCR preprocessing."""

    image = Image.new("RGB", (256, 256), color=(240, 240, 240))
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return buffer.getvalue()


@patch("api.ocr.pytesseract.image_to_string", return_value=OCR_TEXT)
@patch("api.ocr.pytesseract.image_to_data", return_value=OCR_DATA)
def test_extract_id_fields_returns_expected_data(
    mock_data,
    mock_text,
    sample_image_bytes: bytes,
) -> None:
    result = extract_id_fields(sample_image_bytes, include_address=True)

    assert result.cin == "AB123456"
    assert result.full_name == "REDACTED PERSON"
    assert result.date_of_birth.isoformat() == "1985-02-01"
    assert result.address == "99 RUE EXEMPLE RABAT"


@patch("api.ocr.pytesseract.image_to_string", return_value=OCR_TEXT)
@patch("api.ocr.pytesseract.image_to_data", return_value=OCR_DATA)
def test_extract_id_fields_can_skip_address(
    mock_data,
    mock_text,
    sample_image_bytes: bytes,
) -> None:
    result = extract_id_fields(sample_image_bytes, include_address=False)

    assert result.cin == "AB123456"
    assert result.address is None


@patch("api.ocr.pytesseract.image_to_string", return_value="INCOMPLETE DATA")
@patch("api.ocr.pytesseract.image_to_data", return_value={"text": ["HELLO"], "conf": ["90"]})
def test_extract_id_fields_raises_when_fields_missing(
    mock_data,
    mock_text,
    sample_image_bytes: bytes,
) -> None:
    with pytest.raises(OCRProcessingError):
        extract_id_fields(sample_image_bytes, include_address=True)
