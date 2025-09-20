"""OCR and parsing utilities for Moroccan ID cards."""

from __future__ import annotations

import io
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Optional

import cv2  # type: ignore
import numpy as np
from PIL import Image, UnidentifiedImageError
import pytesseract
from pytesseract import Output, TesseractError, TesseractNotFoundError

from .schemas import IDCardFields

LOGGER = logging.getLogger(__name__)


class OCRProcessingError(RuntimeError):
    """Raised when OCR processing fails to produce structured data."""


@dataclass(frozen=True)
class OCRResult:
    """Container for the intermediate OCR outputs."""

    text: str
    data: dict[str, list]


def extract_id_fields(image_bytes: bytes, include_address: bool) -> IDCardFields:
    """Run OCR on the provided Moroccan ID card image and parse structured fields.

    Parameters
    ----------
    image_bytes:
        Raw bytes representing the uploaded ID card image.
    include_address:
        When ``True`` the parser will attempt to extract the postal address. When
        ``False`` the address is omitted from the returned data.

    Returns
    -------
    IDCardFields
        Parsed CIN, full name, date of birth and optionally the address.

    Raises
    ------
    OCRProcessingError
        Raised when the OCR engine fails or the expected fields cannot be
        detected in the resulting text.
    """

    image = _load_image(image_bytes)
    processed = _preprocess_for_ocr(image)
    ocr_result = _perform_ocr(processed)

    lines = _extract_lines(ocr_result.text)
    cin = _parse_cin(ocr_result.data, ocr_result.text)
    full_name = _parse_full_name(lines, ocr_result.data)
    date_of_birth = _parse_date_of_birth(lines)
    address = _parse_address(lines, ocr_result.data) if include_address else None

    missing = [
        field
        for field, value in {
            "cin": cin,
            "full_name": full_name,
            "date_of_birth": date_of_birth,
        }.items()
        if value in (None, "")
    ]
    if missing:
        raise OCRProcessingError(
            "Unable to detect the following field(s) on the ID card: "
            + ", ".join(missing)
        )

    return IDCardFields(
        cin=cin,
        full_name=full_name,
        date_of_birth=date_of_birth,
        address=address,
    )


def _load_image(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise OCRProcessingError("The uploaded image appears to be empty.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            return img.convert("RGB")
    except UnidentifiedImageError as exc:  # pragma: no cover - safety net
        raise OCRProcessingError("Unsupported or corrupted image file provided.") from exc


def _preprocess_for_ocr(image: Image.Image) -> np.ndarray:
    """Apply grayscale conversion, denoising and adaptive thresholding."""

    np_image = np.array(image)
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
    thresholded = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8,
    )
    return thresholded


def _perform_ocr(image: np.ndarray) -> OCRResult:
    try:
        text = pytesseract.image_to_string(
            image,
            lang="eng+fra",
            config="--oem 3 --psm 6",
        )
        data = pytesseract.image_to_data(
            image,
            lang="eng+fra",
            config="--oem 3 --psm 6",
            output_type=Output.DICT,
        )
    except (TesseractNotFoundError, TesseractError) as exc:  # pragma: no cover
        LOGGER.exception("Tesseract OCR execution failed")
        raise OCRProcessingError("OCR engine is not available on the server.") from exc

    return OCRResult(text=text, data=data)


def _extract_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _parse_cin(data: dict[str, list], raw_text: str) -> Optional[str]:
    token_candidates = _iter_tokens(data)
    for raw_token, normalized in token_candidates:
        normalized_token = normalized.replace(" ", "")
        match = re.fullmatch(r"[A-Z]{1,2}\d{5,6}", normalized_token)
        if match:
            return normalized_token

    normalized_text = _normalise_text(raw_text)
    match = re.search(r"\b([A-Z]{1,2}\d{5,6})\b", normalized_text)
    if match:
        return match.group(1)

    return None


def _parse_full_name(lines: list[str], data: dict[str, list]) -> Optional[str]:
    label_pattern = re.compile(
        r"^\s*(?:nom(?:\s+et\s+pr[eé]nom[s]?)?|pr[eé]nom[s]?)[:\s\-]*(?P<value>.+)$",
        re.IGNORECASE,
    )
    for line in lines:
        match = label_pattern.match(line)
        if match:
            return _clean_value(match.group("value"))

    # Fallback to aggregated tokens - longest alphabetical span
    tokens = [orig for orig, norm in _iter_tokens(data) if orig and orig.isalpha()]
    if tokens:
        candidate = " ".join(tokens)
        if candidate:
            return _clean_value(candidate)

    return None


def _parse_date_of_birth(lines: list[str]) -> Optional[date]:
    date_patterns = [
        re.compile(
            r"(?:naissance|date\s+de\s+naissance|n[eé]e?\s+le)[:\s\-]*(\d{2}[\-/\.]\d{2}[\-/\.]\d{4})",
            re.IGNORECASE,
        ),
        re.compile(r"\b(\d{2}[\-/\.]\d{2}[\-/\.]\d{4})\b"),
    ]
    for line in lines:
        for pattern in date_patterns:
            match = pattern.search(line)
            if match:
                raw = match.group(1)
                parsed = _parse_date(raw)
                if parsed:
                    return parsed
    return None


def _parse_address(lines: list[str], data: dict[str, list]) -> Optional[str]:
    address_pattern = re.compile(
        r"^\s*(?:adresse|adress|adr)[\s:.-]*(?P<value>.+)$",
        re.IGNORECASE,
    )
    for idx, line in enumerate(lines):
        match = address_pattern.match(line)
        if match:
            value = _clean_value(match.group("value"))
            if not value and idx + 1 < len(lines):
                value = _clean_value(line + " " + lines[idx + 1])
            if value:
                return value

    # Fallback to join tokens following address keyword
    tokens = list(_iter_tokens(data))
    for index, (original, normalized) in enumerate(tokens):
        if normalized.startswith("ADRESSE"):
            trailing = [tok for tok, _ in tokens[index + 1 : index + 6]]
            candidate = _clean_value(" ".join(trailing))
            if candidate:
                return candidate
    return None


def _parse_date(raw: str) -> Optional[date]:
    cleaned = raw.replace(".", "/").replace("-", "/")
    for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def _iter_tokens(data: dict[str, list]) -> Iterable[tuple[str, str]]:
    texts = data.get("text", []) or []
    confs = data.get("conf", []) or []
    for raw, conf in zip(texts, confs):
        if not raw:
            continue
        try:
            confidence = float(conf)
        except (TypeError, ValueError):  # pragma: no cover
            confidence = -1.0
        if confidence < 0:  # ignore uncertain detections
            continue
        normalized = _normalise_text(raw)
        if not normalized:
            continue
        yield _clean_value(raw), normalized


def _normalise_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"[^A-Z0-9\s:/-]", "", normalized.upper())
    return _clean_value(normalized)


def _clean_value(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


__all__ = ["extract_id_fields", "OCRProcessingError"]
