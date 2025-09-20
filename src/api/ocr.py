"""OCR parsing utilities for Moroccan ID cards."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import BytesIO
import re
from typing import Iterable, Mapping, Sequence

from PIL import Image, UnidentifiedImageError

try:  # pragma: no cover - exercised via integration tests
    import pytesseract
    from pytesseract import Output
except ImportError:  # pragma: no cover - handled gracefully at runtime
    pytesseract = None  # type: ignore[assignment]
    Output = None  # type: ignore[assignment]

from .schemas import IDCardFields


@dataclass(frozen=True)
class BoundingBox:
    """Normalized bounding box with coordinates expressed between 0 and 1."""

    left: float
    top: float
    right: float
    bottom: float

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2


@dataclass(frozen=True)
class OcrToken:
    """OCR token enriched with a normalized bounding box."""

    text: str
    confidence: float
    bbox: BoundingBox


@dataclass(frozen=True)
class CardRegion:
    """Normalized rectangular area corresponding to a field on the ID card."""

    left: float
    top: float
    right: float
    bottom: float

    def contains(self, x: float, y: float) -> bool:
        """Return True when the provided point lies within the region bounds."""

        return self.left <= x <= self.right and self.top <= y <= self.bottom


DEFAULT_CARD_REGIONS: dict[str, CardRegion] = {
    "cin": CardRegion(left=0.05, top=0.05, right=0.35, bottom=0.20),
    "name": CardRegion(left=0.05, top=0.22, right=0.95, bottom=0.40),
    "birth_date": CardRegion(left=0.05, top=0.42, right=0.45, bottom=0.55),
    "address": CardRegion(left=0.05, top=0.57, right=0.95, bottom=0.88),
}


_MIN_CONFIDENCE = 40.0
_DATE_PATTERN = re.compile(r"(\d{1,2})\D+(\d{1,2})\D+(\d{2,4})")


def _compute_card_dimensions(
    lefts: Sequence[float],
    tops: Sequence[float],
    widths: Sequence[float],
    heights: Sequence[float],
) -> tuple[float, float]:
    """Estimate the width and height of the card from token bounding boxes."""

    max_right = max((left + width for left, width in zip(lefts, widths)), default=0.0)
    max_bottom = max((top + height for top, height in zip(tops, heights)), default=0.0)

    # Avoid division by zero when normalizing bounding boxes later on.
    return max(max_right, 1.0), max(max_bottom, 1.0)


def _coerce_float(value: str | float | int) -> float:
    """Safely cast Tesseract values to floats."""

    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive branch
        return 0.0


def normalise_tokens(
    ocr_data: Mapping[str, Sequence],
    *,
    drop_empty: bool = True,
) -> list[OcrToken]:
    """Transform the ``pytesseract.image_to_data`` output into normalized tokens."""

    required_keys = {"text", "conf", "left", "top", "width", "height"}
    missing = required_keys.difference(ocr_data)
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(f"OCR data is missing required keys: {missing_keys}")

    texts: Sequence[str] = ocr_data["text"]  # type: ignore[assignment]
    confidences: Sequence[str | float | int] = ocr_data["conf"]  # type: ignore[assignment]
    lefts: Sequence[float | int] = ocr_data["left"]  # type: ignore[assignment]
    tops: Sequence[float | int] = ocr_data["top"]  # type: ignore[assignment]
    widths: Sequence[float | int] = ocr_data["width"]  # type: ignore[assignment]
    heights: Sequence[float | int] = ocr_data["height"]  # type: ignore[assignment]

    if not isinstance(texts, Iterable):  # pragma: no cover - defensive guard
        raise ValueError("OCR data 'text' entry must be iterable.")

    card_width, card_height = _compute_card_dimensions(
        [float(value) for value in lefts],
        [float(value) for value in tops],
        [float(value) for value in widths],
        [float(value) for value in heights],
    )

    tokens: list[OcrToken] = []
    for text, conf, left, top, width, height in zip(
        texts, confidences, lefts, tops, widths, heights
    ):
        if drop_empty and (not isinstance(text, str) or not text.strip()):
            continue

        confidence = _coerce_float(conf)
        left_f = float(left)
        top_f = float(top)
        width_f = float(width)
        height_f = float(height)

        bbox = BoundingBox(
            left=max(left_f / card_width, 0.0),
            top=max(top_f / card_height, 0.0),
            right=min((left_f + width_f) / card_width, 1.0),
            bottom=min((top_f + height_f) / card_height, 1.0),
        )
        tokens.append(OcrToken(text=text.strip(), confidence=confidence, bbox=bbox))

    return tokens


def _sort_tokens(tokens: Iterable[OcrToken]) -> list[OcrToken]:
    """Sort OCR tokens in a stable reading order."""

    return sorted(
        tokens,
        key=lambda token: (round(token.bbox.top, 3), token.bbox.left),
    )


def _aggregate_region_text(
    tokens: Iterable[OcrToken],
    region: CardRegion,
    *,
    min_confidence: float = _MIN_CONFIDENCE,
) -> str:
    """Combine the text for tokens whose centers fall inside ``region``."""

    region_tokens = [
        token
        for token in tokens
        if region.contains(token.bbox.center_x, token.bbox.center_y)
    ]
    if not region_tokens:
        return ""

    confident_tokens = [
        token for token in region_tokens if token.confidence >= min_confidence
    ]
    chosen = confident_tokens or region_tokens

    ordered_tokens = _sort_tokens(chosen)
    words = [token.text for token in ordered_tokens if token.text]
    return " ".join(words).strip()


def _normalise_cin(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]", "", text.upper())
    return cleaned


def _parse_birth_date(text: str) -> date | None:
    match = _DATE_PATTERN.search(text)
    if not match:
        return None

    day_s, month_s, year_s = match.groups()
    try:
        day = int(day_s)
        month = int(month_s)
        year = int(year_s)
    except ValueError:  # pragma: no cover - validated by regex
        return None

    if year < 100:
        year += 2000 if year < 30 else 1900

    try:
        return date(year, month, day)
    except ValueError:  # Invalid calendar date
        return None


def parse_id_card_fields(
    ocr_data: Mapping[str, Sequence],
    *,
    include_address: bool = True,
    regions: Mapping[str, CardRegion] = DEFAULT_CARD_REGIONS,
) -> IDCardFields:
    """Derive structured ID card fields from raw Tesseract OCR data."""

    tokens = normalise_tokens(ocr_data)
    if not tokens:
        raise ValueError("No OCR tokens could be extracted from the provided data.")

    cin_region = regions.get("cin")
    name_region = regions.get("name")
    birth_region = regions.get("birth_date")
    address_region = regions.get("address")

    if not all([cin_region, name_region, birth_region]):
        raise ValueError("Card regions configuration is incomplete.")

    cin_text = _aggregate_region_text(tokens, cin_region)
    cin_value = _normalise_cin(cin_text)
    if not cin_value:
        raise ValueError("Unable to determine the CIN from the OCR results.")

    full_name = _aggregate_region_text(tokens, name_region)
    if not full_name:
        raise ValueError("Unable to determine the full name from the OCR results.")

    birth_text = _aggregate_region_text(tokens, birth_region)
    birth_date = _parse_birth_date(birth_text)
    if birth_date is None:
        raise ValueError("Unable to parse the date of birth from the OCR results.")

    address_value: str | None = None
    if include_address and address_region is not None:
        address_text = _aggregate_region_text(tokens, address_region)
        address_value = address_text or None

    return IDCardFields(
        cin=cin_value,
        full_name=full_name,
        date_of_birth=birth_date,
        address=address_value,
    )


def extract_fields_from_image(
    image_bytes: bytes,
    *,
    include_address: bool = True,
    regions: Mapping[str, CardRegion] = DEFAULT_CARD_REGIONS,
) -> IDCardFields:
    """Run OCR over ``image_bytes`` and parse the resulting ID card fields."""

    if not image_bytes:
        raise ValueError("The uploaded file appears to be empty.")

    if pytesseract is None or Output is None:
        raise RuntimeError(
            "pytesseract is required to extract fields but it is not installed."
        )

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            rgb_image = image.convert("RGB")
            ocr_result = pytesseract.image_to_data(rgb_image, output_type=Output.DICT)
    except UnidentifiedImageError as exc:  # pragma: no cover - exercised in integration
        raise ValueError("The uploaded file is not a valid image.") from exc
    except pytesseract.TesseractError as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError("Tesseract OCR failed to process the provided image.") from exc

    return parse_id_card_fields(ocr_result, include_address=include_address, regions=regions)
