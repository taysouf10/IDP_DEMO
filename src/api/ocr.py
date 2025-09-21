"""OCR utilities for extracting Moroccan ID card information.

This module focuses on converting the raw output produced by
``pytesseract.image_to_data`` into structured information.  The helper
functions normalise bounding boxes, aggregate tokens inside calibrated
regions of the ID card and surface descriptive errors whenever required
content is missing.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Mapping, MutableMapping, Sequence


class OCRExtractionError(ValueError):
    """Raised when OCR data cannot be converted into structured fields."""


@dataclass(frozen=True)
class NormalizedBBox:
    """Bounding box expressed as ratios of the full image width/height."""

    left: float
    top: float
    right: float
    bottom: float

    @property
    def center(self) -> tuple[float, float]:
        """Return the centre point of the bounding box."""

        return ((self.left + self.right) / 2.0, (self.top + self.bottom) / 2.0)


@dataclass(frozen=True)
class NormalizedRegion:
    """Pre-calibrated rectangular region on the ID card."""

    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, bbox: NormalizedBBox) -> bool:
        """Return ``True`` when the bounding box centre lies inside the region."""

        cx, cy = bbox.center
        return self.x0 <= cx <= self.x1 and self.y0 <= cy <= self.y1


@dataclass
class OCRToken:
    """Container describing an OCR token and its normalised bounding box."""

    text: str
    confidence: float
    bbox: NormalizedBBox


# Card layout calibration expressed as ratios relative to the detected card size.
CARD_ZONES: Mapping[str, NormalizedRegion] = {
    "cin": NormalizedRegion(0.55, 0.05, 0.95, 0.20),
    "full_name": NormalizedRegion(0.05, 0.22, 0.55, 0.36),
    "date_of_birth": NormalizedRegion(0.05, 0.38, 0.45, 0.52),
    "address": NormalizedRegion(0.05, 0.54, 0.95, 0.78),
    "city": NormalizedRegion(0.05, 0.78, 0.40, 0.92),
}


def _ensure_required_keys(ocr_data: Mapping[str, Sequence[object]]) -> None:
    """Validate that the OCR dictionary contains the required keys."""

    required = {"text", "conf", "left", "top", "width", "height"}
    missing = required.difference(ocr_data)
    if missing:
        raise OCRExtractionError(
            "OCR data is missing required keys: " + ", ".join(sorted(missing))
        )


def _parse_int(value: object) -> int:
    """Parse a single value into an integer, treating blanks as zero."""

    if value is None:
        return 0
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(float(text)) if text else 0


def _parse_float(value: object) -> float:
    """Parse a single value into a float, treating blanks as ``-1``."""

    if value is None:
        return -1.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    return float(text) if text else -1.0


def _normalise_tokens(ocr_data: Mapping[str, Sequence[object]]) -> list[OCRToken]:
    """Convert ``pytesseract.image_to_data`` output into :class:`OCRToken` objects."""

    _ensure_required_keys(ocr_data)

    lefts = [max(_parse_int(value), 0) for value in ocr_data["left"]]
    tops = [max(_parse_int(value), 0) for value in ocr_data["top"]]
    widths = [max(_parse_int(value), 0) for value in ocr_data["width"]]
    heights = [max(_parse_int(value), 0) for value in ocr_data["height"]]
    confidences = [_parse_float(value) for value in ocr_data["conf"]]
    texts = [str(value or "").strip() for value in ocr_data["text"]]

    card_width = max((left + width) for left, width in zip(lefts, widths))
    card_height = max((top + height) for top, height in zip(tops, heights))

    if card_width <= 0 or card_height <= 0:
        raise OCRExtractionError(
            "Unable to determine card dimensions from OCR data; received zero width/height."
        )

    tokens: list[OCRToken] = []
    for left, top, width, height, conf, text in zip(
        lefts, tops, widths, heights, confidences, texts
    ):
        if not text or conf < 0:
            continue

        bbox = NormalizedBBox(
            left=left / card_width,
            top=top / card_height,
            right=(left + width) / card_width,
            bottom=(top + height) / card_height,
        )
        tokens.append(OCRToken(text=text, confidence=conf, bbox=bbox))

    return tokens


def _tokens_in_region(tokens: Iterable[OCRToken], region: NormalizedRegion) -> list[OCRToken]:
    """Return tokens whose centre lies inside the provided region."""

    return [token for token in tokens if region.contains(token.bbox)]


def _group_by_line(tokens: Sequence[OCRToken], threshold: float = 0.02) -> list[list[OCRToken]]:
    """Group tokens into textual lines based on their vertical proximity."""

    if not tokens:
        return []

    sorted_tokens = sorted(tokens, key=lambda tok: (tok.bbox.top, tok.bbox.left))

    lines: list[list[OCRToken]] = []
    current_line: list[OCRToken] = [sorted_tokens[0]]
    current_top = sorted_tokens[0].bbox.top

    for token in sorted_tokens[1:]:
        if abs(token.bbox.top - current_top) <= threshold:
            current_line.append(token)
        else:
            lines.append(current_line)
            current_line = [token]
            current_top = token.bbox.top

    lines.append(current_line)
    return lines


def _join_tokens(tokens: Sequence[OCRToken]) -> str:
    """Join tokens ordered by their horizontal position."""

    return " ".join(token.text for token in sorted(tokens, key=lambda tok: tok.bbox.left))


def _zone_text(tokens: Sequence[OCRToken]) -> str:
    """Return cleaned text extracted from a region."""

    lines = _group_by_line(tokens)
    joined = [
        _join_tokens(line).strip()
        for line in lines
        if any(word.text.strip() for word in line)
    ]
    return "\n".join(filter(None, joined)).strip()


def _normalise_cin(tokens: Sequence[OCRToken]) -> str:
    """Derive the CIN value from the corresponding region tokens."""

    cin_text = re.sub(r"\W+", "", _join_tokens(tokens)).upper()
    if not cin_text:
        raise OCRExtractionError("No characters recognised in the CIN zone.")
    return cin_text


def _normalise_name(tokens: Sequence[OCRToken]) -> str:
    """Derive the full name string from the relevant tokens."""

    name = _zone_text(tokens)
    if not name:
        raise OCRExtractionError("No characters recognised in the name zone.")
    # Normalise redundant spacing whilst preserving capitalisation from the card.
    name_parts = [part for part in re.split(r"\s+", name) if part]
    return " ".join(name_parts)


def _parse_birth_date(tokens: Sequence[OCRToken]) -> date:
    """Parse a date of birth from tokens located in the calibrated region."""

    text = re.sub(r"[^0-9]", " ", _join_tokens(tokens))
    parts = [part for part in text.split() if part]
    if len(parts) != 3:
        raise OCRExtractionError(
            "Unable to determine date of birth from detected text: " + _join_tokens(tokens)
        )

    values = [int(part) for part in parts]
    if values[0] > 1900:  # Format: YYYY MM DD
        year, month, day = values
    else:  # Default to Moroccan format: DD MM YYYY
        day, month, year = values
        if year < 100:  # Two-digit year fallback.
            year += 2000 if year < 30 else 1900

    try:
        return date(year, month, day)
    except ValueError as exc:  # pragma: no cover - defensive programming
        raise OCRExtractionError("Invalid date detected in the birth date zone.") from exc


def _normalise_address(tokens: Sequence[OCRToken]) -> str:
    """Return the multi-line address text."""

    address = _zone_text(tokens)
    if not address:
        raise OCRExtractionError("No characters recognised in the address zone.")
    return address


def _normalise_city(tokens: Sequence[OCRToken]) -> str:
    """Return the city extracted from its dedicated zone."""

    city = _zone_text(tokens)
    if not city:
        raise OCRExtractionError("No characters recognised in the city zone.")
    return city


def build_id_card_fields_from_ocr(
    ocr_data: Mapping[str, Sequence[object]],
    *,
    include_address: bool = True,
) -> MutableMapping[str, object]:
    """Aggregate OCR tokens into structured ID card fields.

    Parameters
    ----------
    ocr_data:
        Dictionary returned by :func:`pytesseract.image_to_data` with
        ``output_type`` set to ``pytesseract.Output.DICT``.
    include_address:
        When ``False`` the address zone is ignored and ``None`` is returned for
        the ``address`` field.
    """

    tokens = _normalise_tokens(ocr_data)
    if not tokens:
        raise OCRExtractionError("No OCR tokens detected on the ID card image.")

    zone_tokens = {
        name: _tokens_in_region(tokens, region)
        for name, region in CARD_ZONES.items()
    }

    # Ensure mandatory zones contain text and raise meaningful errors otherwise.
    mandatory_fields = ["cin", "full_name", "date_of_birth", "city"]
    for field in mandatory_fields:
        if not zone_tokens[field]:
            raise OCRExtractionError(
                f"No text detected in the {field.replace('_', ' ')} zone of the ID card."
            )

    aggregated: MutableMapping[str, object] = {
        "cin": _normalise_cin(zone_tokens["cin"]),
        "full_name": _normalise_name(zone_tokens["full_name"]),
        "date_of_birth": _parse_birth_date(zone_tokens["date_of_birth"]),
        "city": _normalise_city(zone_tokens["city"]),
    }

    if include_address:
        address_tokens = zone_tokens["address"]
        if not address_tokens:
            raise OCRExtractionError("No text detected in the address zone of the ID card.")
        aggregated["address"] = _normalise_address(address_tokens)
    else:
        aggregated["address"] = None

    return aggregated


def extract_id_card_fields(
    image_bytes: bytes,
    *,
    include_address: bool = True,
) -> "IDCardFields":
    """Run OCR on the provided image and return structured ID card fields."""

    if not image_bytes:
        raise OCRExtractionError("The uploaded file appears to be empty.")

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency missing during runtime
        raise OCRExtractionError("Pillow is required to decode ID card images.") from exc

    try:
        import pytesseract
    except ImportError as exc:  # pragma: no cover - dependency missing during runtime
        raise OCRExtractionError("pytesseract is required to run the OCR parser.") from exc

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb_image = img.convert("RGB")
            ocr_dict = pytesseract.image_to_data(
                rgb_image, output_type=pytesseract.Output.DICT
            )
    except pytesseract.TesseractError as exc:  # pragma: no cover - runtime safeguard
        raise OCRExtractionError("Tesseract OCR failed to process the supplied image.") from exc
    except OSError as exc:  # pragma: no cover - invalid image payloads
        raise OCRExtractionError("Unable to open the uploaded image for OCR.") from exc

    aggregated = build_id_card_fields_from_ocr(ocr_dict, include_address=include_address)

    # Import locally to avoid circular imports at module load time.
    from .schemas import IDCardFields

    return IDCardFields(**aggregated)

