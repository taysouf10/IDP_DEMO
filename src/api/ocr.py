"""Utility helpers for transforming OCR output into structured ID card data."""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from typing import Callable, Mapping, Sequence

__all__ = [
    "OCRExtractionError",
    "build_id_card_fields_from_ocr",
    "extract_id_card_fields",
]


class OCRExtractionError(RuntimeError):
    """Raised when OCR output cannot be converted into structured information."""


FieldValue = str | date
RawZoneTokens = Sequence[str] | str
ZoneMapping = Mapping[str, RawZoneTokens]
FieldParser = Callable[[Sequence[str]], FieldValue]

_RESULT_FIELDS = ("cin", "full_name", "date_of_birth", "city", "address")

_MONTH_VARIANTS: dict[int, tuple[str, ...]] = {
    1: ("jan", "janv", "january", "janvier"),
    2: ("feb", "fev", "february", "fevrier"),
    3: ("mar", "mars", "march"),
    4: ("apr", "apr", "april", "avr", "avril"),
    5: ("may", "mai"),
    6: ("jun", "june", "juin"),
    7: ("jul", "july", "juil", "juillet"),
    8: ("aug", "aou", "aout", "august"),
    9: ("sep", "sept", "september", "septembre"),
    10: ("oct", "october", "octobre"),
    11: ("nov", "november", "novembre"),
    12: ("dec", "december", "decembre"),
}

_MONTH_ALIASES = {
    alias: month
    for month, aliases in _MONTH_VARIANTS.items()
    for alias in aliases
}


def _strip_accents(value: str) -> str:
    """Return a lower-cased version of *value* without diacritical marks."""

    normalized = unicodedata.normalize("NFKD", value)
    without_marks = "".join(char for char in normalized if not unicodedata.combining(char))
    return without_marks.casefold()


def _collapse_whitespace(value: str) -> str:
    """Collapse consecutive whitespace characters into a single space."""

    return re.sub(r"\s+", " ", value).strip()


def _prepare_tokens(raw_tokens: RawZoneTokens) -> list[str]:
    """Normalise a token collection into a list of cleaned strings."""

    if isinstance(raw_tokens, str):
        iterable: Sequence[str] = [raw_tokens]
    else:
        iterable = raw_tokens

    cleaned: list[str] = []
    for token in iterable:
        if token is None:
            continue
        text = str(token).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _canonical_zone_name(name: str) -> str:
    """Convert a raw zone name into a canonical identifier."""

    return "".join(char for char in name.casefold() if char.isalnum())


def _parse_numeric_component(token: str) -> int | None:
    """Extract the first integer present in *token*, if any."""

    match = re.search(r"\d+", token)
    if not match:
        return None
    return int(match.group())


def _parse_day_token(token: str) -> int | None:
    """Attempt to parse a day value from *token*."""

    day = _parse_numeric_component(token)
    if day is None or not 1 <= day <= 31:
        return None
    return day


def _parse_year_token(token: str) -> int | None:
    """Attempt to parse a four-digit year from *token*."""

    match = re.search(r"\d+", token)
    if not match:
        return None
    raw = match.group()
    year = int(raw)
    if len(raw) == 2:
        year += 1900 if year >= 50 else 2000
    if year < 1900 or year > date.today().year + 1:
        return None
    return year


def _parse_month_token(token: str) -> int | None:
    """Attempt to parse a month value from *token* supporting digits and month names."""

    stripped = _strip_accents(token)
    letters_only = re.sub(r"[^a-z]", "", stripped)
    if letters_only:
        if letters_only.isdigit():
            month_value = int(letters_only)
            if 1 <= month_value <= 12:
                return month_value
        alias = _MONTH_ALIASES.get(letters_only)
        if alias is not None:
            return alias
        if len(letters_only) >= 3:
            alias = _MONTH_ALIASES.get(letters_only[:3])
            if alias is not None:
                return alias

    digit_match = re.search(r"\d+", stripped)
    if not digit_match:
        return None
    month_value = int(digit_match.group())
    if 1 <= month_value <= 12:
        return month_value
    return None


def _normalise_cin(tokens: Sequence[str]) -> str:
    """Return a normalised CIN string from the provided *tokens*."""

    combined = "".join(token.strip() for token in tokens)
    combined = combined.replace(" ", "").replace("-", "").upper()
    if not combined:
        raise OCRExtractionError("No characters found for CIN field.")
    match = re.fullmatch(r"([A-Z]{1,2})(\d{5,7})", combined)
    if match is None:
        raise OCRExtractionError(f"Invalid CIN format detected: {combined!r}")
    return f"{match.group(1)}{match.group(2)}"


def _normalise_name(tokens: Sequence[str]) -> str:
    """Return a clean representation of the citizen's full name."""

    value = _collapse_whitespace(" ".join(tokens))
    if not value:
        raise OCRExtractionError("Name tokens did not contain any characters.")
    return value


def _parse_birth_date(tokens: Sequence[str]) -> date:
    """Parse a birth date from OCR tokens."""

    prepared = _prepare_tokens(tokens)
    if not prepared:
        raise OCRExtractionError("Birth date tokens are empty.")

    joined = _collapse_whitespace(" ".join(prepared))
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(joined, fmt).date()
        except ValueError:
            continue

    numeric_parts = re.findall(r"\d+", joined)
    if len(numeric_parts) >= 3:
        day = int(numeric_parts[0])
        month = int(numeric_parts[1])
        year = int(numeric_parts[2])
        if len(numeric_parts[2]) == 2:
            year += 1900 if year >= 50 else 2000
        try:
            return date(year, month, day)
        except ValueError:
            pass

    for idx in range(len(prepared) - 2):
        window = prepared[idx : idx + 3]
        day = _parse_day_token(window[0])
        month = _parse_month_token(window[1])
        year = _parse_year_token(window[2])
        if day and month and year:
            try:
                return date(year, month, day)
            except ValueError:
                continue

    raise OCRExtractionError(f"Unable to parse date of birth from tokens: {prepared!r}")


def _normalise_city(tokens: Sequence[str]) -> str:
    """Normalise the city/place of birth field."""

    value = _collapse_whitespace(" ".join(tokens))
    if not value:
        raise OCRExtractionError("City tokens did not contain any characters.")
    return value


def _normalise_address(tokens: Sequence[str]) -> str:
    """Normalise the postal address extracted from the ID card."""

    value = _collapse_whitespace(" ".join(tokens))
    if not value:
        raise OCRExtractionError("Address tokens did not contain any characters.")
    return value


_FIELD_NORMALISERS: dict[str, FieldParser] = {
    "cin": _normalise_cin,
    "full_name": _normalise_name,
    "date_of_birth": _parse_birth_date,
    "city": _normalise_city,
    "address": _normalise_address,
}

_ZONE_TO_FIELD: dict[str, str] = {
    "cin": "cin",
    "cinzone": "cin",
    "idnumber": "cin",
    "identificationnumber": "cin",
    "numerocin": "cin",
    "fullname": "full_name",
    "name": "full_name",
    "prenomnom": "full_name",
    "citizenname": "full_name",
    "dateofbirth": "date_of_birth",
    "birthdate": "date_of_birth",
    "dob": "date_of_birth",
    "datenaissance": "date_of_birth",
    "birthplace": "city",
    "birthcity": "city",
    "city": "city",
    "lieunaissance": "city",
    "placeofbirth": "city",
    "ville": "city",
    "address": "address",
    "adresse": "address",
    "residence": "address",
    "domicile": "address",
}


def build_id_card_fields_from_ocr(
    zones: ZoneMapping, *, include_address: bool = True
) -> dict[str, FieldValue | None]:
    """Convert OCR *zones* into a dictionary of structured ID card fields."""

    result: dict[str, FieldValue | None] = {field: None for field in _RESULT_FIELDS}

    for raw_zone, raw_tokens in zones.items():
        canonical = _canonical_zone_name(str(raw_zone))
        field_name = _ZONE_TO_FIELD.get(canonical)
        if field_name is None:
            continue
        if field_name == "address" and not include_address:
            continue

        tokens = _prepare_tokens(raw_tokens)
        if not tokens:
            continue

        normaliser = _FIELD_NORMALISERS[field_name]
        try:
            result[field_name] = normaliser(tokens)
        except OCRExtractionError:
            result[field_name] = None

    if not include_address:
        result["address"] = None

    if all(value is None for value in result.values()):
        raise OCRExtractionError("No ID card fields could be extracted from OCR output.")

    return result


def extract_id_card_fields(
    zones: ZoneMapping, *, include_address: bool = True
) -> dict[str, FieldValue | None]:
    """Public helper that returns structured ID card data from OCR zones."""

    return build_id_card_fields_from_ocr(zones, include_address=include_address)

