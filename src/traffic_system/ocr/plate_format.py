"""
License plate format validation and OCR-error correction.

Implements the Indian plate format (SS-DD-AA-NNNN) ruleset referenced in
config.yaml's lpr.format_region. Built as a small pluggable registry keyed
by region code so a different country's format can be added without
touching the OCR pipeline itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ValidationResult:
    corrected_text: str
    format_valid: bool


class PlateFormatValidator:
    """Base class — region-specific validators implement `validate`."""

    def validate(self, raw_text: str) -> ValidationResult:
        raise NotImplementedError


class IndianPlateValidator(PlateFormatValidator):
    _PATTERN = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$")

    # Common OCR confusions, applied positionally — letters where letters
    # are expected, digits where digits are expected.
    _LETTER_POS_FIX = {"0": "O", "1": "I", "8": "B", "5": "S", "2": "Z"}
    _DIGIT_POS_FIX = {"O": "0", "I": "1", "B": "8", "S": "5", "Z": "2", "Q": "0"}

    def validate(self, raw_text: str) -> ValidationResult:
        cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
        corrected = self._apply_positional_correction(cleaned)
        is_valid = bool(self._PATTERN.match(corrected))
        return ValidationResult(corrected_text=corrected, format_valid=is_valid)

    def _apply_positional_correction(self, text: str) -> str:
        if len(text) < 6:
            return text
        chars = list(text)
        # Positions 0-1: state code letters
        for i in (0, 1):
            chars[i] = self._LETTER_POS_FIX.get(chars[i], chars[i])
        # Positions 2-3: district code digits
        for i in (2, 3):
            chars[i] = self._DIGIT_POS_FIX.get(chars[i], chars[i])
        # Last 4 positions: registration number digits
        for i in range(len(chars) - 4, len(chars)):
            chars[i] = self._DIGIT_POS_FIX.get(chars[i], chars[i])
        return "".join(chars)


_REGISTRY: dict[str, type[PlateFormatValidator]] = {
    "IN": IndianPlateValidator,
}


def get_validator(region_code: str) -> PlateFormatValidator:
    validator_cls = _REGISTRY.get(region_code)
    if validator_cls is None:
        raise ValueError(
            f"No plate format validator registered for region '{region_code}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return validator_cls()
