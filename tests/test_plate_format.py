import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from traffic_system.ocr.plate_format import IndianPlateValidator, get_validator


class TestIndianPlateValidator:
    def setup_method(self):
        self.validator = IndianPlateValidator()

    def test_valid_plate_passes_unchanged(self):
        result = self.validator.validate("MH12AB1234")
        assert result.format_valid is True
        assert result.corrected_text == "MH12AB1234"

    def test_corrects_zero_to_o_in_state_code_position(self):
        # "0H" should become "OH" since positions 0-1 must be letters
        result = self.validator.validate("0H12AB1234")
        assert result.corrected_text[:2] == "OH"

    def test_corrects_o_to_zero_in_digit_position(self):
        # district code positions (2-3) must be digits; "O" -> "0"
        result = self.validator.validate("MHO2AB1234")
        assert result.corrected_text[2] == "0"

    def test_too_short_text_returns_unchanged_and_invalid(self):
        result = self.validator.validate("AB")
        assert result.format_valid is False
        assert result.corrected_text == "AB"

    def test_strips_non_alphanumeric_characters(self):
        result = self.validator.validate("MH-12-AB-1234")
        assert result.corrected_text == "MH12AB1234"

    def test_invalid_format_flagged(self):
        result = self.validator.validate("12345678")
        assert result.format_valid is False


class TestValidatorRegistry:
    def test_get_validator_returns_indian_validator_for_in(self):
        validator = get_validator("IN")
        assert isinstance(validator, IndianPlateValidator)

    def test_get_validator_raises_for_unknown_region(self):
        try:
            get_validator("ZZ")
            assert False, "expected ValueError"
        except ValueError:
            pass
