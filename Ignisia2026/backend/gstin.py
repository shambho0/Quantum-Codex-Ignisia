"""
GSTIN format validation (India).

Structure (15 characters):
  1–2   State code (digits)
  3–12  PAN: 5 letters + 4 digits + 1 letter
  13    Entity number (single digit)
  14    Default character 'Z'
  15    Check character (A–Z or 0–9)
"""

from __future__ import annotations

import re

_PAN_IN_GSTIN = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")


def validate_gstin(raw: str) -> tuple[bool, str]:
    """
    Returns (is_valid, error_message). error_message is empty when valid.
    """
    s = (raw or "").strip().upper()
    if len(s) != 15:
        return False, "GSTIN must be exactly 15 characters (no more, no less)."

    if not s[:2].isdigit():
        return False, "Positions 1-2 must be the State Code (two digits, e.g. 27 for Maharashtra)."

    pan = s[2:12]
    if not _PAN_IN_GSTIN.match(pan):
        return False, (
            "Positions 3-12 must be a valid PAN: five letters, four digits, "
            "then one letter (e.g. ABCDE1234F)."
        )

    if not s[12].isdigit():
        return False, "Position 13 must be the Entity Number (a single digit 0–9)."

    if s[13] != "Z":
        return False, "Position 14 must be 'Z' (the default GSTIN character)."

    if s[14] not in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return False, "Position 15 must be the check code (letter A–Z or digit 0–9)."

    return True, ""
