"""Container number normalization (ISO 6346)."""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple


ISO6346_LETTER_VALUES = {
    "A": 10, "B": 12, "C": 13, "D": 14, "E": 15, "F": 16, "G": 17, "H": 18, "I": 19,
    "J": 20, "K": 21, "L": 23, "M": 24, "N": 25, "O": 26, "P": 27, "Q": 28, "R": 29,
    "S": 30, "T": 31, "U": 32, "V": 34, "W": 35, "X": 36, "Y": 37, "Z": 38,
}


def compute_check_digit(code10: str) -> Optional[str]:
    """Compute ISO 6346 check digit for a 10-character container code."""
    if not code10 or len(code10) != 10:
        return None

    total = 0
    for idx, ch in enumerate(code10):
        if ch.isdigit():
            value = int(ch)
        else:
            value = ISO6346_LETTER_VALUES.get(ch)
            if value is None:
                return None
        total += value * (2 ** idx)

    check = (total % 11) % 10
    return str(check)


def validate_container_number(code11: str) -> Optional[bool]:
    """Validate ISO 6346 check digit for a full 11-character container number."""
    if not code11 or len(code11) != 11:
        return None
    core = code11[:10]
    check = code11[10]
    computed = compute_check_digit(core)
    if computed is None:
        return None
    return check == computed


def _map_chars(text: str, mapping: Dict[str, str]) -> Tuple[str, int]:
    fixed = []
    changes = 0
    for ch in text:
        if ch in mapping:
            fixed.append(mapping[ch])
            changes += 1
        else:
            fixed.append(ch)
    return "".join(fixed), changes


def normalize_container_number(raw: str) -> Optional[Dict[str, object]]:
    """
    Normalize a raw model string to ISO 6346 container number format.

    Returns a dict with keys: text, formatted, check_digit, check_digit_ok,
    changes, score.
    """
    if not raw:
        return None

    cleaned = re.sub(r"[^0-9A-Z]", "", str(raw).upper())
    if len(cleaned) < 10:
        return None

    digit_map = {
        "O": "0", "Q": "0", "D": "0", "U": "0",
        "I": "1", "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
        "G": "6",
        "T": "7",
        "A": "4",
    }

    letter_map = {
        "0": "O", "1": "I", "2": "Z", "3": "B", "4": "A",
        "5": "S", "6": "G", "7": "T", "8": "B", "9": "G",
    }

    candidates = []

    for length in (11, 10):
        if len(cleaned) < length:
            continue
        for i in range(0, len(cleaned) - length + 1):
            sub = cleaned[i:i + length]

            head = sub[:4]
            tail = sub[4:10]
            check = sub[10] if length == 11 else None

            head_fixed, head_changes = _map_chars(head, letter_map)
            tail_fixed, tail_changes = _map_chars(tail, digit_map)

            if not (head_fixed.isalpha() and tail_fixed.isdigit()):
                continue

            check_fixed = None
            check_changes = 0
            if check is not None:
                check_fixed, check_changes = _map_chars(check, digit_map)
                if not check_fixed.isdigit():
                    continue

            core = f"{head_fixed}{tail_fixed}"
            computed = compute_check_digit(core)
            if computed is None:
                continue

            if check_fixed is None:
                final_check = computed
                check_ok = False
            else:
                final_check = check_fixed
                check_ok = final_check == computed

            changes = head_changes + tail_changes + check_changes
            text = f"{core}{final_check}"
            formatted = f"{head_fixed} {tail_fixed} {final_check}"
            score = changes * 10
            if check_ok:
                score -= 15

            candidates.append({
                "text": text,
                "formatted": formatted,
                "check_digit": final_check,
                "check_digit_ok": check_ok,
                "changes": changes,
                "score": score,
            })

    if not candidates:
        return None

    candidates.sort(key=lambda c: c["score"])
    return candidates[0]
