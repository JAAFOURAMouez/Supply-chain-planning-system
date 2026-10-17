"""
Deterministic extraction of machine disruptions (pannes / indisponibilités)
from a natural-language description. Not delegated to an LLM: the interval
must be copied exactly, never guessed.

Supported formulations (French / English):
  - "indisponibilité de M2 sur l'intervalle [0, 2)"
  - "panne de la machine M2 pendant les deux premières heures / unités"
  - "M2 en panne de 3 à 5" / "entre 3 et 5"
  - "M2 breakdown during the first 2 hours"
"""
import re
from typing import Dict, List, Tuple

_KEYWORDS = re.compile(r"panne|indisponib|arr[êe]t|maintenance|downtime|unavailab|breakdown", re.IGNORECASE)
_MACHINE = re.compile(r"\b(?:M|Machine\s*)(\d+)\b", re.IGNORECASE)
_BRACKET = re.compile(r"[\[\(]\s*(\d+)\s*[,;]\s*(\d+)\s*[\)\]]")
_NUM_WORDS = {"un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6,
              "sept": 7, "huit": 8, "neuf": 9, "dix": 10, "one": 1, "two": 2, "three": 3,
              "four": 4, "five": 5, "six_": 6}
_NUM = r"(\d+|" + "|".join(k for k in _NUM_WORDS if not k.endswith("_")) + r")"
_FIRST = re.compile(r"(?:" + _NUM + r"\s+)?(?:premi[èe]res?|first)\s+(?:" + _NUM + r"\s+)?(unit[ée]s?(?:\s+de\s+temps)?|heures?|hours?|time\s+units?)", re.IGNORECASE)
_FROM_TO = re.compile(r"(?:\bde\b|\bfrom\b|\bentre\b|\bbetween\b)\s+(\d+)\s*(?:h|heures?|unit[ée]s?|hours?)?\s*(?:\bà\b|\bto\b|\bet\b|\band\b|-)\s*(\d+)", re.IGNORECASE)


def _to_int(token: str) -> int:
    return int(token) if token.isdigit() else _NUM_WORDS[token.lower()]


def mentions_disruption(text: str) -> bool:
    return bool(_KEYWORDS.search(text or ""))


def extract_disruptions(text: str) -> Tuple[Dict[str, List[List[int]]], List[str]]:
    """
    Returns (machine_availability, warnings).
    Each interval is attached to the closest machine mentioned before it in the
    same sentence (or the only machine of the sentence). Sentences must contain
    a breakdown / unavailability keyword.
    """
    availability: Dict[str, List[List[int]]] = {}
    warnings: List[str] = []

    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        if not _KEYWORDS.search(sentence):
            continue
        found = []  # (position, start, end, note)
        for m in _BRACKET.finditer(sentence):
            found.append((m.start(), int(m.group(1)), int(m.group(2)), None))
        for m in _FIRST.finditer(sentence):
            n_tok = m.group(1) or m.group(2)
            if not n_tok:
                continue
            unit = m.group(3).lower()
            note = None
            if unit.startswith(("heure", "hour")):
                note = (f"Hypothèse: « {n_tok} {unit} » interprété comme {_to_int(n_tok)} unités de temps "
                        "(1 heure = 1 unité). Les durées des données n'ont pas d'unité précisée: à confirmer.")
            found.append((m.start(), 0, _to_int(n_tok), note))
        if not found:
            for m in _FROM_TO.finditer(sentence):
                found.append((m.start(), int(m.group(1)), int(m.group(2)), None))

        machines_all = {f"M{m.group(1)}" for m in _MACHINE.finditer(sentence)}
        for pos, a, b, note in found:
            before = [m for m in _MACHINE.finditer(sentence) if m.start() < pos]
            if before:
                machine_id = f"M{before[-1].group(1)}"
            elif len(machines_all) == 1:
                machine_id = next(iter(machines_all))
            else:
                warnings.append(f"Intervalle [{a}, {b}) sans machine identifiable: ignoré (précision nécessaire)")
                continue
            if b <= a:
                warnings.append(f"Intervalle invalide [{a}, {b}) pour {machine_id}: ignoré")
                continue
            ivs = availability.setdefault(machine_id, [])
            if [a, b] not in ivs:
                ivs.append([a, b])
            if note:
                warnings.append(note)
    return availability, warnings


def parse_interval(text: str):
    """
    Parse one time interval from free text.
    Returns ((start, end), note_or_None) or None if nothing usable was found.
    """
    m = _BRACKET.search(text or "")
    if m and int(m.group(2)) > int(m.group(1)):
        return (int(m.group(1)), int(m.group(2))), None
    m = _FIRST.search(text or "")
    if m:
        n_tok = m.group(1) or m.group(2)
        if n_tok:
            n = _to_int(n_tok)
            unit = m.group(3).lower()
            note = None
            if unit.startswith(("heure", "hour")):
                note = (f"Hypothèse: « {n_tok} {unit} » interprété comme {n} unités de temps "
                        "(1 heure = 1 unité). Les durées des données n'ont pas d'unité précisée: à confirmer.")
            return (0, n), note
    m = _FROM_TO.search(text or "")
    if m and int(m.group(2)) > int(m.group(1)):
        return (int(m.group(1)), int(m.group(2))), None
    return None
