"""
Conversational front-end: understands a disruption request, asks for what is missing
(slot filling) and only then returns a structured scenario to simulate.

Slots are extracted deterministically against the loaded dataset (known machines / components),
so nothing is ever guessed. The LLM (if any) is only used to answer follow-up questions from
the computed facts.
"""
import re
from typing import Any, Dict, List, Optional

from src.utils.disruptions import parse_interval

I = re.IGNORECASE

_COMPONENT_INTENT = re.compile(
    r"composant|component|mati[èe]re|approvisionnement|fournisseur|p[ée]nurie|rupture|shortage|"
    r"retard\w*\s+(?:de\s+)?livraison|livraison\s+(?:en\s+)?retard", I)
_MACHINE_INTENT = re.compile(r"panne|indisponib|breakdown|downtime|maintenance|arr[êe]t\s+(?:de\s+la\s+)?machine", I)
_CANCEL = re.compile(r"\b(annule|annuler|laisse\s+tomber|oublie|stop|cancel)\b", I)

_MACHINE_GENERIC = re.compile(r"\b(?:machine\s*|M)(\d+)\b", I)
_COMPONENT_GENERIC = re.compile(r"composant\s+(?:n[°o]\s*)?([A-Za-z]*\d[\w\-]*)", I)
_DATE = re.compile(
    r"(?:jusqu['’]?\s*(?:à|a|au|en)|arriv\w*\s+(?:à|a|au|en|vers)|disponible\s+(?:à partir de|a partir de|à|a|au|en)"
    r"|nouvelle\s+date\w*\s*(?:d['’]arriv\w+)?\s*(?:est|:|=|à|au)?|(?:au\s+)?temps|\bà\s+t\s*=|\bt\s*=)"
    r"\s*(?:t\s*=\s*)?(\d+)", I)
_DELAY = re.compile(r"(?:retard|d[ée]cal\w+|report\w*)\s+de\s+(\d+)|(\d+)\s+(?:unit[ée]s?(?:\s+de\s+temps)?|heures?)\s+de\s+retard", I)
_BARE_NUM = re.compile(r"^\s*(?:t\s*=\s*)?(\d+)\s*(?:unit[ée]s?(?:\s+de\s+temps)?|heures?|h)?\s*\.?\s*$", I)
_BARE_PAIR = re.compile(r"^\D*(\d+)\D+(\d+)\D*$")

HELP = ("Je peux simuler une perturbation sur votre planning et comparer **avant / après** :\n"
        "- une **panne de machine** (ex. « panne de M2 pendant les deux premières heures »)\n"
        "- un **retard de livraison d'un composant** (ex. « le composant C1 arrive au temps 6 »)\n\n"
        "Si une information manque, je vous la demande.")
NO_DATA = ("Je n'ai pas encore de données de production. Chargez un fichier **JSON/CSV** ou cliquez sur "
           "**Charger l'exemple** dans la barre latérale : je ne peux pas calculer un planning sans jobs ni durées, "
           "et je n'invente rien.")


def build_context(problem: Dict[str, Any]) -> Dict[str, Any]:
    comps = dict(problem.get("component_availability", {}))
    usage = {c: sorted({j["job_id"] for j in problem["jobs"] for op in j["operations"]
                        if c in op.get("components", [])}) for c in comps}
    return {"machines": list(problem["machines"]), "components": comps, "usage": usage,
            "jobs": [j["job_id"] for j in problem["jobs"]]}


def detect_intent(text: str) -> Optional[str]:
    if _COMPONENT_INTENT.search(text):
        return "component_delay"
    if _MACHINE_INTENT.search(text):
        return "machine_breakdown"
    return None


def find_known(text: str, ids) -> Optional[str]:
    """Earliest mention of a known identifier (case-insensitive, whole token)."""
    best = None
    for i in ids:
        m = re.search(rf"(?<![\w]){re.escape(i)}(?![\w])", text, I)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), i)
    return best[1] if best else None


def _reply(text, pending=None, run=None):
    return {"reply": text, "pending": pending or {}, "run": run}


def _fill_machine(text, slots, ctx, notes):
    warn = ""
    m = find_known(text, ctx["machines"])
    if m:
        slots["machine"] = m
    else:
        g = _MACHINE_GENERIC.search(text)
        if g and "machine" not in slots:
            warn = f"La machine « M{g.group(1)} » n'existe pas dans vos données. "
    iv = parse_interval(text)
    if iv is None and slots.get("machine") and "start" not in slots and not _MACHINE_INTENT.search(text):
        pair = _BARE_PAIR.match(text)
        if pair and int(pair.group(2)) > int(pair.group(1)):
            iv = ((int(pair.group(1)), int(pair.group(2))), None)
    if iv:
        (slots["start"], slots["end"]), note = iv
        if note:
            notes.append(note)
    if "machine" not in slots:
        return warn + f"Quelle **machine** est en panne ? Machines de vos données : {', '.join(ctx['machines'])}."
    if "start" not in slots:
        return (f"{warn}Panne de **{slots['machine']}** : sur quel **intervalle de temps** ? "
                "(ex. « de 0 à 2 », « [0, 2) » ou « les 2 premières heures »)")
    return None


def _fill_component(text, slots, ctx, notes):
    comps = ctx["components"]
    warn = ""
    c = find_known(text, comps)
    if c:
        slots["component"] = c
    else:
        g = _COMPONENT_GENERIC.search(text)
        if g and "component" not in slots:
            warn = f"Le composant « {g.group(1)} » n'existe pas dans vos données. "

    d = _DATE.search(text)
    dl = _DELAY.search(text)
    if d:
        slots["new_date"] = int(d.group(1))
    elif dl:
        slots["delay"] = int(dl.group(1) or dl.group(2))
    elif "new_date" not in slots and "delay" not in slots and slots.get("component"):
        b = _BARE_NUM.match(text)
        if b:
            slots["new_date"] = int(b.group(1))

    if "component" not in slots:
        listing = "\n".join(f"- **{k}** (utilisé par {', '.join(v) if v else 'aucune commande'})"
                            for k, v in ctx["usage"].items())
        return f"{warn}Quel **composant** est concerné ? Composants de vos données :\n{listing}"
    if "new_date" not in slots and "delay" not in slots:
        base = comps[slots["component"]]
        return (f"Le composant **{slots['component']}** est actuellement disponible à t={base}. "
                "À quelle **date (unité de temps)** arrive-t-il maintenant ? "
                "Vous pouvez aussi dire « retard de 3 unités ».")
    return None


def handle_message(message: str, pending: Optional[Dict[str, Any]], ctx: Optional[Dict[str, Any]],
                   llm=None, last_facts: str = "") -> Dict[str, Any]:
    """Returns {"reply": str, "pending": dict, "run": scenario|None}."""
    text = (message or "").strip()
    pending = dict(pending or {})

    if pending and _CANCEL.search(text):
        return _reply("D'accord, j'annule cette simulation. Que voulez-vous simuler ?")
    if ctx is None:
        return _reply(NO_DATA)

    intent = detect_intent(text)
    if intent is None and pending:
        intent = pending.get("type")
    elif intent and pending and pending.get("type") != intent:
        pending = {}

    if intent is None:
        if llm is not None and last_facts:
            answer = answer_question(llm, text, last_facts)
            if answer:
                return _reply(answer)
        return _reply(HELP)

    slots = dict(pending)
    slots["type"] = intent
    notes: List[str] = list(slots.get("notes", []))

    if intent == "component_delay" and not ctx["components"]:
        return _reply("Votre jeu de données ne décrit **aucun composant** (colonne `components` en CSV, ex. `C1|C2`, "
                      "ou clé `components` en JSON). Sans cette information je ne peux pas simuler ce retard, "
                      "et je n'invente pas de composants.")

    question = (_fill_machine if intent == "machine_breakdown" else _fill_component)(text, slots, ctx, notes)
    slots["notes"] = list(dict.fromkeys(notes))
    if question:
        return _reply(question, pending=slots)

    if intent == "machine_breakdown":
        scenario = {"type": intent, "machine": slots["machine"], "start": slots["start"], "end": slots["end"],
                    "notes": slots["notes"]}
        msg = f"Compris : panne de **{scenario['machine']}** sur **[{scenario['start']}, {scenario['end']})**."
    else:
        base = ctx["components"][slots["component"]]
        new_date = slots["new_date"] if "new_date" in slots else base + slots["delay"]
        scenario = {"type": intent, "component": slots["component"], "new_date": new_date, "notes": slots["notes"]}
        msg = f"Compris : le composant **{scenario['component']}** n'arrivera qu'à **t={new_date}** (au lieu de t={base})."
    return _reply(msg + " Je recalcule le planning avec OR-Tools…", run=scenario)


_UNIT_WORDS = re.compile(r"\b(jours?|heures?|semaines?|minutes?|journ[ée]es?)\b", I)


def normalize_units(text: str) -> str:
    """The data has no time unit: force the neutral wording whatever the LLM wrote."""
    def repl(m):
        return "unités de temps" if m.group(1).lower().endswith("s") else "unité de temps"
    return _UNIT_WORDS.sub(repl, text)


UNIT_RULE = ("Les durées et dates sont en « unités de temps » (l'unité réelle n'est PAS connue). "
             "N'écris jamais jours, heures, semaines ni minutes.")


def answer_question(llm, question: str, facts: str) -> str:
    """Answer a follow-up question strictly from computed facts. Returns '' on failure."""
    from src.utils.llm import call_llm
    prompt = ("Tu es l'assistant de planification d'une chaîne d'approvisionnement. Réponds en français, "
              "brièvement, UNIQUEMENT à partir des FAITS ci-dessous. Si l'information n'y figure pas, "
              "dis-le explicitement et n'invente rien. " + UNIT_RULE + "\n\nFAITS:\n" + facts + f"\n\nQUESTION: {question}\n\nRÉPONSE:")
    try:
        return normalize_units(call_llm(llm, prompt).strip())
    except Exception as e:
        return f"(Le modèle de langage n'a pas pu répondre : {e})"


def commentary(llm, summary_md: str) -> str:
    """Short LLM comment on the computed report. Returns '' on failure."""
    from src.utils.llm import call_llm
    prompt = ("Voici le résultat chiffré d'une simulation de perturbation sur un planning de production. "
              "Rédige en français 3 à 4 phrases expliquant l'impact sur la production et les livraisons. "
              "N'utilise QUE les chiffres ci-dessous et n'en invente aucun. " + UNIT_RULE + " "
              "Ne liste pas de données manquantes supposées (coûts, capacités, mesures d'atténuation...) : "
              "seuls les avertissements ⚠️ du rapport comptent.\n\n"
              + summary_md + "\n\nANALYSE:")
    try:
        return normalize_units(call_llm(llm, prompt).strip())
    except Exception:
        return ""
