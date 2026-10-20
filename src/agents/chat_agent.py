"""
Conversational front-end = a small tool-routing agent.

For every message the agent first decides WHICH action is best (see ACTIONS):
  - machine_breakdown / component_delay : tools that need the OR-Tools simulator (slot filling for what is missing)
  - show_baseline                       : tool that shows the reference schedule + Gantt (no disruption)
  - answer                              : no tool fits -> the LLM answers like a plain assistant, from the data it has
  - cancel                              : drop the pending question

The decision is made by the LLM when one is configured (structured JSON output), otherwise by rules.
Whatever the LLM proposes is validated against the loaded dataset (known machines / components, numbers that
really appear in the message), so nothing is ever guessed. Results are always computed by the solver.
"""
import json
import re
from typing import Any, Dict, List, Optional

from src.utils.disruptions import _NUM, _NUM_WORDS, _to_int, parse_interval

I = re.IGNORECASE

_COMPONENT_INTENT = re.compile(
    r"composant|component|mati[èe]re|approvisionnement|fournisseur|p[ée]nurie|rupture|shortage|"
    r"retard\w*\s+(?:de\s+)?livraison|livraison\s+(?:en\s+)?retard", I)
_MACHINE_INTENT = re.compile(r"panne|indisponib|breakdown|downtime|maintenance|arr[êe]t\s+(?:de\s+la\s+)?machine", I)
_CANCEL = re.compile(r"\b(annule|annuler|laisse\s+tomber|oublie|stop|cancel)\b", I)
# "sans panne", "sans aucune perturbation"... : the disruption keyword is NEGATED, it must not trigger a simulation
_NEGATED = re.compile(r"\b(?:sans|hors|aucune?|without|no)\s+(?:aucune?\s+|la\s+|de\s+|d['’]\s*|any\s+)?"
                      r"(?:panne|perturbation|retard|indisponib\w*|incident|probl[èe]me|arr[êe]t|breakdown|disruption|delay)s?\b", I)
_BASELINE_WORDS = re.compile(r"gantt|diagramme|planning|ordonnancement|calendrier|schedule|r[ée]f[ée]rence|nominal|initial|baseline", I)
_SHOW = re.compile(r"affich|montr|voir|visualis|donne|g[ée]n[èe]re|trace|show|display|plot", I)
_WORST = re.compile(r"\b(?:pire|pires|worst|plus\s+(?:grave|critique|dangereu\w+|impactant\w*|p[ée]nalisant\w*|d[ée]favorable|d[ée]sastreu\w+)|"
                    r"cas\s+extr[êe]me|sc[ée]nario\s+catastrophe|maximum\s+de\s+(?:perturbation|retard|d[ée]g[âa]t))", I)
_DURATION = re.compile(r"(?:pendant|durant|de\s+dur[ée]e|dur[ée]e\s+de|for|of|sur|d['’]une?)?\s*" + _NUM +
                       r"\s*(?:unit[ée]s?(?:\s+de\s+temps)?|heures?|hours?|h\b)", I)
_SCOPE_MACHINE = re.compile(r"machine|panne|indisponib|breakdown|\bM\d+\b", I)
_SCOPE_COMPONENT = re.compile(r"composant|component|livraison|fournisseur|approvisionnement|mati[èe]re", I)
ACTIONS = ("machine_breakdown", "component_delay", "worst_case", "show_baseline", "answer", "cancel")

_MACHINE_GENERIC = re.compile(r"\b(?:machine\s*|M)(\d+)\b", I)
_COMPONENT_GENERIC = re.compile(r"composant\s+(?:n[°o]\s*)?([A-Za-z]*\d[\w\-]*)", I)
_DATE = re.compile(
    r"(?:jusqu['’]?\s*(?:à|a|au|en)|arriv\w*\s+(?:à|a|au|en|vers)|disponible\s+(?:à partir de|a partir de|à|a|au|en)"
    r"|nouvelle\s+date\w*\s*(?:d['’]arriv\w+)?\s*(?:est|:|=|à|au)?|(?:au\s+)?temps|\bà\s+t\s*=|\bt\s*=)"
    r"\s*(?:t\s*=\s*)?(\d+)", I)
_DELAY = re.compile(r"(?:retard|d[ée]cal\w+|report\w*)\s+de\s+(\d+)|(\d+)\s+(?:unit[ée]s?(?:\s+de\s+temps)?|heures?)\s+de\s+retard", I)
_BARE_NUM = re.compile(r"^\s*(?:t\s*=\s*)?(\d+)\s*(?:unit[ée]s?(?:\s+de\s+temps)?|heures?|h)?\s*\.?\s*$", I)
_BARE_PAIR = re.compile(r"^\D*(\d+)\D+(\d+)\D*$")

HELP = ("Je peux :\n"
        "- **afficher le planning de référence** et son Gantt (ex. « montre le Gantt sans panne »)\n"
        "- simuler une **panne de machine** (ex. « panne de M2 pendant les deux premières heures »)\n"
        "- simuler un **retard de livraison d'un composant** (ex. « le composant C1 arrive au temps 6 »)\n"
        "- chercher le **pire cas** (ex. « quelle est la pire perturbation de 2 heures ? ») : je teste toutes les pannes/retards possibles\n"
        "- répondre à vos questions sur vos données (avec une clé API LLM)\n\n"
        "Si une information manque, je vous la demande.")
NO_DATA = ("Je n'ai pas encore de données de production. Chargez un fichier **JSON/CSV** ou cliquez sur "
           "**Charger l'exemple** dans la barre latérale : je ne peux pas calculer un planning sans jobs ni durées, "
           "et je n'invente rien.")


def build_context(problem: Dict[str, Any]) -> Dict[str, Any]:
    comps = dict(problem.get("component_availability", {}))
    usage = {c: sorted({j["job_id"] for j in problem["jobs"] for op in j["operations"]
                        if c in op.get("components", [])}) for c in comps}
    lines = [f"Machines : {', '.join(problem['machines'])}.",
             "Composants (disponible à partir de t=) : " + (", ".join(f"{c}={t}" for c, t in comps.items()) or "aucun") + "."]
    for j in problem["jobs"]:
        ops = " → ".join(f"{o['operation_id']} sur {o['machine_id']} (durée {o['duration']}"
                         + (f", composants {'/'.join(o['components'])}" if o.get("components") else "") + ")"
                         for o in sorted(j["operations"], key=lambda o: o["order_index"]))
        lines.append(f"Commande {j['job_id']} (échéance {j.get('due_date', 'inconnue')}, priorité {j.get('priority', 1)}) : {ops}")
    return {"machines": list(problem["machines"]), "components": comps, "usage": usage,
            "jobs": [j["job_id"] for j in problem["jobs"]], "facts": "\n".join(lines)}


def detect_intent(text: str) -> Optional[str]:
    text = _NEGATED.sub(" ", text)  # "sans panne" is not a breakdown
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


# --------------------------------------------------------------------------- routing (which action?)
ROUTER_PROMPT = """Tu es le routeur d'un assistant de planification de production (ordonnancement job shop, solveur OR-Tools).
Comprends le message de l'utilisateur (langage naturel) et choisis la MEILLEURE action. Réponds UNIQUEMENT par un objet JSON.

ACTIONS (outils disponibles) :
- "show_baseline"     : afficher le planning de référence / diagramme de Gantt SANS aucune perturbation
                        (ex. « montre le Gantt sans panne », « affiche le planning », « planning normal »).
                        ATTENTION : « sans panne », « sans retard » = PAS de perturbation -> show_baseline, jamais machine_breakdown.
- "machine_breakdown" : simuler une panne / indisponibilité d'une machine. Champs : machine, start, end.
- "component_delay"   : simuler le retard de livraison d'un composant. Champs : component, et new_date OU delay.
- "worst_case"        : ANALYSE qui demande un calcul : pire perturbation / scénario le plus critique / machine ou composant le
                        plus sensible, pour une durée donnée (ex. « quel est le pire cas d'une perturbation de 2 heures ? »).
                        Champs : duration (entier), scope ("machine", "component" ou "all" si non précisé).
- "answer"            : aucun outil n'est nécessaire (question sur les données, les durées, les résultats déjà calculés,
                        explication, salutation...). Un autre module répondra à partir des données.
- "cancel"            : l'utilisateur abandonne la simulation en cours.

RÈGLES : n'invente JAMAIS une valeur. Si l'utilisateur ne l'a pas donnée, mets null.
machine doit être EXACTEMENT l'une des machines connues, component l'un des composants connus.
Les temps sont des entiers en unités de temps. « les N premières heures/unités » => start 0, end N.
Si une question est en attente (ci-dessous), le message est très probablement la réponse à cette question.

DONNÉES CHARGÉES :
{facts}

SIMULATION EN ATTENTE (valeurs déjà connues) : {pending}
UN RÉSULTAT A DÉJÀ ÉTÉ CALCULÉ : {has_result}

MESSAGE : {message}

JSON (clés : action, machine, start, end, component, new_date, delay, duration, scope) :"""


def _num_in_text(n: int, text: str) -> bool:
    """A number given by the LLM is only trusted if the user really wrote it (digits or French/English word)."""
    if str(n) in re.findall(r"\d+", text):
        return True
    return any(v == n and not k.endswith("_") and re.search(rf"\b{k}\b", text, I) for k, v in _NUM_WORDS.items())


def route_with_llm(llm, text: str, pending: Dict[str, Any], ctx: Dict[str, Any], has_result: bool) -> Optional[Dict[str, Any]]:
    """Ask the LLM which action to take. Returns None if it fails / answers garbage (-> rule-based fallback)."""
    from src.utils.llm import call_llm
    shown = {k: v for k, v in pending.items() if k not in ("notes",)}
    prompt = ROUTER_PROMPT.format(facts=ctx["facts"], pending=json.dumps(shown, ensure_ascii=False) if shown else "aucune",
                                  has_result="oui" if has_result else "non", message=text)
    try:
        raw = call_llm(llm, prompt)
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("action") not in ACTIONS:
        return None
    return data


def route_with_rules(text: str, pending: Dict[str, Any]) -> Dict[str, Any]:
    """Offline fallback (no LLM): keyword rules, with negation handling."""
    if pending and _CANCEL.search(text):
        return {"action": "cancel"}
    if _WORST.search(text):
        return {"action": "worst_case"}
    intent = detect_intent(text)
    if intent:
        return {"action": intent}
    if _NEGATED.search(text) or (_BASELINE_WORDS.search(text) and _SHOW.search(text)):
        return {"action": "show_baseline"}
    if pending:
        return {"action": pending["type"]}
    return {"action": "answer"}


def _llm_slots(route: Dict[str, Any], slots: Dict[str, Any], text: str, ctx: Dict[str, Any]) -> None:
    """Merge the slots proposed by the LLM, after validation against the dataset and the user's own words."""
    def as_int(v):
        return v if isinstance(v, int) and not isinstance(v, bool) else None
    by_lower = {m.lower(): m for m in ctx["machines"]}
    comp_lower = {c.lower(): c for c in ctx["components"]}
    m = by_lower.get(str(route.get("machine") or "").lower())
    if m:
        slots["machine"] = m
    c = comp_lower.get(str(route.get("component") or "").lower())
    if c:
        slots["component"] = c
    st, en = as_int(route.get("start")), as_int(route.get("end"))
    if st is not None and en is not None and en > st and _num_in_text(en, text) and (st == 0 or _num_in_text(st, text)):
        slots["start"], slots["end"] = st, en
    du = as_int(route.get("duration"))
    if du is not None and du > 0 and _num_in_text(du, text):
        slots["duration"] = du
    if route.get("scope") in ("machine", "component", "all"):
        slots["scope"] = route["scope"]
    nd, dl = as_int(route.get("new_date")), as_int(route.get("delay"))
    if nd is not None and nd >= 0 and _num_in_text(nd, text):
        slots["new_date"] = nd
    elif dl is not None and dl > 0 and _num_in_text(dl, text):
        slots["delay"] = dl


def _fill_worst(text, slots, ctx, notes):
    m = _DURATION.search(text)
    if m:
        slots["duration"] = _to_int(m.group(1))
        if re.search(r"heure|hour|\bh\b", m.group(0), I):
            notes.append(f"Hypothèse: « {m.group(1)} heures » interprété comme {slots['duration']} unités de temps "
                         "(1 heure = 1 unité). Les durées des données n'ont pas d'unité précisée: à confirmer.")
    elif "duration" not in slots:
        b = _BARE_NUM.match(text)
        if b:
            slots["duration"] = int(b.group(1))
    machine, comp = _SCOPE_MACHINE.search(text), _SCOPE_COMPONENT.search(text)
    if machine and not comp:
        slots["scope"] = "machine"
    elif comp and not machine:
        slots["scope"] = "component"
    if slots.get("scope") == "component" and not ctx["components"]:
        slots["scope"] = "machine"
        notes.append("Aucun composant dans les données : seules les pannes de machines sont analysées.")
    if "duration" not in slots:
        return "Quelle **durée de perturbation** (en unités de temps) dois-je considérer ? (ex. « 2 »)"
    return None


def handle_message(message: str, pending: Optional[Dict[str, Any]], ctx: Optional[Dict[str, Any]],
                   llm=None, last_facts: str = "") -> Dict[str, Any]:
    """Returns {"reply": str, "pending": dict, "run": scenario|None}.

    `run` is a scenario for the simulator: machine_breakdown, component_delay or {"type": "baseline"}.
    `run is None` means no tool is needed (question to ask, or plain LLM answer).
    """
    text = (message or "").strip()
    pending = dict(pending or {})

    if ctx is None:
        if pending and _CANCEL.search(text):
            return _reply("D'accord, j'annule cette simulation. Que voulez-vous simuler ?")
        return _reply(NO_DATA)

    # 1. Which action? LLM first (natural language), rules as a safety net
    route = route_with_llm(llm, text, pending, ctx, bool(last_facts)) if llm is not None else None
    if route is None:
        route = route_with_rules(text, pending)
    action = route["action"]

    if action == "cancel":
        return _reply("D'accord, j'annule cette simulation. Que voulez-vous simuler ?")

    # 2. Tool: reference schedule + Gantt, no disruption
    if action == "show_baseline":
        return _reply("Compris : je calcule et j'affiche le **planning de référence** (sans aucune perturbation) "
                      "avec OR-Tools…", run={"type": "baseline", "notes": []})

    # 3. No tool needed: answer like a plain LLM, only from the data we hold
    if action == "answer":
        if llm is not None:
            answer = answer_question(llm, text, ctx["facts"] + ("\n\nDERNIER RÉSULTAT CALCULÉ :\n" + last_facts if last_facts else ""))
            if answer:
                return _reply(answer, pending=pending)  # a pending question stays open
        return _reply(HELP, pending=pending)

    # 4. Tool: disruption simulation (slot filling for what is missing)
    intent = action
    if pending and pending.get("type") != intent:
        pending = {}
    slots = dict(pending)
    slots["type"] = intent
    notes: List[str] = list(slots.get("notes", []))
    _llm_slots(route, slots, text, ctx)

    if intent == "component_delay" and not ctx["components"]:
        return _reply("Votre jeu de données ne décrit **aucun composant** (colonne `components` en CSV, ex. `C1|C2`, "
                      "ou clé `components` en JSON). Sans cette information je ne peux pas simuler ce retard, "
                      "et je n'invente pas de composants.")

    fill = {"machine_breakdown": _fill_machine, "component_delay": _fill_component, "worst_case": _fill_worst}[intent]
    question = fill(text, slots, ctx, notes)
    slots["notes"] = list(dict.fromkeys(notes))
    if question:
        return _reply(question, pending=slots)

    if intent == "worst_case":
        scope = slots.get("scope", "all")
        what = {"machine": "pannes de machines", "component": "retards de composants", "all": "pannes de machines et retards de composants"}[scope]
        scenario = {"type": intent, "duration": slots["duration"], "scope": scope, "notes": slots["notes"]}
        return _reply(f"Compris : je cherche le **pire cas** d'une perturbation de **{slots['duration']} unités de temps** "
                      f"en testant toutes les {what} possibles avec OR-Tools…", run=scenario)
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
    """Answer a question strictly from the dataset + computed facts (plain LLM, no tool). Returns '' on failure."""
    from src.utils.llm import call_llm
    prompt = ("Tu es l'assistant de planification d'une chaîne d'approvisionnement. Réponds en français, "
              "brièvement, UNIQUEMENT à partir des FAITS ci-dessous (données de production et éventuel dernier résultat). "
              "Si l'information n'y figure pas, "
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
