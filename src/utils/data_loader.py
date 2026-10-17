"""
Load a production dataset (jobs / operations / machines) from a JSON or CSV file.

JSON format:
  {"machines": ["M1","M2"],            # optional, deduced from operations if absent
   "jobs": [{"job_id": "J1", "due_date": 10, "priority": 3,
             "operations": [{"operation_id": "O11", "machine_id": "M1", "duration": 3, "order_index": 0}]}]}

CSV columns (header required):
  job_id, operation_id, machine_id, duration  [, order_index, due_date, priority]
  order_index defaults to the row order inside each job.
  Optional column `components`: components needed by the operation, separated by | (e.g. C1|C2).
  In JSON: "components": ["C1"] on an operation, and optional top-level "components": {"C1": 0} (id -> available from).
"""
import csv
import io
import json
import re
from typing import Any, Dict, List


class DataLoadError(ValueError):
    pass


def _finalize(jobs: List[Dict[str, Any]], machines: List[str], components: Dict[str, int] = None) -> Dict[str, Any]:
    if not jobs:
        raise DataLoadError("Aucun job trouvé dans le fichier.")
    seen = list(dict.fromkeys(machines))
    for job in jobs:
        for op in job["operations"]:
            if op["machine_id"] not in seen:
                seen.append(op["machine_id"])
    comps = {str(c): int(v) for c, v in (components or {}).items()}
    for job in jobs:
        for op in job["operations"]:
            for c in op.get("components", []):
                comps.setdefault(c, 0)
    return {"machines": seen, "jobs": jobs, "component_availability": comps}


def load_json(content: bytes) -> Dict[str, Any]:
    try:
        data = json.loads(content.decode("utf-8-sig"))
    except Exception as e:
        raise DataLoadError(f"JSON invalide: {e}")
    if not isinstance(data, dict) or "jobs" not in data:
        raise DataLoadError("Le JSON doit contenir une clé 'jobs'.")
    jobs = []
    for j in data["jobs"]:
        ops = []
        for i, op in enumerate(j.get("operations", [])):
            try:
                ops.append({
                    "operation_id": str(op.get("operation_id", f"{j['job_id']}_{i}")),
                    "job_id": str(j["job_id"]),
                    "machine_id": str(op["machine_id"]),
                    "duration": int(op["duration"]),
                    "order_index": int(op.get("order_index", i)),
                    "components": [str(c) for c in op.get("components", [])],
                })
            except KeyError as e:
                raise DataLoadError(f"Champ manquant {e} dans une opération du job {j.get('job_id')}")
        jobs.append({"job_id": str(j["job_id"]), "operations": ops,
                     "due_date": j.get("due_date"), "priority": int(j.get("priority", 1))})
    comps = data.get("components", {})
    if isinstance(comps, list):
        comps = {str(c): 0 for c in comps}
    return _finalize(jobs, [str(m) for m in data.get("machines", [])], comps)


def load_csv(content: bytes) -> Dict[str, Any]:
    text = content.decode("utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=",;\t")
    except Exception:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    required = {"job_id", "operation_id", "machine_id", "duration"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise DataLoadError(f"Colonnes manquantes dans le CSV: {sorted(missing)}")
    jobs: Dict[str, Dict[str, Any]] = {}
    for row in reader:
        jid = row["job_id"].strip()
        job = jobs.setdefault(jid, {"job_id": jid, "operations": [], "due_date": None, "priority": 1})
        if row.get("due_date", "").strip():
            job["due_date"] = int(row["due_date"])
        if row.get("priority", "").strip():
            job["priority"] = int(row["priority"])
        idx = row.get("order_index", "").strip()
        job["operations"].append({
            "operation_id": row["operation_id"].strip(), "job_id": jid,
            "machine_id": row["machine_id"].strip(), "duration": int(row["duration"]),
            "order_index": int(idx) if idx else len(job["operations"]),
            "components": [c.strip() for c in re.split(r"[|;]", row.get("components", "") or "") if c.strip()],
        })
    return _finalize(list(jobs.values()), [])


def load_file(filename: str, content: bytes) -> Dict[str, Any]:
    if filename.lower().endswith(".json"):
        return load_json(content)
    if filename.lower().endswith((".csv", ".txt")):
        return load_csv(content)
    raise DataLoadError("Format non supporté (JSON ou CSV attendu).")
