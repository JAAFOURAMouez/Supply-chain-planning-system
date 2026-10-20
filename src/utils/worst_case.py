"""
Worst-case analysis: try EVERY disruption of a given duration and rank them with the OR-Tools solver.

  - machine breakdown : each machine x each start time t -> machine unavailable on [t, t+duration)
  - component delay   : each component delayed by `duration` time units

Nothing is estimated: every row of the ranking is a full re-solve of the schedule.
"""
import copy
from typing import Any, Dict, List

from src.agents.planning_agent import create_planning_agent
from src.models.supply_chain import SupplyChainProblem


def _lateness(problem, result) -> int:
    return sum(max(0, result.job_completion_times[j.job_id] - j.due_date)
               for j in problem.jobs if j.due_date is not None and j.job_id in result.job_completion_times)


def analyse(problem_dict: Dict[str, Any], duration: int, scope: str = "all", max_runs: int = 400) -> Dict[str, Any]:
    """scope: 'machine' | 'component' | 'all'. Returns {"ref_makespan", "ref_lateness", "runs", "rows"} (rows: worst first)."""
    agent = create_planning_agent()
    base = SupplyChainProblem(**copy.deepcopy(problem_dict))
    ref = agent.solve_job_shop(base)
    if ref.status not in ("OPTIMAL", "FEASIBLE"):
        return {"ref_makespan": None, "ref_lateness": None, "runs": 0, "rows": [], "error": "Pas de planning de référence faisable."}
    ref_late = _lateness(base, ref)

    cands: List[Dict[str, Any]] = []
    if scope in ("machine", "all"):
        horizon = ref.makespan
        step = max(1, (len(base.machines) * horizon) // max_runs)  # keep the number of solves bounded
        for m in base.machines:
            for t in range(0, horizon, step):
                cands.append({"type": "machine_breakdown", "machine": m, "start": t, "end": t + duration})
    if scope in ("component", "all"):
        for c, avail in base.component_availability.items():
            cands.append({"type": "component_delay", "component": c, "new_date": avail + duration})

    rows = []
    for scn in cands:
        if scn["type"] == "machine_breakdown":
            av = {m: [list(i) for i in iv] for m, iv in base.machine_availability.items()}
            av.setdefault(scn["machine"], []).append([scn["start"], scn["end"]])
            p = base.model_copy(update={"machine_availability": av})
        else:
            comp = dict(base.component_availability)
            comp[scn["component"]] = scn["new_date"]
            p = base.model_copy(update={"component_availability": comp,
                                        "baseline_component_availability": dict(base.component_availability)})
        r = agent.solve_job_shop(p)
        ok = r.status in ("OPTIMAL", "FEASIBLE")
        rows.append({
            "scenario": scn, "status": r.status,
            "makespan": r.makespan if ok else None,
            "delta": (r.makespan - ref.makespan) if ok else None,
            "total_lateness": _lateness(p, r) if ok else None,
            "late_jobs": sorted(j.job_id for j in p.jobs if ok and j.due_date is not None
                                and r.job_completion_times.get(j.job_id, 0) > j.due_date),
        })
    # infeasible first (worst), then biggest makespan, then total lateness
    rows.sort(key=lambda x: (x["makespan"] is not None, -(x["makespan"] or 0), -(x["total_lateness"] or 0)))
    return {"ref_makespan": ref.makespan, "ref_lateness": ref_late, "runs": len(rows), "rows": rows}


def describe(scn: Dict[str, Any]) -> str:
    if scn["type"] == "machine_breakdown":
        return f"panne de {scn['machine']} sur [{scn['start']}, {scn['end']})"
    return f"retard du composant {scn['component']} jusqu'à t={scn['new_date']}"


def markdown(res: Dict[str, Any], duration: int, top: int = 5) -> str:
    if not res["rows"]:
        return "⚠️ " + res.get("error", "Aucune perturbation testée.")
    best = res["rows"][0]
    lines = [f"**Analyse du pire cas** pour une perturbation de **{duration} unités de temps** : "
             f"{res['runs']} scénarios testés avec OR-Tools (makespan de référence = {res['ref_makespan']}).", ""]
    if best["delta"] == 0 and best["status"] in ("OPTIMAL", "FEASIBLE"):
        lines.append("Aucune perturbation de cette durée n'allonge le makespan.")
    else:
        w = "planning infaisable" if best["makespan"] is None else f"makespan {res['ref_makespan']} → {best['makespan']} ({best['delta']:+d})"
        lines.append(f"**Pire cas : {describe(best['scenario'])}** — {w}.")
    lines += ["", "| # | Perturbation | Makespan | Écart | Commandes en retard |", "|---|---|---|---|---|"]
    for i, r in enumerate(res["rows"][:top], 1):
        lines.append(f"| {i} | {describe(r['scenario'])} | {r['makespan'] if r['makespan'] is not None else 'infaisable'} | "
                     f"{'' if r['delta'] is None else format(r['delta'], '+d')} | {', '.join(r['late_jobs']) or 'aucune'} |")
    lines.append("\nℹ️ Toutes les valeurs sont en **unités de temps** : l'unité réelle (heure, jour…) n'est pas précisée dans les données.")
    return "\n".join(lines)
