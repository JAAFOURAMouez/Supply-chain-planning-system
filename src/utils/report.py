"""Deterministic French report for a simulated scenario (no LLM: every number comes from the solver)."""
from typing import Any, Dict


def describe_scenario(scn: Dict[str, Any], baseline: Dict[str, int] = None) -> str:
    if scn["type"] == "machine_breakdown":
        return f"panne de la machine **{scn['machine']}** sur l'intervalle **[{scn['start']}, {scn['end']})**"
    base = (baseline or {}).get(scn["component"], 0)
    return (f"retard du composant **{scn['component']}** : disponible à **t={scn['new_date']}** "
            f"(au lieu de t={base})")


def _fmt(items):
    return ", ".join(items) if items else "aucune"


def summary_markdown(problem, scn, ref, pert, impact) -> str:
    baseline = problem.baseline_component_availability or {}
    lines = [f"**Scénario simulé** : {describe_scenario(scn, baseline)}.", ""]
    lines.append(f"**Statut du solveur (OR-Tools CP-SAT)** : avant = `{ref.status}`, après = `{pert.status}`")
    if impact is None:
        lines.append("\n⚠️ Pas de planning faisable dans au moins un des deux scénarios : comparaison impossible.")
        lines.append("\nℹ️ Toutes les valeurs sont en **unités de temps** : l'unité réelle (heure, jour…) n'est pas précisée dans les données.")
        return "\n".join(lines)
    d = impact["makespan_pert"] - impact["makespan_ref"]
    lines.append(f"\n**Makespan** : {impact['makespan_ref']} → {impact['makespan_pert']} ({d:+d})")
    lines.append(f"\n**Commandes impactées** (date de fin modifiée) : "
                 + _fmt([f"{r['job_id']} ({r['delta']:+d}{'' if r['concerned'] else ', indirect: conflit de ressources'})"
                         for r in impact["jobs"] if r["impacted"]]))
    lines.append(f"\n**Commandes concernées mais non retardées** (marge suffisante) : {_fmt(impact['concerned_not_delayed'])}")
    before = [f"{r['job_id']} (+{r['ref_lateness']})" for r in impact["jobs"] if r["ref_lateness"]]
    after = [f"{r['job_id']} (+{r['pert_lateness']})" for r in impact["jobs"] if r["pert_lateness"]]
    lines.append(f"\n**Retards de livraison avant** : {_fmt(before)}")
    lines.append(f"\n**Retards de livraison après** : {_fmt(after)}")
    if any(r["due_date"] is None for r in impact["jobs"]):
        lines.append("\n⚠️ Donnée manquante : certaines commandes n'ont pas de date de livraison, leur retard n'est pas calculé.")
    lines.append("\nℹ️ Toutes les valeurs sont en **unités de temps** : l'unité réelle (heure, jour…) n'est pas précisée dans les données.")
    return "\n".join(lines)


def baseline_markdown(problem, ref) -> str:
    """Deterministic report of the reference schedule (no disruption)."""
    lines = ["**Planning de référence** (aucune perturbation).", "",
             f"**Statut du solveur (OR-Tools CP-SAT)** : `{ref.status}`"]
    if ref.status not in ("OPTIMAL", "FEASIBLE"):
        lines.append("\n⚠️ Aucun planning faisable avec ces données.")
        return "\n".join(lines)
    lines.append(f"\n**Makespan** : {ref.makespan}")
    late = []
    for j in problem.jobs:
        c = ref.job_completion_times.get(j.job_id)
        if c is not None and j.due_date is not None and c > j.due_date:
            late.append(f"{j.job_id} (+{c - j.due_date})")
    lines.append(f"\n**Commandes en retard de livraison** : {_fmt(late)}")
    if any(j.due_date is None for j in problem.jobs):
        lines.append("\n⚠️ Donnée manquante : certaines commandes n'ont pas de date de livraison, leur retard n'est pas calculé.")
    lines.append("\nℹ️ Toutes les valeurs sont en **unités de temps** : l'unité réelle (heure, jour…) n'est pas précisée dans les données.")
    return "\n".join(lines)
