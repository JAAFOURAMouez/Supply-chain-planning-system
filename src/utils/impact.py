"""Before/after comparison of two schedules: which orders (jobs) are impacted, with start/end times."""
from typing import Any, Dict, Optional


def _late(completion: Optional[int], due: Optional[int]) -> Optional[int]:
    if completion is None or due is None:
        return None
    return max(0, completion - due)


def build_impact(problem, scenario: Optional[Dict[str, Any]], ref, pert) -> Dict[str, Any]:
    scenario = scenario or {}
    component = scenario.get("component")
    machine = scenario.get("machine")

    rows = []
    impacted, concerned_only = [], []
    for job in problem.jobs:
        concerned = bool(
            (component and any(component in op.components for op in job.operations))
            or (machine and any(op.machine_id == machine for op in job.operations))
        )
        rc = ref.job_completion_times.get(job.job_id)
        pc = pert.job_completion_times.get(job.job_id)
        delta = None if rc is None or pc is None else pc - rc
        row = {
            "job_id": job.job_id,
            "due_date": job.due_date,
            "ref_completion": rc,
            "pert_completion": pc,
            "delta": delta,
            "ref_lateness": _late(rc, job.due_date),
            "pert_lateness": _late(pc, job.due_date),
            "concerned": concerned,
            "impacted": bool(delta),
        }
        rows.append(row)
        if row["impacted"]:
            impacted.append(job.job_id)
        elif concerned:
            concerned_only.append(job.job_id)

    def op_times(result):
        out = {}
        for machine_id, ops in result.schedule.items():
            for o in ops:
                out[(o["job_id"], o["operation_id"])] = (machine_id, o["start"], o["end"])
        return out

    ref_ops, pert_ops = op_times(ref), op_times(pert)
    operations = []
    for (job_id, op_id), (m, s, e) in sorted(ref_ops.items(), key=lambda kv: (kv[0][0], kv[1][1])):
        ps, pe = pert_ops.get((job_id, op_id), (m, None, None))[1:]
        operations.append({"job_id": job_id, "operation_id": op_id, "machine_id": m,
                           "ref_start": s, "ref_end": e, "pert_start": ps, "pert_end": pe,
                           "shifted": (ps, pe) != (s, e)})

    return {
        "jobs": rows,
        "operations": operations,
        "impacted_jobs": impacted,
        "concerned_not_delayed": concerned_only,
        "makespan_ref": ref.makespan,
        "makespan_pert": pert.makespan,
        "late_before": [r["job_id"] for r in rows if r["ref_lateness"]],
        "late_after": [r["job_id"] for r in rows if r["pert_lateness"]],
    }
