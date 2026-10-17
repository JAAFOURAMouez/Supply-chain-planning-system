"""Fictional example dataset (3 jobs, 3 machines, 3 components) used by the 'Charger l'exemple' button."""

SAMPLE_PROBLEM = {
    "machines": ["M1", "M2", "M3"],
    "component_availability": {"C1": 0, "C2": 0, "C3": 0},
    "jobs": [
        {"job_id": "JOB1", "due_date": 10, "priority": 3, "operations": [
            {"operation_id": "O11", "job_id": "JOB1", "machine_id": "M1", "duration": 3, "order_index": 0, "components": ["C1"]},
            {"operation_id": "O12", "job_id": "JOB1", "machine_id": "M2", "duration": 2, "order_index": 1, "components": []},
            {"operation_id": "O13", "job_id": "JOB1", "machine_id": "M3", "duration": 2, "order_index": 2, "components": []}]},
        {"job_id": "JOB2", "due_date": 12, "priority": 2, "operations": [
            {"operation_id": "O21", "job_id": "JOB2", "machine_id": "M2", "duration": 2, "order_index": 0, "components": ["C2"]},
            {"operation_id": "O22", "job_id": "JOB2", "machine_id": "M1", "duration": 4, "order_index": 1, "components": []},
            {"operation_id": "O23", "job_id": "JOB2", "machine_id": "M3", "duration": 3, "order_index": 2, "components": []}]},
        {"job_id": "JOB3", "due_date": 15, "priority": 4, "operations": [
            {"operation_id": "O31", "job_id": "JOB3", "machine_id": "M3", "duration": 4, "order_index": 0, "components": ["C1"]},
            {"operation_id": "O32", "job_id": "JOB3", "machine_id": "M2", "duration": 3, "order_index": 1, "components": []},
            {"operation_id": "O33", "job_id": "JOB3", "machine_id": "M1", "duration": 5, "order_index": 2, "components": ["C3"]}]},
    ],
}
