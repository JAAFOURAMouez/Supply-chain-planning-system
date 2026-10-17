"""
Agent for evaluating optimization results against reference solutions
"""
from typing import Dict, Any, Optional, Tuple
from src.models.supply_chain import SupplyChainProblem, OptimizationResult
import math


class EvaluationAgent:
    """Evaluates optimization results by comparing to reference solutions or benchmarks"""

    def __init__(self):
        pass

    def evaluate_solution(self,
                         problem: SupplyChainProblem,
                         result: OptimizationResult) -> Dict[str, Any]:
        """
        Evaluate the quality of an optimization solution

        Args:
            problem: The original problem
            result: The optimization result to evaluate

        Returns:
            Dictionary containing evaluation metrics
        """
        if result.status not in ["OPTIMAL", "FEASIBLE"] or result.makespan is None:
            return {
                "evaluation_possible": False,
                "reason": "No feasible solution to evaluate",
                "metrics": {}
            }

        metrics = {}

        # Basic metrics
        metrics["makespan"] = result.makespan
        metrics["num_jobs"] = len(problem.jobs)
        metrics["num_machines"] = len(problem.machines)
        metrics["total_operations"] = sum(len(job.operations) for job in problem.jobs)

        # Calculate utilization
        total_work_time = sum(
            op.duration
            for job in problem.jobs
            for op in job.operations
        )

        if result.makespan > 0 and len(problem.machines) > 0:
            # Theoretical minimum makespan (if perfect parallelization)
            theoretical_min = total_work_time / len(problem.machines)
            metrics["lower_bound_makespan"] = theoretical_min
            metrics["utilization_percentage"] = (theoretical_min / result.makespan) * 100
            metrics["efficiency_ratio"] = theoretical_min / result.makespan
        else:
            metrics["utilization_percentage"] = 0.0
            metrics["efficiency_ratio"] = 0.0

        # Tardiness metrics
        tardy_jobs = sum(1 for t in result.tardiness.values() if t > 0)
        total_tardiness = sum(max(0, t) for t in result.tardiness.values())
        avg_tardiness = total_tardiness / len(problem.jobs) if problem.jobs else 0

        metrics["num_tardy_jobs"] = tardy_jobs
        metrics["total_tardiness"] = total_tardiness
        metrics["average_tardiness"] = avg_tardiness

        # Only calculate on-time delivery rate if at least one job has a due date
        jobs_with_due_dates = [job for job in problem.jobs if job.due_date is not None]
        if jobs_with_due_dates:
            # For jobs with due dates, calculate on-time delivery rate
            tardy_jobs_with_due_dates = sum(1 for job in jobs_with_due_dates if result.tardiness.get(job.job_id, 0) > 0)
            metrics["on_time_delivery_rate"] = (len(jobs_with_due_dates) - tardy_jobs_with_due_dates) / len(jobs_with_due_dates) if jobs_with_due_dates else 1.0
            metrics["jobs_with_due_dates"] = len(jobs_with_due_dates)
            metrics["jobs_without_due_dates"] = len(problem.jobs) - len(jobs_with_due_dates)
        else:
            # No due dates provided, so on-time delivery rate is not applicable
            metrics["on_time_delivery_rate"] = None
            metrics["jobs_with_due_dates"] = 0
            metrics["jobs_without_due_dates"] = len(problem.jobs)

        # Schedule validity checks
        validity_checks = self._validate_schedule(problem, result)
        metrics.update(validity_checks)

        # Load balancing
        load_balancing = self._calculate_load_balancing(problem, result)
        metrics.update(load_balancing)

        return {
            "evaluation_possible": True,
            "reason": "Solution evaluated successfully",
            "metrics": metrics
        }

    def _validate_schedule(self, problem: SupplyChainProblem, result: OptimizationResult) -> Dict[str, Any]:
        """Validate that the schedule satisfies all constraints"""
        validation = {
            "schedule_valid": True,
            "validation_errors": []
        }

        # Check 1: Machine capacity (no overlap)
        for machine in problem.machines:
            machine_schedule = result.schedule.get(machine, [])
            machine_schedule.sort(key=lambda x: x['start'])

            for i in range(len(machine_schedule) - 1):
                if machine_schedule[i]['end'] > machine_schedule[i+1]['start']:
                    validation["schedule_valid"] = False
                    validation["validation_errors"].append(
                        f"Overlap on machine {machine}: {machine_schedule[i]['job_id']} "
                        f"ends at {machine_schedule[i]['end']} but {machine_schedule[i+1]['job_id']} "
                        f"starts at {machine_schedule[i+1]['start']}"
                    )

        # Check 2: Job operation precedence
        for job in problem.jobs:
            job_operations = []
            for machine_schedule in result.schedule.values():
                for op in machine_schedule:
                    if op['job_id'] == job.job_id:
                        job_operations.append(op)

            job_operations.sort(key=lambda x: x['start'])

            # Check that operations are in correct order
            expected_order = [op.order_index for op in job.operations]
            actual_order = [op['operation_id'] for op in job_operations]

            # Map operation IDs to expected indices (simplified)
            op_index_map = {op.operation_id: op.order_index for op in job.operations}
            actual_indices = []
            for op in job_operations:
                if op['operation_id'] in op_index_map:
                    actual_indices.append(op_index_map[op['operation_id']])
                else:
                    validation["schedule_valid"] = False
                    validation["validation_errors"].append(
                        f"Unknown operation {op['operation_id']} in job {job.job_id}"
                    )

            if actual_indices != sorted(actual_indices):
                validation["schedule_valid"] = False
                validation["validation_errors"].append(
                    f"Operation order violated for job {job.job_id}"
                )

        # Check 3: All operations scheduled
        scheduled_operations = set()
        for machine_schedule in result.schedule.values():
            for op in machine_schedule:
                scheduled_operations.add((op['job_id'], op['operation_id']))

        expected_operations = set()
        for job in problem.jobs:
            for op in job.operations:
                expected_operations.add((job.job_id, op.operation_id))

        missing_ops = expected_operations - scheduled_operations
        extra_ops = scheduled_operations - expected_operations

        if missing_ops:
            validation["schedule_valid"] = False
            validation["validation_errors"].append(
                f"Missing operations: {missing_ops}"
            )

        if extra_ops:
            validation["schedule_valid"] = False
            validation["validation_errors"].append(
                f"Extra operations: {extra_ops}"
            )

        return validation

    def _calculate_load_balancing(self, problem: SupplyChainProblem, result: OptimizationResult) -> Dict[str, Any]:
        """Calculate load balancing metrics across machines"""
        machine_loads = {}
        for machine in problem.machines:
            machine_schedule = result.schedule.get(machine, [])
            total_load = sum(op['duration'] for op in machine_schedule)
            machine_loads[machine] = total_load

        if machine_loads:
            loads = list(machine_loads.values())
            avg_load = sum(loads) / len(loads)
            max_load = max(loads)
            min_load = min(loads) if loads else 0

            # Load balancing coefficient (0 = perfect balance, higher = worse balance)
            if avg_load > 0:
                load_variance = sum((l - avg_load) ** 2 for l in loads) / len(loads)
                load_balancing_coefficient = math.sqrt(load_variance) / avg_load
            else:
                load_balancing_coefficient = 0.0

            return {
                "machine_loads": machine_loads,
                "average_machine_load": avg_load,
                "max_machine_load": max_load,
                "min_machine_load": min_load,
                "load_balancing_coefficient": load_balancing_coefficient,
                "load_balance_percentage": (1 - load_balancing_coefficient) * 100 if avg_load > 0 else 100
            }
        else:
            return {
                "machine_loads": {},
                "average_machine_load": 0,
                "max_machine_load": 0,
                "min_machine_load": 0,
                "load_balancing_coefficient": 0.0,
                "load_balance_percentage": 100.0
            }

    def compare_with_reference(self,
                              result: OptimizationResult,
                              reference_result: OptimizationResult) -> Dict[str, Any]:
        """
        Compare optimization result with a reference solution

        Args:
            result: The optimization result to evaluate
            reference_result: The reference solution to compare against

        Returns:
            Dictionary containing comparison metrics
        """
        if result.status not in ["OPTIMAL", "FEASIBLE"] or reference_result.status not in ["OPTIMAL", "FEASIBLE"]:
            return {
                "comparison_possible": False,
                "reason": "One or both solutions are not feasible",
                "metrics": {}
            }

        if result.makespan is None or reference_result.makespan is None:
            return {
                "comparison_possible": False,
                "reason": "Makespan not available for one or both solutions",
                "metrics": {}
            }

        makespan_improvement = (
            (reference_result.makespan - result.makespan) / reference_result.makespan * 100
            if reference_result.makespan > 0 else 0
        )

        return {
            "comparison_possible": True,
            "reason": "Comparison completed successfully",
            "metrics": {
                "result_makespan": result.makespan,
                "reference_makespan": reference_result.makespan,
                "makespan_improvement_percentage": makespan_improvement,
                "is_better": result.makespan < reference_result.makespan,
                "is_equal": result.makespan == reference_result.makespan,
                "is_worse": result.makespan > reference_result.makespan
            }
        }


# Factory function
def create_evaluation_agent() -> EvaluationAgent:
    return EvaluationAgent()