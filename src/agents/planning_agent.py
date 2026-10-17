"""
Agent for planning and optimization using Google OR-Tools CP-SAT
"""
from typing import Dict, Any, Optional, List, Tuple
from src.models.supply_chain import SupplyChainProblem, OptimizationResult, Job, Operation
try:
    from ortools.sat.python import cp_model
except ImportError:
    cp_model = None


class PlanningAgent:
    """Plans and solves supply chain optimization problems using OR-Tools CP-SAT"""

    def __init__(self):
        self.model = None
        self.solver = None
        self.variables = {}

    def solve_job_shop(self, problem: SupplyChainProblem) -> OptimizationResult:
        """
        Solve the job shop scheduling problem using OR-Tools CP-SAT

        Args:
            problem: Validated SupplyChainProblem instance

        Returns:
            OptimizationResult instance with solution
        """
        if cp_model is None:
            raise ImportError("ortools n'est pas installé: impossible de résoudre avec CP-SAT")

        try:
            # Create the CP-SAT model
            model = cp_model.CpModel()

            # Calculate horizon (upper bound on time variables)
            horizon = 0
            for job in problem.jobs:
                for op in job.operations:
                    horizon += op.duration  # Sum of all durations as upper bound
            horizon += max(problem.component_availability.values(), default=0)
            # Downtime can push operations later: the horizon must cover it
            for intervals in problem.machine_availability.values():
                for _, end_unavailable in intervals:
                    horizon += end_unavailable

            machine_intervals = {}  # machine -> list of intervals
            job_intervals = {}      # job -> list of intervals per operation

            # Initialize data structures
            for machine in problem.machines:
                machine_intervals[machine] = []

            for job in problem.jobs:
                job_intervals[job.job_id] = []

            # Create variables for each operation
            all_tasks = {}
            for job in problem.jobs:
                for op in job.operations:
                    # Create start, end, and interval variables
                    suffix = f"_{job.job_id}_{op.operation_id}"
                    start_var = model.NewIntVar(0, horizon, f"start{suffix}")
                    end_var = model.NewIntVar(0, horizon, f"end{suffix}")
                    interval_var = model.NewIntervalVar(start_var, op.duration, end_var, f"interval{suffix}")

                    # Components: the operation cannot start before all its components have arrived
                    release = max((problem.component_availability.get(c, 0) for c in op.components), default=0)
                    if release > 0:
                        model.Add(start_var >= release)

                    all_tasks[(job.job_id, op.operation_id)] = {
                        'start': start_var,
                        'end': end_var,
                        'interval': interval_var,
                        'job': job,
                        'operation': op
                    }

                    # Add to machine and job lists
                    machine_intervals[op.machine_id].append(interval_var)
                    job_intervals[job.job_id].append((op.order_index, start_var, end_var, op.duration))

            # One NoOverlap per machine: operations + fixed unavailability intervals
            for machine in problem.machines:
                intervals = list(machine_intervals[machine])
                for start_u, end_u in problem.machine_availability.get(machine, []):
                    intervals.append(model.NewFixedSizeIntervalVar(
                        start_u, end_u - start_u, f"unavailable_{machine}_{start_u}_{end_u}"))
                if intervals:
                    model.AddNoOverlap(intervals)

            # Add constraints: operation order within each job
            for job in problem.jobs:
                job_ops = job_intervals[job.job_id]
                job_ops.sort(key=lambda x: x[0])  # Sort by order_index
                for i in range(len(job_ops) - 1):
                    # Finish of operation i <= start of operation i+1
                    model.Add(job_ops[i][2] <= job_ops[i+1][1])  # end_i <= start_{i+1}

            # Objective: minimize makespan
            makespan_var = model.NewIntVar(0, horizon, "makespan")
            end_vars = [task['end'] for task in all_tasks.values()]
            model.AddMaxEquality(makespan_var, end_vars)
            model.Minimize(makespan_var)

            # Solve
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 30.0  # 30 second timeout
            solver.parameters.num_search_workers = 8
            status = solver.Solve(model)

            # Process results
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                makespan = solver.Value(makespan_var)

                # Build schedule
                schedule = {machine: [] for machine in problem.machines}
                job_completion_times = {}
                tardiness = {}

                for job in problem.jobs:
                    job_completion_times[job.job_id] = 0
                    tardiness[job.job_id] = 0

                for (job_id, op_id), task in all_tasks.items():
                    start = solver.Value(task['start'])
                    end = solver.Value(task['end'])
                    machine = task['operation'].machine_id

                    schedule[machine].append({
                        'job_id': job_id,
                        'operation_id': op_id,
                        'start': start,
                        'end': end,
                        'duration': task['operation'].duration
                    })

                    # Update job completion time
                    if end > job_completion_times[job_id]:
                        job_completion_times[job_id] = end

                # Calculate tardiness
                for job in problem.jobs:
                    if job.due_date is not None:
                        tardiness[job.job_id] = job_completion_times[job.job_id] - job.due_date
                    else:
                        tardiness[job.job_id] = 0

                # Sort schedule by start time for each machine
                for machine in schedule:
                    schedule[machine].sort(key=lambda x: x['start'])

                return OptimizationResult(
                    status="OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
                    makespan=makespan,
                    schedule=schedule,
                    job_completion_times=job_completion_times,
                    tardiness=tardiness,
                    objective_value=float(makespan)
                )
            else:
                # No solution found
                return OptimizationResult(
                    status="INFEASIBLE" if status == cp_model.INFEASIBLE else "UNKNOWN",
                    makespan=None,
                    schedule={},
                    job_completion_times={},
                    tardiness={},
                    objective_value=None
                )

        except Exception as e:
            # No silent heuristic fallback: it would ignore downtime and look valid
            return OptimizationResult(status=f"ERROR: {e}", makespan=None, schedule={},
                                      job_completion_times={}, tardiness={}, objective_value=None)

    def _fallback_solution(self, problem: SupplyChainProblem) -> OptimizationResult:
        """
        Fallback solution using simple heuristics when OR-Tools is not available or fails
        """
        # Simple greedy scheduling algorithm
        schedule = {machine: [] for machine in problem.machines}
        machine_available_time = {machine: 0 for machine in problem.machines}
        job_completion_times = {}
        operation_start_times = {}  # (job_id, op_id) -> start_time

        # Initialize job completion times
        for job in problem.jobs:
            job_completion_times[job.job_id] = 0

        # Create list of all operations sorted by job priority and order
        all_operations = []
        for job in problem.jobs:
            for op in job.operations:
                all_operations.append((job, op))

        # Sort by priority (higher first) then by job ID
        all_operations.sort(key=lambda x: (-x[0].priority, x[0].job_id))

        # Schedule operations greedily
        for job, op in all_operations:
            # Find earliest start time considering machine availability and job precedence
            earliest_start = machine_available_time[op.machine_id]

            # Check if previous operation in same job is completed
            if op.order_index > 0:
                prev_op_id = f"O{job.job_id[3:]}{op.order_index}"  # Simplified
                prev_key = (job.job_id, f"O{job.job_id[3:]}{op.order_index}")
                # Actually, we need to track by order index properly
                prev_key = (job.job_id, op.order_index - 1)
                if prev_key in operation_start_times:
                    prev_end = operation_start_times[prev_key] + \
                             next(o.duration for o in job.operations if o.order_index == op.order_index - 1)
                    earliest_start = max(earliest_start, prev_end)

            start_time = earliest_start
            end_time = start_time + op.duration

            # Update tracking
            operation_start_times[(job.job_id, op.order_index)] = start_time
            machine_available_time[op.machine_id] = end_time

            if end_time > job_completion_times[job.job_id]:
                job_completion_times[job.job_id] = end_time

            # Add to schedule
            schedule[op.machine_id].append({
                'job_id': job.job_id,
                'operation_id': op.operation_id,
                'start': start_time,
                'end': end_time,
                'duration': op.duration
            })

        # Calculate makespan
        makespan = max(machine_available_time.values()) if machine_available_time else 0

        # Calculate tardiness
        tardiness = {}
        for job in problem.jobs:
            if job.due_date is not None:
                tardiness[job.job_id] = job_completion_times[job.job_id] - job.due_date
            else:
                tardiness[job.job_id] = 0

        # Sort schedule by start time
        for machine in schedule:
            schedule[machine].sort(key=lambda x: x['start'])

        return OptimizationResult(
            status="FEASIBLE",  # Heuristic solution
            makespan=makespan,
            schedule=schedule,
            job_completion_times=job_completion_times,
            tardiness=tardiness,
            objective_value=float(makespan)
        )


# Factory function
def create_planning_agent() -> PlanningAgent:
    return PlanningAgent()