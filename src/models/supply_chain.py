"""
Pydantic models for Supply Chain optimization problem
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator, root_validator
from datetime import datetime, timedelta


class Operation(BaseModel):
    """Represents a single operation in a job"""
    operation_id: str = Field(..., description="Unique identifier for the operation")
    job_id: str = Field(..., description="Identifier of the job this operation belongs to")
    machine_id: str = Field(..., description="Identifier of the machine required for this operation")
    duration: int = Field(..., gt=0, description="Processing time in time units")
    order_index: int = Field(..., ge=0, description="Order of this operation within the job (0-based)")
    components: List[str] = Field(default_factory=list, description="Components required before this operation can start")

    @validator('duration')
    def duration_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Duration must be positive')
        return v


class Job(BaseModel):
    """Represents a job to be processed"""
    job_id: str = Field(..., description="Unique identifier for the job")
    operations: List[Operation] = Field(..., description="List of operations in order")
    due_date: Optional[int] = Field(None, description="Due date for the job (time units)")
    priority: int = Field(1, ge=1, le=5, description="Priority level (1=lowest, 5=highest)")

    @validator('operations')
    def operations_must_be_ordered(cls, v):
        if not v:
            raise ValueError('Job must have at least one operation')
        # Check that operations are properly ordered
        order_indices = [op.order_index for op in v]
        if sorted(order_indices) != list(range(len(v))):
            raise ValueError('Operations must have sequential order_index starting from 0')
        return v


class SupplyChainProblem(BaseModel):
    """Represents the complete supply chain optimization problem"""
    jobs: List[Job] = Field(..., description="List of jobs to be scheduled")
    machines: List[str] = Field(..., description="List of available machines")
    machine_availability: Dict[str, List[List[int]]] = Field(
        default_factory=dict,
        description="Machine availability windows as dict of machine_id -> list of [start, end] unavailable intervals"
    )
    component_availability: Dict[str, int] = Field(
        default_factory=dict,
        description="component_id -> earliest time the component is available (operations needing it cannot start before)"
    )
    baseline_component_availability: Optional[Dict[str, int]] = Field(
        None, description="Component availability before a delay scenario (used for the reference run)"
    )
    time_horizon: Optional[int] = Field(None, description="Maximum time horizon for scheduling")
    downtime_scenario: Optional[str] = Field(None, description="Scenario identifier for downtime variations (e.g., 'reference', 'perturbed')")

    @validator('jobs')
    def must_have_jobs(cls, v):
        if not v:
            raise ValueError('Problem must have at least one job')
        return v

    @validator('machines')
    def must_have_machines(cls, v):
        if not v:
            raise ValueError('Problem must have at least one machine')
        return v

    @validator('machine_availability')
    def validate_machine_availability(cls, v):
        # Basic validation: ensure values are lists of [start, end] intervals
        for machine_id, intervals in v.items():
            if not isinstance(intervals, list):
                raise ValueError(f'Availability for machine {machine_id} must be a list')
            for interval in intervals:
                if not isinstance(interval, list) or len(interval) != 2:
                    raise ValueError(f'Each interval for machine {machine_id} must be a list of two integers [start, end]')
                start, end = interval
                if not isinstance(start, int) or not isinstance(end, int):
                    raise ValueError(f'Start and end times must be integers for machine {machine_id}')
                if start < 0:
                    raise ValueError(f'Start time must be non-negative for machine {machine_id}')
                if end <= start:
                    raise ValueError(f'End time must be greater than start time for machine {machine_id}')
        return v

    @validator('machines')
    def machines_must_include_availability_keys(cls, v, values):
        # Ensure that if machine_availability is provided, all its keys are in machines
        if 'machine_availability' in values and values['machine_availability']:
            availability_machines = set(values['machine_availability'].keys())
            machine_set = set(v)
            if not availability_machines.issubset(machine_set):
                raise ValueError(f'Machine availability references machines not in machines list: {availability_machines - machine_set}')
        return v

    @root_validator(skip_on_failure=True)
    def check_referential_integrity(cls, values):
        # Extract the relevant fields
        machines = values.get('machines', [])
        jobs = values.get('jobs', [])
        machine_availability = values.get('machine_availability', {})
        component_availability = values.get('component_availability', {})

        # Check each job
        for job in jobs:
            job_id = job.job_id
            # Check each operation in the job
            for op in job.operations:
                op_machine_id = op.machine_id
                op_job_id = op.job_id
                op_components = op.components

                # Check machine_id exists in machines
                if op_machine_id not in machines:
                    raise ValueError(f"Machine ID '{op_machine_id}' in operation not found in machines list.")

                # Check job_id matches the parent job's job_id
                if op_job_id != job_id:
                    raise ValueError(f"Operation job_id '{op_job_id}' does not match parent job ID '{job_id}'.")

                # Check each component exists in component_availability
                for comp in op_components:
                    if comp not in component_availability:
                        raise ValueError(f"Component ID '{comp}' in operation not found in component_availability.")

        # Also check that machine_availability keys are in machines (though there is already a validator for this)
        for machine_id in machine_availability:
            if machine_id not in machines:
                raise ValueError(f"Machine ID '{machine_id}' in machine_availability not found in machines list.")

        return values


class OptimizationResult(BaseModel):
    """Represents the result of the optimization"""
    status: str = Field(..., description="Status of the optimization (OPTIMAL, FEASIBLE, INFEASIBLE, etc.)")
    makespan: Optional[int] = Field(None, description="Total completion time")
    schedule: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Schedule per machine: {machine_id: [{'job_id': str, 'operation_id': str, 'start': int, 'end': int}]}"
    )
    job_completion_times: Dict[str, int] = Field(
        default_factory=dict,
        description="Completion time for each job"
    )
    tardiness: Dict[str, int] = Field(
        default_factory=dict,
        description="Tardiness for each job (negative if early)"
    )
    objective_value: Optional[float] = Field(None, description="Value of the objective function")


class AgentState(BaseModel):
    """State for the LangGraph agent workflow"""
    problem_description: str = Field("", description="Original problem description in natural language")
    supply_chain_problem: Optional[SupplyChainProblem] = Field(None, description="Parsed and validated problem")
    reference_result: Optional[OptimizationResult] = Field(None, description="Result from the reference (no downtime) scenario")
    perturbed_result: Optional[OptimizationResult] = Field(None, description="Result from the perturbed (with downtime) scenario")
    explanation: str = Field("", description="Explanation of results from the LLM")
    error_messages: List[str] = Field(default_factory=list, description="Any error messages encountered")
    current_step: str = Field("start", description="Current step in the workflow")
    extracted_data: Optional[Dict[str, Any]] = Field(None, description="Extracted problem data from analyzer")
    evaluation_result: Optional[Dict[str, Any]] = Field(None, description="Evaluation results from evaluation agent")
    loaded_problem: Optional[Dict[str, Any]] = Field(None, description="Dataset loaded by the user (machines + jobs); used instead of LLM extraction")
    scenario: Optional[Dict[str, Any]] = Field(None, description="Structured disruption scenario (machine_breakdown / component_delay)")
    impact: Optional[Dict[str, Any]] = Field(None, description="Before/after comparison per job and operation")
    disruption_requested: bool = Field(False, description="True if the description asks for a machine breakdown/unavailability")
    warnings: List[str] = Field(default_factory=list, description="Non-blocking warnings (missing data, ignored input...)")