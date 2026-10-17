"""
LangGraph workflow for Supply Chain Optimization
"""
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from src.models.supply_chain import AgentState
from src.agents.problem_analyzer import create_problem_analyzer
from src.agents.validation_agent import create_validation_agent
from src.agents.planning_agent import create_planning_agent
from src.agents.evaluation_agent import create_evaluation_agent
from src.agents.explanation_agent import create_explanation_agent
from src.utils.disruptions import extract_disruptions, mentions_disruption
from src.utils.impact import build_impact


def create_supply_chain_workflow(hf_token: Optional[str] = None):
    """Create the LangGraph workflow for supply chain optimization"""

    # Initialize agents
    problem_analyzer = create_problem_analyzer(hf_token)
    validation_agent = create_validation_agent()
    planning_agent = create_planning_agent()
    evaluation_agent = create_evaluation_agent()
    explanation_agent = create_explanation_agent(hf_token)

    # Define the workflow nodes
    def analyze_problem_node(state: AgentState) -> AgentState:
        """Node 1: Analyze the problem description"""
        state.current_step = "analyze_problem"

        # A dataset loaded by the user always wins: jobs/durations are never extracted
        # (or invented) by the LLM when real data is available.
        if state.loaded_problem:
            import copy
            state.extracted_data = copy.deepcopy(state.loaded_problem)
            state.error_messages = []
            return state

        # Analyze the problem description
        analysis_result = problem_analyzer.analyze_problem(state.problem_description)

        if analysis_result["success"]:
            # Convert extracted data to SupplyChainProblem
            try:
                problem_dict = analysis_result["data"]
                # Ensure we have the right structure
                if "jobs" not in problem_dict:
                    problem_dict["jobs"] = []  # Will be filled by analyzer

                if not problem_dict.get("jobs") or not problem_dict.get("machines"):
                    state.error_messages = [
                        "Donnée manquante : aucun job / aucune machine dans la description et aucun jeu de données chargé. "
                        "Chargez un fichier JSON ou CSV (barre latérale) ou décrivez les jobs, machines et durées."
                    ]
                    return state
                state.supply_chain_problem = None  # Will be validated in next step
                state.extracted_data = problem_dict
                state.error_messages = []  # Clear errors
            except Exception as e:
                state.error_messages = [f"Failed to process analysis result: {str(e)}"]
        else:
            state.error_messages = [analysis_result.get("error", "Unknown error in problem analysis")]

        return state

    def validate_problem_node(state: AgentState) -> AgentState:
        """Node 2: Validate the extracted problem data"""
        state.current_step = "validate_problem"

        if not state.error_messages and hasattr(state, 'extracted_data'):
            is_valid, validated_problem, errors = validation_agent.validate_problem(
                state.extracted_data
            )
            if errors:
                state.error_messages = errors
                state.supply_chain_problem = None
            else:
                state.supply_chain_problem = validated_problem
                state.error_messages = []  # Clear errors on success
        elif not hasattr(state, 'extracted_data'):
            state.error_messages = ["No extracted data to validate"]

        return state

    def _merge_availability(problem, availability):
        merged = {m: [list(i) for i in ivs] for m, ivs in problem.machine_availability.items()}
        for m, ivs in availability.items():
            merged.setdefault(m, [])
            for iv in ivs:
                if iv not in merged[m]:
                    merged[m].append(iv)
        return problem.model_copy(update={"machine_availability": merged})

    def apply_disruption_node(state: AgentState) -> AgentState:
        """Node 2b: deterministic. Check the target exists, then apply the disruption."""
        state.current_step = "apply_disruption"
        state.warnings = []
        scn = state.scenario
        state.disruption_requested = bool(scn) or mentions_disruption(state.problem_description)

        if state.error_messages or not state.supply_chain_problem:
            return state
        problem = state.supply_chain_problem

        # Structured scenario (from the chatbot)
        if scn:
            state.warnings = list(scn.get("notes", []))
            if scn.get("type") == "machine_breakdown":
                machine = scn.get("machine")
                if machine not in problem.machines:
                    state.error_messages = [f"Précision nécessaire : la machine {machine} n'existe pas "
                                            f"(machines connues : {problem.machines})."]
                    return state
                state.supply_chain_problem = _merge_availability(
                    problem, {machine: [[int(scn["start"]), int(scn["end"])]]})
            elif scn.get("type") == "component_delay":
                comp = scn.get("component")
                if comp not in problem.component_availability:
                    state.error_messages = [f"Précision nécessaire : le composant {comp} n'existe pas dans les données "
                                            f"(composants connus : {sorted(problem.component_availability)})."]
                    return state
                baseline = dict(problem.component_availability)
                delayed = dict(baseline)
                delayed[comp] = int(scn["new_date"])
                if delayed[comp] <= baseline[comp]:
                    state.warnings.append(f"La nouvelle date (t={delayed[comp]}) n'est pas postérieure à la date "
                                          f"actuelle (t={baseline[comp]}) : aucun retard réel simulé.")
                state.supply_chain_problem = problem.model_copy(update={
                    "component_availability": delayed, "baseline_component_availability": baseline})
            else:
                state.error_messages = [f"Type de scénario inconnu : {scn.get('type')}"]
            return state

        # Free-text fallback (no chatbot): machine breakdown only
        if not state.disruption_requested:
            return state

        availability, warnings = extract_disruptions(state.problem_description)
        state.warnings = warnings
        if not availability:
            state.error_messages = [
                "Précision nécessaire : panne demandée mais machine ou horaires non identifiables "
                "(ex. « machine M2 pendant les 2 premières heures » ou « M2 sur [0, 2) »). "
                "Rien n'a été inventé."
            ]
            return state
        unknown = sorted(set(availability) - set(problem.machines))
        if unknown:
            state.error_messages = [
                f"Précision nécessaire : machine(s) {unknown} absente(s) des données "
                f"(machines connues : {problem.machines}). Indisponibilité non ajoutée, "
                "quelle machine faut-il utiliser ?"
            ]
            return state
        state.supply_chain_problem = _merge_availability(problem, availability)
        return state

    def plan_solution_node(state: AgentState) -> AgentState:
        """Node 3: Plan and solve the optimization problem"""
        state.current_step = "plan_solution"

        # Initialize result fields
        state.reference_result = None
        state.perturbed_result = None

        if not state.error_messages and state.supply_chain_problem:
            try:
                # Check if we have downtime information (machine_availability)
                has_downtime = bool(state.supply_chain_problem.machine_availability) or \
                    state.supply_chain_problem.baseline_component_availability is not None

                if has_downtime:
                    # Run reference scenario (no downtime)
                    reference_problem = state.supply_chain_problem.model_copy()
                    reference_problem.machine_availability = {}  # Clear downtime for reference
                    if reference_problem.baseline_component_availability is not None:
                        reference_problem.component_availability = dict(reference_problem.baseline_component_availability)
                    reference_problem.downtime_scenario = "reference"

                    reference_result = planning_agent.solve_job_shop(reference_problem)
                    state.reference_result = reference_result

                    # Run perturbed scenario (with downtime)
                    perturbed_problem = state.supply_chain_problem.model_copy()
                    perturbed_problem.downtime_scenario = "perturbed"

                    perturbed_result = planning_agent.solve_job_shop(perturbed_problem)
                    state.perturbed_result = perturbed_result

                    if (state.scenario and reference_result.status in ("OPTIMAL", "FEASIBLE")
                            and perturbed_result.status in ("OPTIMAL", "FEASIBLE")):
                        state.impact = build_impact(state.supply_chain_problem, state.scenario,
                                                    reference_result, perturbed_result)

                elif state.disruption_requested:
                    raise ValueError("Panne demandée mais absente du modèle: scénario perturbé non calculé")
                else:
                    # No downtime specified, run single scenario and treat as reference
                    result = planning_agent.solve_job_shop(state.supply_chain_problem)
                    state.reference_result = result  # Store as reference for consistency
                    state.perturbed_result = result  # Also store as perturbed for consistency in single scenario case

                state.error_messages = []  # Clear errors on success
            except Exception as e:
                state.error_messages = [f"Planning failed: {str(e)}"]
                state.reference_result = None
                state.perturbed_result = None
        else:
            state.reference_result = None
            state.perturbed_result = None

        return state

    def evaluate_solution_node(state: AgentState) -> AgentState:
        """Node 4: Evaluate the solution quality"""
        state.current_step = "evaluate_solution"

        # Initialize evaluation result
        state.evaluation_result = {
            "evaluation_possible": False,
            "reason": "",
            "metrics": {},
            "reference_evaluation": None,
            "perturbed_evaluation": None,
            "comparative_metrics": {}
        }

        if not state.error_messages and state.supply_chain_problem:
            # Evaluate reference scenario if available
            if state.reference_result:
                try:
                    reference_evaluation = evaluation_agent.evaluate_solution(
                        state.supply_chain_problem,
                        state.reference_result
                    )
                    state.evaluation_result["reference_evaluation"] = reference_evaluation
                except Exception as e:
                    state.evaluation_result["reference_evaluation"] = {"evaluation_possible": False, "reason": str(e), "metrics": {}}

            # Evaluate perturbed scenario if available
            if state.perturbed_result:
                try:
                    perturbed_evaluation = evaluation_agent.evaluate_solution(
                        state.supply_chain_problem,
                        state.perturbed_result
                    )
                    state.evaluation_result["perturbed_evaluation"] = perturbed_evaluation
                except Exception as e:
                    state.evaluation_result["perturbed_evaluation"] = {"evaluation_possible": False, "reason": str(e), "metrics": {}}

            # If we have at least one evaluation, mark as possible
            if state.evaluation_result["reference_evaluation"] or state.evaluation_result["perturbed_evaluation"]:
                state.evaluation_result["evaluation_possible"] = True
                state.evaluation_result["reason"] = "Evaluation completed"

                # Calculate comparative metrics if both scenarios are available
                if (state.evaluation_result["reference_evaluation"] and
                    state.evaluation_result["reference_evaluation"].get("evaluation_possible") and
                    state.evaluation_result["perturbed_evaluation"] and
                    state.evaluation_result["perturbed_evaluation"].get("evaluation_possible")):

                    ref_metrics = state.evaluation_result["reference_evaluation"].get("metrics", {})
                    pert_metrics = state.evaluation_result["perturbed_evaluation"].get("metrics", {})

                    # Calculate differences for key metrics
                    comparative_metrics = {}
                    for key in set(ref_metrics.keys()) | set(pert_metrics.keys()):
                        ref_val = ref_metrics.get(key, 0)
                        pert_val = pert_metrics.get(key, 0)
                        if isinstance(ref_val, (int, float)) and isinstance(pert_val, (int, float)):
                            comparative_metrics[f"{key}_difference"] = pert_val - ref_val
                            comparative_metrics[f"{key}_percent_change"] = ((pert_val - ref_val) / ref_val * 100) if ref_val != 0 else 0

                    state.evaluation_result["comparative_metrics"] = comparative_metrics
            else:
                state.evaluation_result["reason"] = "No valid results to evaluate"
        else:
            state.evaluation_result["reason"] = "Missing problem or solution"

        return state

    def explain_results_node(state: AgentState) -> AgentState:
        """Node 5: Generate explanation of results"""
        state.current_step = "explain_results"

        if state.scenario:  # the chatbot builds its own deterministic report
            state.explanation = ""
            return state

        if not state.error_messages and state.supply_chain_problem:
            try:
                # Check if we have both reference and perturbed results for comparison
                if state.reference_result and state.perturbed_result:
                    explanation = explanation_agent.explain_results(
                        state.supply_chain_problem,
                        state.perturbed_result,  # Use perturbed as main result for backward compatibility
                        state.evaluation_result if hasattr(state, 'evaluation_result') else None,
                        reference_result=state.reference_result  # Pass reference result for comparison
                    )
                elif state.reference_result:
                    # Fallback to reference result explanation (single scenario or reference only)
                    explanation = explanation_agent.explain_results(
                        state.supply_chain_problem,
                        state.reference_result,
                        state.evaluation_result if hasattr(state, 'evaluation_result') else None
                    )
                elif state.perturbed_result:
                    # Fallback to perturbed result explanation
                    explanation = explanation_agent.explain_results(
                        state.supply_chain_problem,
                        state.perturbed_result,
                        state.evaluation_result if hasattr(state, 'evaluation_result') else None
                    )
                else:
                    explanation = "Unable to generate explanation due to missing optimization results."

                if state.warnings:
                    explanation = ("⚠️ Hypothèses / points à confirmer :\n"
                                   + "\n".join(f"- {w}" for w in state.warnings) + "\n\n" + explanation)
                state.explanation = explanation
                state.error_messages = []  # Clear errors on success
            except Exception as e:
                state.error_messages = [f"Explanation generation failed: {str(e)}"]
                state.explanation = "Failed to generate explanation due to an error."
        else:
            if state.error_messages:
                state.explanation = (
                    "Impossible de générer l'explication, le workflow s'est arrêté à l'étape "
                    f"« {state.extracted_data is None and 'analyse' or 'validation/planification'} » :\n\n"
                    + "\n".join(f"- {e}" for e in state.error_messages)
                )
            else:
                state.explanation = "Impossible de générer l'explication : problème ou solution manquant."

        return state

    # Define the workflow
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze_problem", analyze_problem_node)
    workflow.add_node("validate_problem", validate_problem_node)
    workflow.add_node("apply_disruption", apply_disruption_node)
    workflow.add_node("plan_solution", plan_solution_node)
    workflow.add_node("evaluate_solution", evaluate_solution_node)
    workflow.add_node("explain_results", explain_results_node)

    # Define the flow
    workflow.set_entry_point("analyze_problem")
    workflow.add_edge("analyze_problem", "validate_problem")
    workflow.add_edge("validate_problem", "apply_disruption")
    workflow.add_edge("apply_disruption", "plan_solution")
    workflow.add_edge("plan_solution", "evaluate_solution")
    workflow.add_edge("evaluate_solution", "explain_results")
    workflow.add_edge("explain_results", END)

    # Compile the workflow
    return workflow.compile()


# Factory function for easy instantiation
def get_supply_chain_workflow(hf_token: Optional[str] = None):
    """Get a compiled supply chain optimization workflow"""
    return create_supply_chain_workflow(hf_token)