"""
Agent for explaining optimization results using LLM
"""
import os
from typing import Dict, Any, Optional
from src.utils.llm import get_llm, call_llm
from langchain_core.prompts import PromptTemplate
from src.models.supply_chain import SupplyChainProblem, OptimizationResult
import json


class ExplanationAgent:
    """Explains optimization results using natural language generation"""

    def __init__(self, hf_token: str = None):
        self.hf_token = hf_token  # None -> get_llm reads GROQ_API_KEY / HF_TOKEN from env
        self.llm = None
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LLM (Groq or Hugging Face, see src/utils/llm.py)"""
        self.llm = get_llm(self.hf_token, temperature=0.3, max_tokens=4096)

    def explain_results(self,
                       problem: SupplyChainProblem,
                       result: OptimizationResult,
                       evaluation_metrics: Optional[Dict[str, Any]] = None,
                       reference_result: Optional[OptimizationResult] = None) -> str:
        """
        Generate a natural language explanation of the optimization results

        Args:
            problem: The original problem
            result: The optimization result (typically perturbed scenario)
            evaluation_metrics: Optional evaluation metrics
            reference_result: Optional reference scenario result for comparison

        Returns:
            Natural language explanation string
        """
        if not self.llm:
            return self._rule_based_explanation(problem, result, evaluation_metrics, reference_result)

        # Prepare context for the LLM
        context = self._prepare_explanation_context(problem, result, evaluation_metrics, reference_result)

        prompt_template = PromptTemplate(
            input_variables=["context"],
            template="""
Vous êtes un expert en optimisation de la chaîne d'approvisionnement.
Expliquez les résultats de l'optimisation de manière claire et concise pour un professionnel de l'industrie.

Contexte:
{context}

Fournissez une explication structurée comprenant:
1. Résumé des résultats obtenus
2. Analyse de la qualité de la solution (temps de traitement, respect des dates de livraison, équilibrage de charge)
3. Recommandations pour l'amélioration si applicable
4. Limites de l'approche utilisée

Explication en français :
"""
        )

        try:
            # Format the prompt and invoke the LLM directly
            prompt = prompt_template.format(context=context)
            explanation = call_llm(self.llm, prompt)
            return explanation.strip()

        except Exception as e:
            return f"Erreur lors de la génération de l'explication: {str(e)}\n\n" + \
                   self._rule_based_explanation(problem, result, evaluation_metrics)

    def _prepare_explanation_context(self,
                                   problem: SupplyChainProblem,
                                   result: OptimizationResult,
                                   evaluation_metrics: Optional[Dict[str, Any]] = None,
                                   reference_result: Optional[OptimizationResult] = None) -> str:
        """Prepare context string for the LLM"""
        context_parts = []

        # Problem description
        context_parts.append(f"PROBLÈME À OPTIMISER:")
        context_parts.append(f"- Nombre de jobs: {len(problem.jobs)}")
        context_parts.append(f"- Nombre de machines: {len(problem.machines)}")
        context_parts.append(f"- Machines disponibles: {', '.join(problem.machines)}")

        total_operations = sum(len(job.operations) for job in problem.jobs)
        context_parts.append(f"- Nombre total d'opérations: {total_operations}")

        context_parts.append("")
        context_parts.append("DÉTAILS DES JOBS:")
        for job in problem.jobs:
            ops_info = [f"Op{op.order_index+1}({op.machine_id},{op.duration})"
                       for op in job.operations]
            context_parts.append(
                f"- Job {job.job_id}: Priorité {job.priority}, "
                f"Date de livraison: {job.due_date if job.due_date else 'Non spécifiée'}, "
                f"Opérations: {' -> '.join(ops_info)}"
            )

        # Add downtime information if present
        if problem.machine_availability:
            context_parts.append("")
            context_parts.append("INDISPONIBILITÉS DES MACHINES:")
            for machine_id, intervals in problem.machine_availability.items():
                if intervals:
                    interval_strs = [f"[{start}-{end}]" for start, end in intervals]
                    context_parts.append(f"- Machine {machine_id}: Indisponible pendant {', '.join(interval_strs)}")
                else:
                    context_parts.append(f"- Machine {machine_id}: Aucune indisponibilité spécifiée")

        context_parts.append("")
        if reference_result is not None:
            context_parts.append("ANALYSE COMPARATIVE: SCÉNARIO DE RÉFÉRENCE (SANS INDISPONIBILITÉ) vs SCÉNARIO PERTURBÉ (AVEC INDISPONIBILITÉ)")
            context_parts.append("")
            context_parts.append("SCÉNARIO DE RÉFÉRENCE:")
            context_parts.append(f"- Statut: {reference_result.status}")
            if reference_result.makespan is not None:
                context_parts.append(f"- Temps de makespan: {reference_result.makespan}")
            else:
                context_parts.append("- Temps de makespan: Non disponible (pas de solution faisable)")

            if reference_result.status in ["OPTIMAL", "FEASIBLE"] and reference_result.makespan is not None:
                context_parts.append("")
                context_parts.append("PLANNING DE RÉFÉRENCE:")
                for machine in sorted(reference_result.schedule.keys()):
                    machine_ops = reference_result.schedule[machine]
                    if machine_ops:
                        ops_str = []
                        for op in machine_ops:
                            ops_str.append(f"{op['job_id']}({op['operation_id']}:{op['start']}-{op['end']})")
                        context_parts.append(f"- Machine {machine}: {' -> '.join(ops_str)}")
                    else:
                        context_parts.append(f"- Machine {machine}: Aucune opération assignée")

                context_parts.append("")
                context_parts.append("MÉTRIQUES DE PERFORMANCE (RÉFÉRENCE):")
                context_parts.append(f"- Temps de complétion par job: {reference_result.job_completion_times}")
                context_parts.append(f"- Retard par job (négatif = en avance): {reference_result.tardiness}")
            context_parts.append("")
            context_parts.append("SCÉNARIO PERTURBÉ:")
        else:
            context_parts.append("RÉSULTATS DE L'OPTIMISATION:")

        context_parts.append(f"- Statut: {result.status}")
        if result.makespan is not None:
            context_parts.append(f"- Temps de makespan: {result.makespan}")
        else:
            context_parts.append("- Temps de makespan: Non disponible (pas de solution faisable)")

        if result.status in ["OPTIMAL", "FEASIBLE"] and result.makespan is not None:
            context_parts.append("")
            context_parts.append("PLANNING OBTENU:")
            for machine in sorted(result.schedule.keys()):
                machine_ops = result.schedule[machine]
                if machine_ops:
                    ops_str = []
                    for op in machine_ops:
                        ops_str.append(f"{op['job_id']}({op['operation_id']}:{op['start']}-{op['end']})")
                    context_parts.append(f"- Machine {machine}: {' -> '.join(ops_str)}")
                else:
                    context_parts.append(f"- Machine {machine}: Aucune opération assignée")

            context_parts.append("")
            context_parts.append("MÉTRIQUES DE PERFORMANCE:")
            context_parts.append(f"- Temps de complétion par job: {result.job_completion_times}")
            context_parts.append(f"- Retard par job (négatif = en avance): {result.tardiness}")

            # Add comparative metrics if both scenarios are available
            if reference_result is not None and reference_result.status in ["OPTIMAL", "FEASIBLE"] and reference_result.makespan is not None:
                context_parts.append("")
                context_parts.append("ANALYSE COMPARATIVE DES PERFORMANCES:")
                makespan_diff = result.makespan - reference_result.makespan
                context_parts.append(f"- Différence de makespan: {makespan_diff} unités de temps ({'augmentation' if makespan_diff > 0 else 'réduction'} par rapport au scénario de référence)")

                # Compare job completion times
                completion_diff_details = []
                for job_id in set(result.job_completion_times.keys()) | set(reference_result.job_completion_times.keys()):
                    ref_time = reference_result.job_completion_times.get(job_id, 0)
                    pert_time = result.job_completion_times.get(job_id, 0)
                    diff = pert_time - ref_time
                    if diff != 0:
                        completion_diff_details.append(f"Job {job_id}: {diff:+d} unités")
                if completion_diff_details:
                    context_parts.append(f"- Changements de complétion: {', '.join(completion_diff_details)}")

                # Compare tardiness
                tardiness_diff_details = []
                for job_id in set(result.tardiness.keys()) | set(reference_result.tardiness.keys()):
                    ref_tardy = reference_result.tardiness.get(job_id, 0)
                    pert_tardy = result.tardiness.get(job_id, 0)
                    diff = pert_tardy - ref_tardy
                    if diff != 0:
                        tardiness_diff_details.append(f"Job {job_id}: {diff:+d}")
                if tardiness_diff_details:
                    context_parts.append(f"- Changements de retard: {', '.join(tardiness_diff_details)}")

            if evaluation_metrics and "metrics" in evaluation_metrics:
                metrics = evaluation_metrics["metrics"]
                context_parts.append("")
                context_parts.append("MÉTRIQUES D'ÉVALUATION:")
                for key, value in metrics.items():
                    if key == "on_time_delivery_rate" and value is None:
                        context_parts.append(f"- {key}: non calculable (aucune date de livraison fournie)")
                    elif isinstance(value, float):
                        context_parts.append(f"- {key}: {value:.2f}")
                    else:
                        context_parts.append(f"- {key}: {value}")

            # Add comparative evaluation metrics if available
            if evaluation_metrics and "comparative_metrics" in evaluation_metrics:
                comp_metrics = evaluation_metrics["comparative_metrics"]
                if comp_metrics:
                    context_parts.append("")
                    context_parts.append("MÉTRIQUES COMPARATIVES:")
                    for key, value in comp_metrics.items():
                        if isinstance(value, float):
                            context_parts.append(f"- {key}: {value:.2f}")
                        else:
                            context_parts.append(f"- {key}: {value}")

        return "\n".join(context_parts)

    def _rule_based_explanation(self,
                               problem: SupplyChainProblem,
                               result: OptimizationResult,
                               evaluation_metrics: Optional[Dict[str, Any]] = None,
                               reference_result: Optional[OptimizationResult] = None) -> str:
        """Rule-based explanation fallback when LLM is not available"""
        explanation_parts = []

        explanation_parts.append("EXPLICATION DES RÉSULTATS D'OPTIMISATION")
        explanation_parts.append("=" * 50)
        explanation_parts.append("")

        # Problem summary
        explanation_parts.append("1. PROBLÈME TRAITÉ:")
        explanation_parts.append(f"   - {len(problem.jobs)} jobs à planifier sur {len(problem.machines)} machines")
        explanation_parts.append(f"   - Machines: {', '.join(problem.machines)}")

        # Add downtime information if present
        if problem.machine_availability:
            downtime_details = []
            for machine_id, intervals in problem.machine_availability.items():
                if intervals:
                    interval_strs = [f"[{start}-{end}]" for start, end in intervals]
                    downtime_details.append(f"{machine_id}: {', '.join(interval_strs)}")
            if downtime_details:
                explanation_parts.append(f"   - Indisponibilités machines: {', '.join(downtime_details)}")

        explanation_parts.append("")

        # Handle comparative explanation if both scenarios are available
        if reference_result is not None:
            explanation_parts.append("2. ANALYSE COMPARATIVE: SCÉNARIO DE RÉFÉRENCE vs SCÉNARIO PERTURBÉ")
            explanation_parts.append("")

            # Reference scenario summary
            explanation_parts.append("   SCÉNARIO DE RÉFÉRENCE (sans indisponibilités):")
            explanation_parts.append(f"   - Statut: {reference_result.status}")
            if reference_result.makespan is not None:
                explanation_parts.append(f"   - Makespan: {reference_result.makespan} unités de temps")
            else:
                explanation_parts.append("   - Aucune solution faisable trouvée")
            explanation_parts.append("")

            # Perturbed scenario summary
            explanation_parts.append("   SCÉNARIO PERTURBÉ (avec indisponibilités):")
            explanation_parts.append(f"   - Statut: {result.status}")
            if result.makespan is not None:
                explanation_parts.append(f"   - Makespan: {result.makespan} unités de temps")
            else:
                explanation_parts.append("   - Aucune solution faisable trouvée")
            explanation_parts.append("")

            # Comparative analysis if both scenarios have solutions
            if (reference_result.status in ["OPTIMAL", "FEASIBLE"] and reference_result.makespan is not None and
                result.status in ["OPTIMAL", "FEASIBLE"] and result.makespan is not None):

                explanation_parts.append("   COMPARAISON DES PERFORMANCES:")
                makespan_diff = result.makespan - reference_result.makespan
                explanation_parts.append(f"   - Différence de makespan: {makespan_diff:+d} unités de temps")
                if makespan_diff > 0:
                    explanation_parts.append("     → L'indisponibilité des machines augmente le temps de makespan")
                elif makespan_diff < 0:
                    explanation_parts.append("     → L'indisponibilité des machines réduit le temps de makespan (inattendu - vérifier les contraintes)")
                else:
                    explanation_parts.append("     → Aucune différence de makespan entre les scénarios")

                # Job completion times comparison
                explanation_parts.append("   - Temps de complétion par job:")
                for job_id in sorted(set(result.job_completion_times.keys()) | set(reference_result.job_completion_times.keys())):
                    ref_time = reference_result.job_completion_times.get(job_id, 0)
                    pert_time = result.job_completion_times.get(job_id, 0)
                    diff = pert_time - ref_time
                    status = "+" if diff > 0 else ("-" if diff < 0 else "0")
                    explanation_parts.append(f"     * Job {job_id}: référence={ref_time}, perturbé={pert_time} ({status}{abs(diff)})")

                # Tardiness comparison
                explanation_parts.append("   - Retard par job (négatif = en avance):")
                for job_id in sorted(set(result.tardiness.keys()) | set(reference_result.tardiness.keys())):
                    ref_tardy = reference_result.tardiness.get(job_id, 0)
                    pert_tardy = result.tardiness.get(job_id, 0)
                    diff = pert_tardy - ref_tardy
                    status = "+" if diff > 0 else ("-" if diff < 0 else "0")
                    explanation_parts.append(f"     * Job {job_id}: référence={ref_tardy}, perturbé={pert_tardy} ({status}{abs(diff)})")

                explanation_parts.append("")

            elif result.status in ["OPTIMAL", "FEASIBLE"] and result.makespan is not None:
                # Only perturbed scenario has a solution
                explanation_parts.append("   NOTE: Seul le scénario perturbé a produit une solution faisable")
                explanation_parts.append("")
            elif reference_result.status in ["OPTIMAL", "FEASIBLE"] and reference_result.makespan is not None:
                # Only reference scenario has a solution
                explanation_parts.append("   NOTE: Seul le scénario de référence a produit une solution faisable")
                explanation_parts.append("")

            # Schedule details for both scenarios
            explanation_parts.append("3. PLANNINGS DÉTAILLÉS:")
            explanation_parts.append("")

            # Reference scenario schedule
            if reference_result.status in ["OPTIMAL", "FEASIBLE"]:
                explanation_parts.append("   SCÉNARIO DE RÉFÉRENCE:")
                for machine in sorted(reference_result.schedule.keys()):
                    machine_ops = reference_result.schedule[machine]
                    if machine_ops:
                        explanation_parts.append(f"     Machine {machine}:")
                        for op in sorted(machine_ops, key=lambda x: x['start']):
                            explanation_parts.append(
                                f"       - Job {op['job_id']} Opération {op['operation_id']}: "
                                f"{op['start']} -> {op['end']} (durée: {op['duration']})"
                            )
                    else:
                        explanation_parts.append(f"     Machine {machine}: Aucune opération")
                explanation_parts.append("")
            else:
                explanation_parts.append("   SCÉNARIO DE RÉFÉRENCE: Aucune solution faisable")
                explanation_parts.append("")

            # Perturbed scenario schedule
            if result.status in ["OPTIMAL", "FEASIBLE"]:
                explanation_parts.append("   SCÉNARIO PERTURBÉ:")
                for machine in sorted(result.schedule.keys()):
                    machine_ops = result.schedule[machine]
                    if machine_ops:
                        explanation_parts.append(f"     Machine {machine}:")
                        for op in sorted(machine_ops, key=lambda x: x['start']):
                            explanation_parts.append(
                                f"       - Job {op['job_id']} Opération {op['operation_id']}: "
                                f"{op['start']} -> {op['end']} (durée: {op['duration']})"
                            )
                    else:
                        explanation_parts.append(f"     Machine {machine}: Aucune opération")
                explanation_parts.append("")
            else:
                explanation_parts.append("   SCÉNARIO PERTURBÉ: Aucune solution faisable")
                explanation_parts.append("")

        else:
            # Single scenario explanation (original behavior)
            explanation_parts.append("2. RÉSULTATS DE L'OPTIMISATION:")
            explanation_parts.append(f"   - Statut de la solution: {result.status}")
            if result.makespan is not None:
                explanation_parts.append(f"   - Temps de makespan (durée totale): {result.makespan} unités de temps")
            else:
                explanation_parts.append("   - Aucune solution faisable trouvée")
                explanation_parts.append("   - Ceci peut être dû à des contraintes trop strictes ou à un problème de modélisation")
                return "\n".join(explanation_parts)

            explanation_parts.append("")

            # Schedule details
            explanation_parts.append("3. PLANNING OBTENU:")
            for machine in sorted(result.schedule.keys()):
                machine_ops = result.schedule[machine]
                if machine_ops:
                    explanation_parts.append(f"   Machine {machine}:")
                    for op in sorted(machine_ops, key=lambda x: x['start']):
                        explanation_parts.append(
                            f"     - Job {op['job_id']} Opération {op['operation_id']}: "
                            f"{op['start']} -> {op['end']} (durée: {op['duration']})"
                        )
                else:
                    explanation_parts.append(f"   Machine {machine}: Aucune opération")
            explanation_parts.append("")

            # Performance metrics
            explanation_parts.append("4. MÉTRIQUES DE PERFORMANCE:")
            explanation_parts.append(f"   - Temps de complétion des jobs: {result.job_completion_times}")
            explanation_parts.append(f"   - Retard des jobs (valeur négative = en avance): {result.tardiness}")

            total_tardiness = sum(max(0, t) for t in result.tardiness.values())
            on_time_jobs = sum(1 for t in result.tardiness.values() if t <= 0)
            explanation_parts.append(f"   - Jobs en retard: {len(problem.jobs) - on_time_jobs}/{len(problem.jobs)}")
            explanation_parts.append(f"   - Retard total: {total_tardiness} unités de temps")
            explanation_parts.append("")

        # Evaluation metrics if available
        if evaluation_metrics and evaluation_metrics.get("evaluation_possible", False):
            metrics = evaluation_metrics.get("metrics", {})
            explanation_parts.append("5. MÉTRIQUES D'ÉVALUATION:")
            if "utilization_percentage" in metrics:
                explanation_parts.append(f"   - Taux d'utilisation des machines: {metrics['utilization_percentage']:.1f}%")
            if "on_time_delivery_rate" in metrics:
                if metrics["on_time_delivery_rate"] is not None:
                    explanation_parts.append(f"   - Taux de livraison à temps: {metrics['on_time_delivery_rate']*100:.1f}%")
                else:
                    explanation_parts.append(f"   - Taux de livraison à temps: non calculable (aucune date de livraison fournie)")
            if "load_balance_percentage" in metrics:
                explanation_parts.append(f"   - Équilibrage de charge: {metrics['load_balance_percentage']:.1f}% (100 × (1 - écart-type des charges/moyenne des charges))")

            # Add comparative evaluation metrics if available
            if evaluation_metrics.get("comparative_metrics"):
                explanation_parts.append("")
                explanation_parts.append("6. MÉTRIQUES COMPARATIVES:")
                comp_metrics = evaluation_metrics["comparative_metrics"]
                for key, value in comp_metrics.items():
                    if isinstance(value, float):
                        explanation_parts.append(f"   - {key}: {value:.2f}")
                    else:
                        explanation_parts.append(f"   - {key}: {value}")
            explanation_parts.append("")

        # Recommendations
        explanation_parts.append("6. RECOMMANDATIONS ET ANALYSE:")
        if reference_result is not None:
            # Comparative recommendations
            if (reference_result.status in ["OPTIMAL", "FEASIBLE"] and result.status in ["OPTIMAL", "FEASIBLE"] and
                reference_result.makespan is not None and result.makespan is not None):
                makespan_diff = result.makespan - reference_result.makespan
                if makespan_diff > 0:
                    explanation_parts.append("   - L'indisponibilité des machines a un impact négatif sur la performance")
                    explanation_parts.append("   - Recommandation: Considérer des stratégies de atténuation comme la maintenance préventive ou des machines de secours")
                elif makespan_diff < 0:
                    explanation_parts.append("   - Résultat inattendu: l'indisponibilité améliore les performances - vérifier la modélisation des contraintes")
                else:
                    explanation_parts.append("   - Aucune différence de performance détectée entre les scénarios")
            elif result.status in ["OPTIMAL", "FEASIBLE"]:
                explanation_parts.append("   - Le scénario perturbé a produit une solution faisable")
                explanation_parts.append("   - Analyser l'impact des indisponibilités sur le planning détaillé ci-dessus")
            else:
                explanation_parts.append("   - Aucune scénario n'a produit de solution faisable - vérifier la cohérence des contraintes d'indisponibilité")
        else:
            # Single scenario recommendations (original behavior)
            if result.status == "OPTIMAL":
                explanation_parts.append("   - La solution est optimale garantie par le solveur OR-Tools CP-SAT")
            elif result.status == "FEASIBLE":
                explanation_parts.append("   - La solution est faisable mais peut ne pas être optimale")
                explanation_parts.append("   - Considérez augmenter le temps de recherche pour potentiellement améliorer la solution")
            else:
                explanation_parts.append("   - Aucune solution trouvée - vérifiez la cohérence des contraintes")

        # Specific recommendations based on metrics
        if evaluation_metrics and evaluation_metrics.get("evaluation_possible", False):
            metrics = evaluation_metrics.get("metrics", {})
            on_time_delivery_rate = metrics.get("on_time_delivery_rate")
            if on_time_delivery_rate is not None and on_time_delivery_rate < 0.8:
                explanation_parts.append("   - Amélioration suggérée: Réviser les dates de livraison ou ajouter des ressources")
            if metrics.get("load_balance_percentage", 100) < 70:
                explanation_parts.append("   - Amélioration suggérée: Better équilibrer la charge entre les machines")

        explanation_parts.append("")
        explanation_parts.append("7. LIMITES DE L'APPROCHE:")
        explanation_parts.append("   - Le solveur CP-SAT trouve des solutions exactes mais peut être lent sur de très grandes instances")
        explanation_parts.append("   - Pour des problèmes très grands, des approches heuristiques ou des métahéuristiques pourraient être plus appropriées")
        explanation_parts.append("   - Cette implémentation modélise les indisponibilités machine comme des intervalles pendant lesquels aucune opération ne peut être planifiée")
        explanation_parts.append("   - L'analyse comparative permet d'évaluer l'impact des perturbations sur la performance globale")

        return "\n".join(explanation_parts)


# Factory function
def create_explanation_agent(hf_token: str = None) -> ExplanationAgent:
    return ExplanationAgent(hf_token)