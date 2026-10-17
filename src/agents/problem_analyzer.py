"""
Agent for analyzing supply chain problem descriptions in natural language
"""
import os
from typing import Dict, Any
from src.utils.llm import get_llm, call_llm
from langchain_core.prompts import PromptTemplate
from src.models.supply_chain import SupplyChainProblem, Job, Operation
import json
import re


class ProblemAnalyzer:
    """Analyzes natural language descriptions of supply chain problems"""

    def __init__(self, hf_token: str = None):
        self.hf_token = hf_token  # None -> get_llm reads GROQ_API_KEY / HF_TOKEN from env
        self.llm = None
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LLM (Groq or Hugging Face, see src/utils/llm.py)"""
        self.llm = get_llm(self.hf_token, temperature=0.1, max_tokens=4096)

    def analyze_problem(self, problem_description: str) -> Dict[str, Any]:
        """
        Analyze a natural language problem description and extract structured data

        Returns:
            Dictionary containing extracted problem data or error information
        """
        if not self.llm:
            # Fallback to rule-based extraction when LLM is not available
            return self._rule_based_extraction(problem_description)

        # Use LLM for extraction
        prompt_template = PromptTemplate(
            input_variables=["problem_description"],
            template="""
Vous êtes un expert en planification de la chaîne d'approvisionnement.
Analysez la description suivante d'un problème de planification et extrayez les informations structurées.

Description du problème:
{problem_description}

Extrayez les informations suivantes au format JSON:
{{
  "machines": ["liste des identificateurs de machines"],
  "jobs": [
    {{
      "job_id": "identificateur_unique",
      "operations": [
        {{
          "operation_id": "identificateur_unique",
          "job_id": "identificateur_unique",
          "machine_id": "identificateur_de_machine",
          "duration": nombre_entier_positif,
          "order_index": indice_entier_commençant_à_0
        }}
      ],
      "due_date": date_début_entière_optionnelle_ou_null,
      "priority": entier_entre_1_et_5
    }}
  ]
}}

Consignes:
- Chaque machine ne traite qu'une opération à la fois
- Les opérations d'une même tâche doivent respecter leur ordre (order_index séquentiel commençant à 0)
- Les durées doivent être des entiers positifs
- Si une date de livraison n'est pas mentionnée, utilisez null
- Si une priorité n'est pas mentionnée, utilisez 1 (priorité la plus basse)
- Soyez précis et n'inventez pas d'informations qui ne sont pas présentes dans la description

JSON:
"""
        )

        try:
            # Format the prompt and invoke the LLM directly
            prompt = prompt_template.format(problem_description=problem_description)
            result = call_llm(self.llm, prompt)

            # Try to parse JSON from the result
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                extracted_data = json.loads(json_str)
                return {"success": True, "data": extracted_data}
            else:
                return {"success": False, "error": "Could not extract JSON from LLM response"}

        except Exception as e:
            return {"success": False, "error": f"LLM analysis failed: {str(e)}"}

    def _rule_based_extraction(self, problem_description: str) -> Dict[str, Any]:
        """
        Rule-based extraction fallback when LLM is not available
        This is a simplified version for demonstration purposes
        """
        # For demo purposes, return a predefined scenario
        # In a real implementation, this would use regex and rule-based parsing

        # Check if description contains keywords for our demo scenario
        if any(keyword in problem_description.lower() for keyword in
               ["démo", "exemple", "demo", "example", "test"]):

            # Return a predefined job shop scheduling problem
            demo_data = {
                "machines": ["M1", "M2", "M3"],
                "jobs": [
                    {
                        "job_id": "JOB1",
                        "operations": [
                            {
                                "operation_id": "O11",
                                "job_id": "JOB1",
                                "machine_id": "M1",
                                "duration": 3,
                                "order_index": 0
                            },
                            {
                                "operation_id": "O12",
                                "job_id": "JOB1",
                                "machine_id": "M2",
                                "duration": 2,
                                "order_index": 1
                            },
                            {
                                "operation_id": "O13",
                                "job_id": "JOB1",
                                "machine_id": "M3",
                                "duration": 2,
                                "order_index": 2
                            }
                        ],
                        "due_date": 10,
                        "priority": 3
                    },
                    {
                        "job_id": "JOB2",
                        "operations": [
                            {
                                "operation_id": "O21",
                                "job_id": "JOB2",
                                "machine_id": "M2",
                                "duration": 2,
                                "order_index": 0
                            },
                            {
                                "operation_id": "O22",
                                "job_id": "JOB2",
                                "machine_id": "M1",
                                "duration": 4,
                                "order_index": 1
                            },
                            {
                                "operation_id": "O23",
                                "job_id": "JOB2",
                                "machine_id": "M3",
                                "duration": 3,
                                "order_index": 2
                            }
                        ],
                        "due_date": 12,
                        "priority": 2
                    },
                    {
                        "job_id": "JOB3",
                        "operations": [
                            {
                                "operation_id": "O31",
                                "job_id": "JOB3",
                                "machine_id": "M3",
                                "duration": 4,
                                "order_index": 0
                            },
                            {
                                "operation_id": "O32",
                                "job_id": "JOB3",
                                "machine_id": "M2",
                                "duration": 3,
                                "order_index": 1
                            },
                            {
                                "operation_id": "O33",
                                "job_id": "JOB3",
                                "machine_id": "M1",
                                "duration": 5,
                                "order_index": 2
                            }
                        ],
                        "due_date": 15,
                        "priority": 4
                    }
                ]
            }
            return {"success": True, "data": demo_data}
        else:
            # No dataset and no explicit demo request: never invent jobs or durations.
            return {"success": False, "error": (
                "Donnée manquante : aucun jeu de données chargé et aucun LLM disponible pour lire les jobs "
                "dans la description. Chargez un fichier JSON/CSV (barre latérale) ou configurez GROQ_API_KEY."
            )}


# Factory function for creating the analyzer
def create_problem_analyzer(hf_token: str = None) -> ProblemAnalyzer:
    return ProblemAnalyzer(hf_token)