"""
Agent for validating supply chain problem parameters using Pydantic
"""
from typing import Dict, Any, Tuple
from src.models.supply_chain import SupplyChainProblem, OptimizationResult
import traceback


class ValidationAgent:
    """Validates extracted problem parameters using Pydantic models"""

    def __init__(self):
        self.validation_errors = []

    def validate_problem(self, extracted_data: Dict[str, Any]) -> Tuple[bool, Any, list]:
        """
        Validate extracted problem data using Pydantic models

        Args:
            extracted_data: Dictionary containing problem data from analyzer

        Returns:
            Tuple of (is_valid, validated_object_or_none, error_messages)
        """
        self.validation_errors = []

        try:
            # Validate and create SupplyChainProblem instance
            problem = SupplyChainProblem(**extracted_data)
            return True, problem, []

        except Exception as e:
            error_msg = f"Validation failed: {str(e)}"
            self.validation_errors.append(error_msg)
            return False, None, self.validation_errors

    def validate_result(self, result_data: Dict[str, Any]) -> Tuple[bool, Any, list]:
        """
        Validate optimization result data

        Args:
            result_data: Dictionary containing optimization results

        Returns:
            Tuple of (is_valid, validated_object_or_none, error_messages)
        """
        self.validation_errors = []

        try:
            result = OptimizationResult(**result_data)
            return True, result, []

        except Exception as e:
            error_msg = f"Result validation failed: {str(e)}"
            self.validation_errors.append(error_msg)
            return False, None, self.validation_errors


# Factory function
def create_validation_agent() -> ValidationAgent:
    return ValidationAgent()